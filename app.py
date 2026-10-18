"""CV analyzer: Azure Document Intelligence (OCR/layout) + TF-IDF match + Foundry model feedback + Blob storage."""

import os
import re

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from openai import AzureOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "png", "jpg", "jpeg"}
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DOCINTEL_ENDPOINT = os.getenv("AZURE_ENDPOINT")
DOCINTEL_KEY = os.getenv("AZURE_KEY")
STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER_NAME", "qualified-resumes")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.65"))

if not DOCINTEL_ENDPOINT or not DOCINTEL_KEY:
    raise ValueError("AZURE_ENDPOINT and AZURE_KEY must be set.")
if not STORAGE_CONNECTION_STRING:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING must be set.")
if not (OPENAI_ENDPOINT and OPENAI_KEY and OPENAI_DEPLOYMENT):
    print("Warning: Foundry model settings incomplete. AI feedback will be unavailable.")

document_intelligence_client = DocumentIntelligenceClient(
    endpoint=DOCINTEL_ENDPOINT, credential=AzureKeyCredential(DOCINTEL_KEY)
)
openai_client = (
    AzureOpenAI(api_key=OPENAI_KEY, api_version=OPENAI_API_VERSION, azure_endpoint=OPENAI_ENDPOINT)
    if OPENAI_ENDPOINT and OPENAI_KEY
    else None
)

SECTION_HEADINGS = ["work experience", "skills", "education", "projects", "certifications"]


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def analyze_document(file_path):
    with open(file_path, "rb") as f:
        poller = document_intelligence_client.begin_analyze_document("prebuilt-layout", body=f.read())
    result = poller.result()
    cleaned_text = clean_up_text(result.content or "")
    return cleaned_text, extract_sections(cleaned_text)


def clean_up_text(text):
    unique_lines, seen = [], set()
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 3 or line in seen:
            continue
        seen.add(line)
        unique_lines.append(line)
    return "\n".join(unique_lines)


def extract_sections(cleaned_text):
    headings = "|".join(SECTION_HEADINGS)
    pattern = rf"^(?P<heading>{headings})\s*$(?P<content>[\s\S]*?)(?=^(?:{headings})\s*$|\Z)"
    return {
        m.group("heading").strip().title(): m.group("content").strip()
        for m in re.finditer(pattern, cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
    }


def compare_texts(resume_text, job_desc_text):
    matrix = TfidfVectorizer(stop_words="english").fit_transform([resume_text, job_desc_text])
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])


def upload_to_blob(file_path, filename):
    service = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)
    container = service.get_container_client(BLOB_CONTAINER)
    if not container.exists():
        container.create_container()
    with open(file_path, "rb") as data:
        container.upload_blob(name=filename, data=data, overwrite=True)


def generate_feedback(resume_text, job_desc_text, sections=None):
    if openai_client is None or not OPENAI_DEPLOYMENT:
        return "Foundry model settings are missing. Cannot generate AI feedback."
    resume_block = (
        "\n".join(f"[{h}]\n{c}\n" for h, c in sections.items()) if sections else resume_text
    )
    user_prompt = (
        f"Resume:\n{resume_block}\n\nJob Description:\n{job_desc_text}\n\n"
        "Give specific, actionable suggestions to improve this resume for the job posting: "
        "missing or underemphasized skills, structure/readability, and where to quantify achievements."
    )
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert career advisor and resume reviewer. Provide exactly four "
                        "bullet points of feedback. Each must be clear, specific and actionable, "
                        "with examples where relevant."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=4096,
        )
        return (response.choices[0].message.content or "").strip() or "No feedback generated."
    except Exception as e:
        return f"Error generating AI feedback: {e}"


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "GET":
        return render_template("upload.html")

    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    try:
        extracted_text, sections = analyze_document(file_path)
        job_description = request.form.get("job_description", "").strip()
        similarity = status = upload_status = feedback = None

        if job_description:
            similarity = compare_texts(extracted_text, job_description)
            feedback = generate_feedback(extracted_text, job_description, sections)
            if similarity >= MATCH_THRESHOLD:
                status = "Qualified"
                upload_to_blob(file_path, filename)
                upload_status = "Your resume has been uploaded to our database."
            else:
                status = "Not Qualified"
                upload_status = "Your resume requires improvements to better match the job requirements."
    finally:
        os.remove(file_path)

    return render_template(
        "result.html",
        text=extracted_text,
        similarity=f"{similarity * 100:.1f}%" if similarity is not None else None,
        status=status,
        upload_status=upload_status,
        feedback=feedback,
    )


if __name__ == "__main__":
    app.run(debug=False)
