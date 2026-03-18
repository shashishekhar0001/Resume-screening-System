from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, Form
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_skills(text):
    skills_list = [
        "python", "java", "aws", "docker", "kubernetes",
        "sql", "machine learning", "data analysis",
        "react", "node", "api", "tensorflow", "pandas"
    ]

    text = text.lower()
    found_skills = []

    for skill in skills_list:
        if skill in text:
            found_skills.append(skill)

    return found_skills

# Extract text from PDF
def extract_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


# Calculate similarity
def calculate_score(resume_text, job_desc):
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform([resume_text, job_desc])
    score = cosine_similarity(vectors[0], vectors[1])
    return round(score[0][0] * 100, 2)

# API endpoint
@app.post("/match-multiple")
async def match_multiple(files: list[UploadFile], job_desc: str = Form(...)):
    results = []

    for file in files:
        content = await file.read()

        filename = file.filename
        temp_path = f"temp_{filename}"

        with open(temp_path, "wb") as f:
            f.write(content)

        text = extract_text(temp_path)
        score = calculate_score(text, job_desc)

        resume_skills = extract_skills(text)
        jd_skills = extract_skills(job_desc)

        matching_skills = list(set(resume_skills) & set(jd_skills))

        results.append({
            "name": filename,
            "score": score,
            "skills": resume_skills,
            "matched": matching_skills
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return {"ranking": results}