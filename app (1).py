import json
import os

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

# ---------- Setup ----------
load_dotenv()  # reads GROQ_API_KEY from the .env file (local use)

MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def get_api_key():
    """Get the key from .env / environment, or from Streamlit secrets (when hosted)."""
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


def read_pdf(uploaded_file):
    """Extract all text from an uploaded PDF."""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text.strip()


def analyze_resume(resume_text, job_description, api_key):
    """Send resume + job description to Groq and get a JSON analysis back."""
    client = Groq(api_key=api_key)

    prompt = f"""
Compare the RESUME with the JOB DESCRIPTION below.

Return ONLY valid JSON in exactly this format:
{{
  "match_score": <integer from 0 to 100>,
  "matching_skills": ["skill1", "skill2"],
  "missing_skills": ["skill or keyword 1", "skill or keyword 2"],
  "suggestions": ["suggestion 1", "suggestion 2", "suggestion 3", "suggestion 4", "suggestion 5"]
}}

Rules:
- match_score reflects how well the resume matches the job description.
- matching_skills: skills/keywords found in BOTH the resume and the job description.
- missing_skills: important skills/keywords in the job description that are NOT in the resume.
- suggestions: exactly 5 clear, practical suggestions to improve the resume for this job.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are an expert ATS (Applicant Tracking System) and career coach. You reply only in JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ---------- UI ----------
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="centered")

st.title("📄 AI Resume Analyzer & ATS Score Checker")
st.write(
    "Upload your resume and paste a job description. "
    "The AI will tell you how well they match and how to improve."
)

api_key = get_api_key()
if not api_key:
    st.error("GROQ_API_KEY not found. Add it to a .env file (see steps).")
    st.stop()

uploaded_resume = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
job_description = st.text_area(
    "Paste the job description here", height=250,
    placeholder="Paste the full job description...",
)

if st.button("Analyze Resume", type="primary"):
    if uploaded_resume is None:
        st.warning("Please upload your resume as a PDF.")
    elif not job_description.strip():
        st.warning("Please paste the job description.")
    else:
        with st.spinner("Reading your resume and analyzing..."):
            try:
                resume_text = read_pdf(uploaded_resume)
                if not resume_text:
                    st.error(
                        "Could not read text from this PDF (it may be a scanned image). "
                        "Try a text-based PDF."
                    )
                    st.stop()

                result = analyze_resume(resume_text, job_description, api_key)
            except json.JSONDecodeError:
                st.error("The AI returned an unexpected format. Please click Analyze again.")
                st.stop()
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()

        # ---------- Output ----------
        score = int(result.get("match_score", 0))
        st.divider()
        st.subheader("Match Score")
        st.metric("ATS Match Score", f"{score} / 100")
        st.progress(min(max(score, 0), 100) / 100)

        if score >= 75:
            st.success("Great match! Your resume fits this job well.")
        elif score >= 50:
            st.info("Decent match. A few improvements can boost your chances.")
        else:
            st.warning("Low match. Your resume needs work for this role.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("✅ Matching Skills")
            for skill in result.get("matching_skills", []):
                st.write(f"- {skill}")
        with col2:
            st.subheader("❌ Missing Skills / Keywords")
            for skill in result.get("missing_skills", []):
                st.write(f"- {skill}")

        st.subheader("💡 5 Suggestions to Improve Your Resume")
        for i, tip in enumerate(result.get("suggestions", []), start=1):
            st.write(f"**{i}.** {tip}")
