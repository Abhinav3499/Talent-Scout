from __future__ import annotations

import json
import os
from typing import Any, Dict

from flask import Flask, render_template, request, redirect, url_for, flash, session
from dotenv import load_dotenv

from llm import GeminiClient, build_report_prompt
from db import save_report, init_db, bootstrap_default_admin, create_interview, get_interview, update_interview_answer, complete_interview

load_dotenv()  # Load variables from .env (e.g., GOOGLE_API_KEY)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")


def extract_pdf_text(file_storage) -> str:
    """Extract text from uploaded PDF. Try pypdf first; fallback to OCR if available."""
    # Save to a temp path
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        file_storage.save(tmp.name)
        pdf_path = tmp.name

    text = ""
    try:
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    except Exception:
        text = ""

    # If no text (possibly scanned), try OCR if available
    if not text.strip():
        try:
            from pdf2image import convert_from_path
            import pytesseract

            images = convert_from_path(pdf_path)
            ocr_texts = [pytesseract.image_to_string(img) for img in images]
            text = "\n".join(ocr_texts)
        except Exception:
            pass

    try:
        os.remove(pdf_path)
    except Exception:
        pass

    return text.strip()


def build_question_sets_prompt(cv_text: str, meta: Dict[str, str]) -> str:
    """Prompt to produce categorized question sets in JSON.

    Expected JSON keys:
    - general (5 short questions)
    - technical (8 questions, increasing difficulty)
    - project (5 questions about projects in resume)
    - experience (5 questions about roles/responsibilities)
    """
    name = meta.get("name", "Candidate")
    college = meta.get("college", "")
    email = meta.get("email", "")
    return (
        "Read the candidate resume text and generate interview questions strictly grounded in that text.\n"
        "Return ONLY a JSON object with keys: general, technical, project, experience.\n"
        "Rules:\n"
        "- Do NOT include topics not present or clearly implied by the resume.\n"
        "- If information is missing, ask clarifying questions tied to the resume content (not generic).\n"
        "- general: 5 concise warm-up questions referencing resume content.\n"
        "- technical: 8 tailored questions (concept + practical), matching the stack/skills in the resume.\n"
        "- project: 5 questions about projects explicitly mentioned in the resume.\n"
        "- experience: 5 about responsibilities, impact, challenges referenced in the resume.\n"
        "Output must be valid JSON with arrays of strings for each key.\n\n"
        f"Candidate: {name}, {email}, {college}\n"
        f"Resume Text:\n{cv_text}"
    )


init_db()
bootstrap_default_admin()
gemini = GeminiClient()


@app.route("/")
def index():
    """Clears session and shows the initial questionnaire form."""
    session.clear()
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    """
    Handles the initial form submission, extracts resume text, generates all questions,
    saves them to database, and redirects to the chat interface.
    """
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()
    college = request.form.get("college", "").strip()
    pdf = request.files.get("resume")

    if not all([name, email, college, pdf]):
        flash("Please fill all fields and upload your resume PDF.", "warning")
        return redirect(url_for("index"))

    cv_text = extract_pdf_text(pdf)
    print(f"\n===== Extracted Resume Text for {name} =====\n{cv_text}\n========================================\n")
    if not cv_text:
        flash("Could not extract text from PDF. Please ensure it's a text-based PDF.", "danger")
        return redirect(url_for("index"))

    # Generate all questions upfront
    prompt = build_question_sets_prompt(cv_text, {"name": name, "email": email, "college": college})
    raw_json = gemini.generate(prompt, json_mode=True)
    
    try:
        # Basic JSON cleaning
        s = raw_json.strip()
        if s.startswith("```"):
            s = s.strip("` \n")
            if s.lower().startswith("json"):
                s = s[4:].strip()
        questions_data = json.loads(s)
    except json.JSONDecodeError:
        flash("Failed to parse questions from the LLM. Please try again.", "danger")
        return redirect(url_for("index"))

    # Flatten questions into a single list for the chat session
    all_questions = []
    for category in ["general", "technical", "project", "experience"]:
        for q in questions_data.get(category, []):
            all_questions.append({"category": category, "question": str(q)})

    if not all_questions:
        flash("No questions were generated. The resume might be empty or unsupported.", "warning")
        return redirect(url_for("index"))

    # Save interview to database instead of session
    interview_id = create_interview(name, email, college, cv_text, all_questions)
    
    # Store only the interview ID in session
    session.clear()
    session["interview_id"] = interview_id

    return redirect(url_for("chat"))


@app.route("/chat", methods=["GET", "POST"])
def chat():
    """Handles the interactive chat session, one question at a time."""
    if "interview_id" not in session:
        flash("Your session has expired. Please start over.", "warning")
        return redirect(url_for("index"))

    interview_id = session["interview_id"]
    interview_data = get_interview(interview_id)
    
    if not interview_data:
        flash("Interview not found. Please start over.", "warning")
        return redirect(url_for("index"))

    all_questions = interview_data["questions_json"]
    current_index = interview_data["current_question_index"]
    answers = interview_data["answers_json"]

    # Handle POST request (user submitting an answer)
    if request.method == "POST":
        answer = request.form.get("answer", "").strip()
        if not answer:
            flash("Please provide an answer.", "warning")
            return redirect(url_for("chat"))

        # Save the answer and advance to next question
        next_index = current_index + 1
        update_interview_answer(interview_id, answer, next_index)

        if next_index >= len(all_questions):
            complete_interview(interview_id)
            return redirect(url_for("end_interview"))
        
        return redirect(url_for("chat"))

    # Handle GET request (displaying the current question)
    if current_index >= len(all_questions):
        return redirect(url_for("end_interview"))

    current_question = all_questions[current_index]["question"]
    progress = f"{current_index + 1} / {len(all_questions)}"

    # Build history from questions and answers
    history = []
    for i in range(current_index):
        if i < len(answers):
            history.append({
                "question": all_questions[i]["question"],
                "answer": answers[i],
                "category": all_questions[i]["category"]
            })

    return render_template(
        "chat.html",
        history=history,
        current_question=current_question,
        progress=progress,
    )


@app.route("/end")
def end_interview():
    """
    Finalizes the interview, generates the report, saves it to the DB,
    and shows a thank you message.
    """
    if "interview_id" not in session:
        return redirect(url_for("index"))

    interview_id = session["interview_id"]
    interview_data = get_interview(interview_id)
    
    if not interview_data:
        return redirect(url_for("index"))

    # Build Q&A history for report generation
    questions = interview_data["questions_json"]
    answers = interview_data["answers_json"]
    
    qna_history = []
    for i, answer in enumerate(answers):
        if i < len(questions):
            qna_history.append({
                "question": questions[i]["question"],
                "answer": answer,
                "category": questions[i]["category"]
            })
    
    # Generate the final report using the LLM
    report_prompt = build_report_prompt(interview_data["cv_text"], qna_history)
    report_json_str = gemini.generate(report_prompt, json_mode=True)
    
    try:
        report_data = json.loads(report_json_str)
    except json.JSONDecodeError:
        report_data = {"error": "Failed to generate a valid JSON report.", "raw": report_json_str}

    # Save the complete record to the database
    report_id = save_report(
        candidate_name=interview_data["candidate_name"],
        email=interview_data["email"],
        profile_text=interview_data["cv_text"],
        qna_json=json.dumps(qna_history, indent=2),
        report_json=json.dumps(report_data, indent=2),
    )

    # Clear the session to free up resources and prevent reuse
    session.clear()

    return render_template("end.html", report_id=report_id)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
