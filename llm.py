"""
Gemini LLM wrapper and prompt utilities for TalentScout.

Uses Google Generative AI (Gemini) via the 'google-generativeai' package.
Set environment variable GOOGLE_API_KEY with your key.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List


class GeminiClient:
    def __init__(self, model: str = "gemini-1.5-flash", api_key: str | None = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = model
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set. Please add it to your .env file.")

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        
        generation_config = {"response_mime_type": "application/json"} if json_mode else {}
        
        model = genai.GenerativeModel(
            self.model_name,
            generation_config=generation_config,
        )
        
        if system:
            prompt = f"System: {system}\n\nUser: {prompt}"
            
        resp = model.generate_content(prompt)
        return resp.text

def build_report_prompt(cv_text: str, qna_history: List[Dict[str, Any]]) -> str:
    """Builds a prompt to generate a final candidate report from the conversation."""
    qna_formatted = "\n".join(
        [f"Q: {item['question']}\nA: {item['answer']}" for item in qna_history]
    )
    return (
        "You are a senior hiring manager. Based on the candidate's resume and their answers "
        "to the screening questions, generate a concise hiring report.\n"
        "The report must be a JSON object with the following keys:\n"
        '- "overall_score": An integer from 0 to 100.\n'
        '- "strengths": A list of strings highlighting key strengths.\n'
        '- "weaknesses": A list of strings highlighting potential weaknesses or risks.\n'
        '- "recommendation": A string (e.g., "Strongly Recommend", "Recommend", "Consider", "Do Not Proceed").\n'
        '- "summary": A brief paragraph summarizing the candidate\'s profile and performance.\n\n'
        f"Resume Text:\n{cv_text}\n\n"
        f"Interview Q&A:\n{qna_formatted}\n\n"
        "Now, provide the JSON report."
    )

