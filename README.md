# TalentScout

Candidate screening app with Flask + Bootstrap frontend and Gemini for question generation. Reports can be added next; SQLite helpers exist in `db.py`.

## Features

- Candidate flow: profile summarization from CV text, dynamic screening and technical questions, final report
- Admin login: view reports list and open detailed report (stored in DB)
- SQLite storage with password hashing for admin
- Gemini LLM wrapper with mock mode if `GOOGLE_API_KEY` is not set

## Setup

1. Install Python 3.9+ and create a virtual environment.
2. Install dependencies:

   - google-generativeai
   - flask
   - pypdf
   - pdf2image (optional OCR)
   - pytesseract (optional OCR)

3. Set environment variables (optional for mock mode):
   - `GOOGLE_API_KEY` — your Google Generative AI key

## Run

From the repository root:

```bash
set GOOGLE_API_KEY=YOUR_KEY  # PowerShell: $env:GOOGLE_API_KEY="YOUR_KEY"
pip install -r requirements.txt
python app.py
```

The app will open in your browser at http://localhost:5000. If GOOGLE_API_KEY is missing, mock outputs are used.

## Notes

- Database file `talentscout.db` is created next to the code.
- For production, set a secure admin password using `db.upsert_admin()` once.
