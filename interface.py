"""
Gradio app for TalentScout RAG-like candidate screening.

Flows:
- Candidate: upload/profile text -> generate screening questions -> collect answers -> generate technical questions -> collect answers -> final report.
- Admin: login -> view saved reports with details.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import gradio as gr

from db import (
	bootstrap_default_admin,
	get_report,
	init_db,
	list_reports,
	save_report,
)
from llm import (
	GeminiClient,
	build_profile_prompt,
	build_report_prompt,
	build_screening_questions_prompt,
	build_technical_questions_prompt,
)


# Initialize DB and default admin
init_db()
bootstrap_default_admin()

gemini = GeminiClient()


def _parse_lines_to_list(text: str) -> List[str]:
	return [line.strip("- ") for line in text.splitlines() if line.strip()]


def gen_profile(cv_text: str) -> str:
	prompt = build_profile_prompt(cv_text)
	return gemini.generate(prompt)


def gen_screening(profile: str) -> List[str]:
	prompt = build_screening_questions_prompt(profile)
	txt = gemini.generate(prompt)
	return _parse_lines_to_list(txt)


def gen_technical(profile: str, role: str) -> List[str]:
	prompt = build_technical_questions_prompt(profile, role)
	txt = gemini.generate(prompt)
	return _parse_lines_to_list(txt)


def synth_report(candidate_name: str, email: str, role: str, profile: str, screening_qna: List[Dict[str, Any]], tech_qna: List[Dict[str, Any]]):
	prompt = build_report_prompt(profile, screening_qna, tech_qna, role)
	out = gemini.generate(prompt, json_mode=True)
	try:
		report = json.loads(out)
	except Exception:
		# Fallback shape
		report = {
			"overall_score": 0,
			"strengths": [],
			"risks": [],
			"recommendation": out[:300],
			"notes": out,
		}

	# Persist
	screening_questions = [qa["q"] for qa in screening_qna]
	technical_questions = [qa["q"] for qa in tech_qna]
	report_id = save_report(
		candidate_name=candidate_name,
		email=email,
		profile_text=profile,
		screening_questions=screening_questions,
		technical_questions=technical_questions,
		qna=screening_qna + tech_qna,
		report=report,
	)
	pretty = json.dumps(report, indent=2, ensure_ascii=False)
	return report_id, pretty


# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
	gr.Markdown("# TalentScout — Candidate Screening")

	with gr.Tabs():
		with gr.Tab("Candidate"):
			with gr.Row():
				candidate_name = gr.Textbox(label="Full Name")
				candidate_email = gr.Textbox(label="Email")
				target_role = gr.Textbox(label="Target Role", value="Software Engineer")

			cv_text = gr.Textbox(label="Paste your CV / Resume text", lines=10)
			profile_btn = gr.Button("Summarize Profile")
			profile_out = gr.Textbox(label="Profile Summary", lines=6)

			screening_btn = gr.Button("Generate Screening Questions")
			screening_list = gr.State([])  # List[str]
			screening_out = gr.Textbox(label="Screening Questions", lines=8, interactive=False)

			gr.Markdown("### Screening Answers")
			screening_answers = gr.Dataframe(headers=["Question", "Answer"], datatype=["str", "str"], row_count=(5, "dynamic"))

			tech_btn = gr.Button("Generate Technical Questions")
			tech_list = gr.State([])
			tech_out = gr.Textbox(label="Technical Questions", lines=10, interactive=False)

			gr.Markdown("### Technical Answers")
			tech_answers = gr.Dataframe(headers=["Question", "Answer"], datatype=["str", "str"], row_count=(8, "dynamic"))

			report_btn = gr.Button("Finish & Submit (Admin will review)")
			report_id_out = gr.Number(label="Your Reference ID", interactive=False)
			candidate_msg = gr.Markdown("")

			def on_profile(cv):
				return gen_profile(cv)

			profile_btn.click(on_profile, inputs=[cv_text], outputs=[profile_out])

			def on_screening(profile):
				qs = gen_screening(profile)
				return qs, "\n".join(qs), [[q, ""] for q in qs]

			screening_btn.click(on_screening, inputs=[profile_out], outputs=[screening_list, screening_out, screening_answers])

			def on_tech(profile, role):
				qs = gen_technical(profile, role)
				return qs, "\n".join(qs), [[q, ""] for q in qs]

			tech_btn.click(on_tech, inputs=[profile_out, target_role], outputs=[tech_list, tech_out, tech_answers])

			def on_report(name, email, role, profile, screening_state, screening_df, tech_state, tech_df):
				# Normalize Q&A
				s_qs: List[str] = screening_state or []
				t_qs: List[str] = tech_state or []

				def df_to_qna(df_rows, qs):
					qna = []
					if isinstance(df_rows, list):
						for i, row in enumerate(df_rows):
							q = (qs[i] if i < len(qs) else (row[0] if row else "")).strip()
							a = (row[1] if len(row) > 1 else "").strip()
							if q:
								qna.append({"q": q, "a": a})
					return qna

				screening_qna = df_to_qna(screening_df, s_qs)
				tech_qna = df_to_qna(tech_df, t_qs)
				rep_id, _rep_json = synth_report(name, email, role, profile, screening_qna, tech_qna)
				msg = f"Thank you, {name or 'Candidate'}. Your screening is complete. An admin will review your report. Reference ID: {rep_id}."
				return rep_id, msg

			report_btn.click(
				on_report,
				inputs=[candidate_name, candidate_email, target_role, profile_out, screening_list, screening_answers, tech_list, tech_answers],
				outputs=[report_id_out, candidate_msg],
			)

		with gr.Tab("Admin"):
			admin_user = gr.Textbox(label="Username")
			admin_pass = gr.Textbox(label="Password", type="password")
			login_btn = gr.Button("Login")
			login_state = gr.State(False)

			reports_df = gr.Dataframe(headers=["id", "candidate_name", "email", "overall_score", "created_at"], interactive=False)
			report_id_in = gr.Number(label="Report ID to open")
			open_btn = gr.Button("Open Report")
			admin_report = gr.Code(label="Report Detail (JSON)")

			def do_login(u, p):
				# Lazy import to avoid circular
				from db import verify_admin

				ok = verify_admin(u, p)
				rows = list_reports(100) if ok else []
				return ok, rows

			login_btn.click(do_login, inputs=[admin_user, admin_pass], outputs=[login_state, reports_df])

			def open_report(is_logged_in, rid):
				if not is_logged_in:
					return json.dumps({"error": "Not authorized"}, indent=2)
				try:
					rid_int = int(rid)
				except Exception:
					return json.dumps({"error": "Invalid report id"}, indent=2)
				data = get_report(rid_int)
				if not data:
					return json.dumps({"error": "Not found"}, indent=2)
				return json.dumps(data, indent=2, ensure_ascii=False)

			open_btn.click(open_report, inputs=[login_state, report_id_in], outputs=[admin_report])


def launch():
	demo.launch()


if __name__ == "__main__":
	launch()

