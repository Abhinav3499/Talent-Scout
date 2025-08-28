"""
SQLite database helpers for TalentScout.

Tables:
- admin(username TEXT PRIMARY KEY, pass_hash TEXT, salt TEXT, created_at TIMESTAMP)
- reports(id INTEGER PRIMARY KEY AUTOINCREMENT, candidate_name TEXT, email TEXT,
		  profile_text TEXT, screening_questions TEXT, technical_questions TEXT,
		  qna_json TEXT, report_json TEXT, overall_score REAL,
		  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
"""
from __future__ import annotations

import base64
import json
import os
import sqlite3
import hashlib
import hmac
from contextlib import closing
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = os.path.join(os.path.dirname(__file__), 'talentscout.db')


def _connect() -> sqlite3.Connection:
	conn = sqlite3.connect(DB_PATH)
	conn.row_factory = sqlite3.Row
	return conn


def init_db() -> None:
	with closing(_connect()) as conn:
		with conn:
			conn.execute(
				"""
				CREATE TABLE IF NOT EXISTS interviews (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					candidate_name TEXT,
					email TEXT,
					college TEXT,
					cv_text TEXT,
					questions_json TEXT,
					current_question_index INTEGER DEFAULT 0,
					answers_json TEXT DEFAULT '[]',
					is_completed INTEGER DEFAULT 0,
					created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
				)
				"""
			)
			conn.execute(
				"""
				CREATE TABLE IF NOT EXISTS admin (
					username TEXT PRIMARY KEY,
					pass_hash TEXT NOT NULL,
					salt TEXT NOT NULL,
					created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
				)
				"""
			)
			conn.execute(
				"""
				CREATE TABLE IF NOT EXISTS reports (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					candidate_name TEXT,
					email TEXT,
					profile_text TEXT,
					screening_questions TEXT,
					technical_questions TEXT,
					qna_json TEXT,
					report_json TEXT,
					overall_score REAL,
					created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
				)
				"""
			)


# Password hashing utilities (PBKDF2-HMAC-SHA256)
def _hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
	if salt is None:
		salt = os.urandom(16)
	pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 200_000)
	return base64.b64encode(pwd_hash).decode('utf-8'), base64.b64encode(salt).decode('utf-8')


def _verify_password(password: str, stored_hash_b64: str, salt_b64: str) -> bool:
	salt = base64.b64decode(salt_b64)
	calc_hash_b64, _ = _hash_password(password, salt)
	return hmac.compare_digest(calc_hash_b64, stored_hash_b64)


def upsert_admin(username: str, password: str) -> None:
	pwd_hash_b64, salt_b64 = _hash_password(password)
	with closing(_connect()) as conn:
		with conn:
			conn.execute(
				"INSERT INTO admin(username, pass_hash, salt) VALUES(?, ?, ?)\n"
				"ON CONFLICT(username) DO UPDATE SET pass_hash=excluded.pass_hash, salt=excluded.salt",
				(username, pwd_hash_b64, salt_b64),
			)


def verify_admin(username: str, password: str) -> bool:
	with closing(_connect()) as conn:
		cur = conn.execute("SELECT pass_hash, salt FROM admin WHERE username=?", (username,))
		row = cur.fetchone()
		if not row:
			return False
		return _verify_password(password, row["pass_hash"], row["salt"])


def create_interview(candidate_name: str, email: str, college: str, cv_text: str, questions: List[Dict[str, Any]]) -> int:
	"""Creates a new interview session and returns its ID."""
	with closing(_connect()) as conn:
		with conn:
			cur = conn.execute(
				"""
				INSERT INTO interviews(candidate_name, email, college, cv_text, questions_json)
				VALUES (?, ?, ?, ?, ?)
				""",
				(candidate_name, email, college, cv_text, json.dumps(questions, ensure_ascii=False)),
			)
			return int(cur.lastrowid)


def get_interview(interview_id: int) -> Optional[Dict[str, Any]]:
	"""Gets interview data by ID."""
	with closing(_connect()) as conn:
		cur = conn.execute("SELECT * FROM interviews WHERE id=?", (interview_id,))
		row = cur.fetchone()
		if not row:
			return None
		data = dict(row)
		data["questions_json"] = json.loads(data["questions_json"]) if data["questions_json"] else []
		data["answers_json"] = json.loads(data["answers_json"]) if data["answers_json"] else []
		return data


def update_interview_answer(interview_id: int, answer: str, next_question_index: int) -> None:
	"""Adds an answer to the interview and updates the current question index."""
	with closing(_connect()) as conn:
		with conn:
			# Get current answers
			cur = conn.execute("SELECT answers_json FROM interviews WHERE id=?", (interview_id,))
			row = cur.fetchone()
			if not row:
				return
			
			current_answers = json.loads(row["answers_json"]) if row["answers_json"] else []
			current_answers.append(answer)
			
			# Update with new answer and question index
			conn.execute(
				"UPDATE interviews SET answers_json=?, current_question_index=? WHERE id=?",
				(json.dumps(current_answers, ensure_ascii=False), next_question_index, interview_id),
			)


def complete_interview(interview_id: int) -> None:
	"""Marks an interview as completed."""
	with closing(_connect()) as conn:
		with conn:
			conn.execute("UPDATE interviews SET is_completed=1 WHERE id=?", (interview_id,))


def save_report(
    candidate_name: str, email: str, profile_text: str, qna_json: str, report_json: str
) -> int:
    """Saves the full candidate report to the database."""
    with closing(_connect()) as conn:
        with conn:
            cur = conn.execute(
                """
                INSERT INTO reports(
                    candidate_name, email, profile_text,
                    qna_json, report_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    candidate_name,
                    email,
                    profile_text,
                    qna_json,
                    report_json,
                ),
            )
            return int(cur.lastrowid)


def list_reports(limit: int = 100) -> List[Dict[str, Any]]:
	with closing(_connect()) as conn:
		cur = conn.execute(
			"SELECT id, candidate_name, email, overall_score, created_at FROM reports ORDER BY created_at DESC LIMIT ?",
			(limit,),
		)
		return [dict(row) for row in cur.fetchall()]


def get_report(report_id: int) -> Optional[Dict[str, Any]]:
	with closing(_connect()) as conn:
		cur = conn.execute("SELECT * FROM reports WHERE id=?", (report_id,))
		row = cur.fetchone()
		if not row:
			return None
		data = dict(row)
		# Deserialize JSON fields
		for key in ("screening_questions", "technical_questions", "qna_json", "report_json"):
			if data.get(key):
				data[key] = json.loads(data[key])
		return data


def bootstrap_default_admin() -> None:
	with closing(_connect()) as conn:
		cur = conn.execute("SELECT COUNT(1) AS c FROM admin")
		if cur.fetchone()["c"] == 0:
			upsert_admin("admin", "admin")

