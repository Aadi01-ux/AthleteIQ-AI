"""
AthleteIQ AI – Athlete Performance Analysis Assistant
======================================================
A multi-agent AI web application powered by Groq API, Flask, Pandas,
Chart.js, and SQLite for comprehensive athlete performance analysis.

Agents:
  1. Athlete Data Agent      – RAG over CSV data, SQLite retrieval
  2. Performance Analysis Agent – Trends, strengths, weaknesses
  3. Training Recommendation Agent – Weekly plans, exercises, recovery
  4. Coaching Insights Agent – Summaries & downloadable reports

Run:  python app.py
URL:  http://127.0.0.1:5000
"""

import os
import io
import csv
import json
import sqlite3
import datetime
import textwrap

import pandas as pd
from flask import (
    Flask, request, jsonify, g, send_file
)
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()  # Load .env file if present

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
DB_PATH      = os.getenv("DB_PATH", "athleteiq.db")
CSV_PATH     = os.getenv("CSV_PATH", "athletes.csv")
DEMO_MODE    = not bool(GROQ_API_KEY)          # True when no API key provided

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024   # 5 MB upload limit

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db():
    """Return a per-request SQLite connection (stored on Flask's g object)."""
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row   # rows behave like dicts
    return g.db


@app.teardown_appcontext
def close_db(exc=None):
    """Close DB connection at end of each request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Create tables and seed data from CSV on first run."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # --- athletes master table ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS athletes (
            athlete_id   TEXT PRIMARY KEY,
            name         TEXT NOT NULL,
            sport        TEXT,
            position     TEXT,
            age          INTEGER,
            weight_kg    REAL,
            height_cm    REAL
        )
    """)

    # --- training sessions / metrics table ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            athlete_id       TEXT,
            session_date     TEXT,
            session_type     TEXT,
            duration_min     REAL,
            distance_km      REAL,
            avg_heart_rate   REAL,
            max_heart_rate   REAL,
            calories_burned  REAL,
            sprint_speed_kmh REAL,
            vo2_max          REAL,
            strength_score   REAL,
            agility_score    REAL,
            endurance_score  REAL,
            recovery_score   REAL,
            match_goals      INTEGER DEFAULT 0,
            match_assists    INTEGER DEFAULT 0,
            match_rating     REAL    DEFAULT 0,
            injuries         INTEGER DEFAULT 0,
            notes            TEXT,
            FOREIGN KEY (athlete_id) REFERENCES athletes(athlete_id)
        )
    """)

    # --- chat history table ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            athlete_id   TEXT,
            agent        TEXT,
            role         TEXT,
            content      TEXT,
            created_at   TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.commit()

    # Seed data from CSV if sessions table is empty
    cur.execute("SELECT COUNT(*) FROM sessions")
    if cur.fetchone()[0] == 0 and os.path.exists(CSV_PATH):
        _seed_from_csv(conn, CSV_PATH)

    conn.close()


def _seed_from_csv(conn, path):
    """Parse CSV and insert rows into athletes + sessions tables."""
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        cur = conn.cursor()
        for _, row in df.iterrows():
            # Upsert athlete master record
            cur.execute("""
                INSERT OR IGNORE INTO athletes
                    (athlete_id, name, sport, position, age, weight_kg, height_cm)
                VALUES (?,?,?,?,?,?,?)
            """, (
                str(row.get("athlete_id", "")),
                str(row.get("name", "Unknown")),
                str(row.get("sport", "")),
                str(row.get("position", "")),
                int(row.get("age", 0)),
                float(row.get("weight_kg", 0)),
                float(row.get("height_cm", 0)),
            ))

            # Insert session
            cur.execute("""
                INSERT INTO sessions
                    (athlete_id, session_date, session_type, duration_min,
                     distance_km, avg_heart_rate, max_heart_rate, calories_burned,
                     sprint_speed_kmh, vo2_max, strength_score, agility_score,
                     endurance_score, recovery_score, match_goals, match_assists,
                     match_rating, injuries, notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                str(row.get("athlete_id", "")),
                str(row.get("session_date", "")),
                str(row.get("session_type", "")),
                float(row.get("duration_min", 0)),
                float(row.get("distance_km", 0)),
                float(row.get("avg_heart_rate", 0)),
                float(row.get("max_heart_rate", 0)),
                float(row.get("calories_burned", 0)),
                float(row.get("sprint_speed_kmh", 0)),
                float(row.get("vo2_max", 0)),
                float(row.get("strength_score", 0)),
                float(row.get("agility_score", 0)),
                float(row.get("endurance_score", 0)),
                float(row.get("recovery_score", 0)),
                int(row.get("match_goals", 0)),
                int(row.get("match_assists", 0)),
                float(row.get("match_rating", 0)),
                int(row.get("injuries", 0)),
                str(row.get("notes", "")),
            ))
        conn.commit()
    except Exception as e:
        print(f"[WARN] CSV seeding failed: {e}")

# ---------------------------------------------------------------------------
# Agent 1 – Athlete Data Agent (RAG over SQLite / CSV)
# ---------------------------------------------------------------------------

class AthleteDataAgent:
    """
    Retrieves structured athlete data from SQLite.
    Acts as the 'retrieval' layer for the multi-agent pipeline.
    """

    @staticmethod
    def get_all_athletes():
        """Return list of all athlete master records."""
        db = get_db()
        rows = db.execute("SELECT * FROM athletes ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def get_athlete_summary(athlete_id: str) -> dict:
        """
        Build a rich summary dict for one athlete by aggregating
        session stats with Pandas.
        """
        db  = get_db()
        ath = db.execute(
            "SELECT * FROM athletes WHERE athlete_id = ?", (athlete_id,)
        ).fetchone()
        if not ath:
            return {}

        rows = db.execute(
            "SELECT * FROM sessions WHERE athlete_id = ? ORDER BY session_date",
            (athlete_id,)
        ).fetchall()

        sessions = [dict(r) for r in rows]
        if not sessions:
            return {"athlete": dict(ath), "sessions": []}

        df = pd.DataFrame(sessions)

        # Aggregate key metrics
        summary = {
            "athlete":            dict(ath),
            "total_sessions":     len(df),
            "total_duration_min": round(df["duration_min"].sum(), 1),
            "total_distance_km":  round(df["distance_km"].sum(), 1),
            "total_calories":     round(df["calories_burned"].sum(), 1),
            "avg_heart_rate":     round(df["avg_heart_rate"].mean(), 1),
            "max_heart_rate":     int(df["max_heart_rate"].max()),
            "avg_vo2_max":        round(df["vo2_max"].mean(), 1),
            "avg_strength":       round(df["strength_score"].mean(), 1),
            "avg_agility":        round(df["agility_score"].mean(), 1),
            "avg_endurance":      round(df["endurance_score"].mean(), 1),
            "avg_recovery":       round(df["recovery_score"].mean(), 1),
            "total_injuries":     int(df["injuries"].sum()),
            "session_types":      df["session_type"].value_counts().to_dict(),
            # Only include match sessions for rating
            "avg_match_rating":   round(
                df[df["match_rating"] > 0]["match_rating"].mean(), 2
            ) if (df["match_rating"] > 0).any() else 0,
            "total_goals":        int(df["match_goals"].sum()),
            "total_assists":      int(df["match_assists"].sum()),
            "recent_sessions":    sessions[-5:],   # last 5 sessions
            "all_sessions":       sessions,
        }
        return summary

    @staticmethod
    def get_chart_data(athlete_id: str) -> dict:
        """Return time-series data formatted for Chart.js."""
        db   = get_db()
        rows = db.execute(
            """SELECT session_date, session_type, duration_min, avg_heart_rate,
                      calories_burned, endurance_score, strength_score,
                      agility_score, recovery_score, vo2_max
               FROM sessions WHERE athlete_id = ? ORDER BY session_date""",
            (athlete_id,)
        ).fetchall()

        if not rows:
            return {}

        df     = pd.DataFrame([dict(r) for r in rows])
        labels = df["session_date"].tolist()

        return {
            "labels":          labels,
            "heart_rate":      df["avg_heart_rate"].tolist(),
            "calories":        df["calories_burned"].tolist(),
            "endurance":       df["endurance_score"].tolist(),
            "strength":        df["strength_score"].tolist(),
            "agility":         df["agility_score"].tolist(),
            "recovery":        df["recovery_score"].tolist(),
            "vo2_max":         df["vo2_max"].tolist(),
            "session_types":   df["session_type"].tolist(),
            "duration":        df["duration_min"].tolist(),
        }

    @staticmethod
    def get_comparison_data() -> dict:
        """Return radar-chart data for all athletes (latest scores)."""
        db   = get_db()
        aths = db.execute("SELECT * FROM athletes").fetchall()
        data = {}
        for ath in aths:
            aid  = ath["athlete_id"]
            rows = db.execute(
                """SELECT AVG(strength_score) s, AVG(agility_score) a,
                          AVG(endurance_score) e, AVG(recovery_score) r,
                          AVG(vo2_max) v
                   FROM sessions WHERE athlete_id = ?""", (aid,)
            ).fetchone()
            if rows and rows["s"]:
                data[ath["name"]] = {
                    "strength":  round(rows["s"], 1),
                    "agility":   round(rows["a"], 1),
                    "endurance": round(rows["e"], 1),
                    "recovery":  round(rows["r"], 1),
                    "vo2_max":   round(rows["v"], 1),
                }
        return data

# ---------------------------------------------------------------------------
# Agent 2 – Performance Analysis Agent
# ---------------------------------------------------------------------------

class PerformanceAnalysisAgent:
    """
    Analyses an athlete's data to surface trends, strengths, and weaknesses.
    Calls the LLM for narrative insight; falls back to rule-based analysis
    in demo mode.
    """

    @staticmethod
    def analyse(athlete_id: str, user_question: str = "") -> str:
        summary = AthleteDataAgent.get_athlete_summary(athlete_id)
        if not summary:
            return "No data found for this athlete."

        context = _build_context_string(summary)
        prompt  = f"""You are an expert sports performance analyst.
Below is the performance data for athlete {summary['athlete']['name']} ({summary['athlete']['sport']}).

{context}

{"User question: " + user_question if user_question else "Provide a comprehensive performance analysis."}

Your analysis must cover:
1. Key performance trends (improving / declining metrics)
2. Top 3 strengths
3. Top 3 areas needing improvement
4. Risk flags (e.g. high injury count, low recovery scores)
5. One concrete, evidence-based recommendation

Be specific, data-driven, and concise. Use bullet points where helpful."""

        if DEMO_MODE:
            return _demo_performance_analysis(summary)

        return _call_groq(prompt, _demo_performance_analysis, summary)

# ---------------------------------------------------------------------------
# Agent 3 – Training Recommendation Agent
# ---------------------------------------------------------------------------

class TrainingRecommendationAgent:
    """
    Generates a personalised weekly training plan based on athlete data
    and identified weaknesses from the performance analysis.
    """

    @staticmethod
    def recommend(athlete_id: str, focus_area: str = "") -> str:
        summary = AthleteDataAgent.get_athlete_summary(athlete_id)
        if not summary:
            return "No data found for this athlete."

        context = _build_context_string(summary)
        sport   = summary["athlete"]["sport"]
        name    = summary["athlete"]["name"]
        prompt  = f"""You are an elite sports conditioning coach specialising in {sport}.
Below is data for {name}.

{context}

{"Focus area requested by coach: " + focus_area if focus_area else ""}

Generate a detailed 7-day training plan that:
1. Addresses their weakest metrics (use the data above)
2. Maintains their strongest attributes
3. Respects recovery needs (recovery score: {summary['avg_recovery']}/100)
4. Is specific to {sport}

Format:
- Day-by-day schedule (Day 1 – Day 7)
- Each day: session type, duration, key exercises/drills, intensity level
- Friday or Saturday: rest or active recovery
- End with 3 nutrition & recovery tips
Keep it practical and actionable."""

        if DEMO_MODE:
            return _demo_training_plan(summary)

        return _call_groq(prompt, _demo_training_plan, summary)

# ---------------------------------------------------------------------------
# Agent 4 – Coaching Insights Agent
# ---------------------------------------------------------------------------

class CoachingInsightsAgent:
    """
    Generates a coaching summary report suitable for sharing
    with coaching staff. Also builds the downloadable text report.
    """

    @staticmethod
    def insights(athlete_id: str, extra_notes: str = "") -> str:
        summary = AthleteDataAgent.get_athlete_summary(athlete_id)
        if not summary:
            return "No data found for this athlete."

        context = _build_context_string(summary)
        name    = summary["athlete"]["name"]
        sport   = summary["athlete"]["sport"]
        prompt  = f"""You are a head coach preparing a professional performance report for {name} ({sport}).

{context}

{"Coach notes: " + extra_notes if extra_notes else ""}

Write a formal coaching insights report with the following sections:
1. **Executive Summary** – 2-3 sentence overview
2. **Performance Highlights** – notable achievements from the data
3. **Areas of Concern** – honest assessment of weaknesses
4. **Tactical Observations** – sport-specific insights
5. **Short-term Goals** (next 4 weeks)
6. **Long-term Development Plan** (3-6 months)
7. **Coach Recommendation** – one clear action item

Use professional language suitable for a sports organisation report."""

        if DEMO_MODE:
            return _demo_coaching_insights(summary)

        return _call_groq(prompt, _demo_coaching_insights, summary)

    @staticmethod
    def generate_report(athlete_id: str) -> str:
        """Build a plain-text downloadable performance report."""
        summary  = AthleteDataAgent.get_athlete_summary(athlete_id)
        if not summary:
            return "No data found."
        ath      = summary["athlete"]
        now      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        insights = CoachingInsightsAgent.insights(athlete_id)
        analysis = PerformanceAnalysisAgent.analyse(athlete_id)

        report = textwrap.dedent(f"""
        ╔══════════════════════════════════════════════════════════════╗
        ║          AthleteIQ AI – Performance Report                  ║
        ╚══════════════════════════════════════════════════════════════╝

        Generated : {now}
        Athlete   : {ath['name']}
        Sport     : {ath['sport']} – {ath['position']}
        Age       : {ath['age']}  |  Weight: {ath['weight_kg']} kg  |  Height: {ath['height_cm']} cm

        ─────────────────────────────────────────────────────────────
        KEY METRICS SUMMARY
        ─────────────────────────────────────────────────────────────
        Total Sessions   : {summary['total_sessions']}
        Total Distance   : {summary['total_distance_km']} km
        Total Calories   : {summary['total_calories']} kcal
        Avg Heart Rate   : {summary['avg_heart_rate']} bpm
        Peak Heart Rate  : {summary['max_heart_rate']} bpm
        Avg VO₂ Max      : {summary['avg_vo2_max']}
        Avg Strength     : {summary['avg_strength']}/100
        Avg Agility      : {summary['avg_agility']}/100
        Avg Endurance    : {summary['avg_endurance']}/100
        Avg Recovery     : {summary['avg_recovery']}/100
        Total Injuries   : {summary['total_injuries']}
        Avg Match Rating : {summary['avg_match_rating']}

        ─────────────────────────────────────────────────────────────
        PERFORMANCE ANALYSIS
        ─────────────────────────────────────────────────────────────
        {analysis}

        ─────────────────────────────────────────────────────────────
        COACHING INSIGHTS
        ─────────────────────────────────────────────────────────────
        {insights}

        ─────────────────────────────────────────────────────────────
        AthleteIQ AI  |  Powered by Groq  |  {now}
        ─────────────────────────────────────────────────────────────
        """)
        return report.strip()

# ---------------------------------------------------------------------------
# Multi-Agent Workflow Orchestrator
# ---------------------------------------------------------------------------

class AthleteIQWorkflow:
    """
    Chains all four agents in sequence:
      Data → Analysis → Recommendations → Coaching Insights
    Returns a structured dict with each agent's output.
    """

    @staticmethod
    def run(athlete_id: str, user_query: str = "") -> dict:
        # Step 1 – Retrieve data (Agent 1)
        summary = AthleteDataAgent.get_athlete_summary(athlete_id)
        if not summary:
            return {"error": f"Athlete '{athlete_id}' not found in database."}

        # Step 2 – Analyse performance (Agent 2)
        analysis = PerformanceAnalysisAgent.analyse(athlete_id, user_query)

        # Step 3 – Generate training recommendations (Agent 3)
        recommendations = TrainingRecommendationAgent.recommend(athlete_id)

        # Step 4 – Coaching insights (Agent 4)
        coaching = CoachingInsightsAgent.insights(athlete_id)

        return {
            "athlete":         summary["athlete"],
            "summary":         summary,
            "analysis":        analysis,
            "recommendations": recommendations,
            "coaching":        coaching,
            "demo_mode":       DEMO_MODE,
        }

# ---------------------------------------------------------------------------
# Groq API helper
# ---------------------------------------------------------------------------

def _call_groq(prompt: str, fallback_fn=None, fallback_arg=None) -> str:
    """
    Send prompt to Groq API; return response text.
    If the call fails for any reason, invoke fallback_fn(fallback_arg)
    so the app keeps working even with a bad/missing key.
    """
    # Guard: never call Groq with an empty key (causes 'Bearer ' header crash)
    if not GROQ_API_KEY or not GROQ_API_KEY.startswith("gsk_"):
        if fallback_fn and fallback_arg is not None:
            return fallback_fn(fallback_arg) + "\n\n_[Demo Mode – add a valid GROQ_API_KEY to .env for AI responses]_"
        return "⚠️ No valid GROQ_API_KEY found. Add your key to the .env file."

    try:
        # Import here so the app still starts without groq installed
        from groq import Groq  # type: ignore
        client = Groq(api_key=GROQ_API_KEY)
        resp   = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=900,   # stay within free-tier OTPM limits
            temperature=0.7,
        )
        # Some models (reasoning models) return content='' and put
        # output in the 'reasoning' field instead — handle both.
        msg     = resp.choices[0].message
        content = (msg.content or "").strip()
        if not content:
            content = (getattr(msg, "reasoning", None) or "").strip()
        return content
    except ImportError:
        if fallback_fn and fallback_arg is not None:
            return fallback_fn(fallback_arg) + "\n\n_[Groq SDK not installed – run `pip install groq>=1.7.0`]_"
        return (
            "⚠️ Groq SDK not installed. Run `pip install groq>=1.7.0` and "
            "set your GROQ_API_KEY in the .env file to enable AI responses."
        )
    except Exception as e:
        # API key invalid, network error, quota exceeded, etc.
        # Fall back to rule-based demo response so the UI always shows something useful
        print(f"[WARN] Groq API error: {e}")
        if fallback_fn and fallback_arg is not None:
            return fallback_fn(fallback_arg) + f"\n\n_[AI unavailable: {str(e)[:80]}]_"
        return f"⚠️ Groq API error: {str(e)}"

# ---------------------------------------------------------------------------
# Demo-mode fallback responses (rule-based, data-driven)
# ---------------------------------------------------------------------------

def _build_context_string(summary: dict) -> str:
    """Convert summary dict to a compact text block for LLM context."""
    ath = summary.get("athlete", {})
    return (
        f"Athlete: {ath.get('name')} | Sport: {ath.get('sport')} | "
        f"Position: {ath.get('position')} | Age: {ath.get('age')}\n"
        f"Sessions: {summary.get('total_sessions')} | "
        f"Distance: {summary.get('total_distance_km')} km | "
        f"Calories: {summary.get('total_calories')} kcal\n"
        f"Avg HR: {summary.get('avg_heart_rate')} bpm | "
        f"Max HR: {summary.get('max_heart_rate')} bpm | "
        f"VO₂ Max: {summary.get('avg_vo2_max')}\n"
        f"Strength: {summary.get('avg_strength')}/100 | "
        f"Agility: {summary.get('avg_agility')}/100 | "
        f"Endurance: {summary.get('avg_endurance')}/100 | "
        f"Recovery: {summary.get('avg_recovery')}/100\n"
        f"Injuries: {summary.get('total_injuries')} | "
        f"Match Rating: {summary.get('avg_match_rating')} | "
        f"Goals: {summary.get('total_goals')} | "
        f"Assists: {summary.get('total_assists')}\n"
        f"Session breakdown: {summary.get('session_types')}"
    )


def _demo_performance_analysis(summary: dict) -> str:
    """Rule-based performance analysis used when no API key is set."""
    ath     = summary["athlete"]
    metrics = {
        "Strength":  summary["avg_strength"],
        "Agility":   summary["avg_agility"],
        "Endurance": summary["avg_endurance"],
        "Recovery":  summary["avg_recovery"],
        "VO₂ Max":   summary["avg_vo2_max"],
    }
    sorted_m  = sorted(metrics.items(), key=lambda x: x[1], reverse=True)
    strengths = sorted_m[:2]
    weakness  = sorted_m[-2:]
    recovery_flag = " ⚠️ Recovery scores are below 75 – monitor fatigue carefully." \
                    if summary["avg_recovery"] < 75 else ""
    injury_flag   = " ⚠️ Injury incidents recorded – review load management." \
                    if summary["total_injuries"] > 0 else ""

    return (
        f"**Performance Analysis – {ath['name']} ({ath['sport']})**\n\n"
        f"📊 **Trends:** Based on {summary['total_sessions']} sessions "
        f"covering {summary['total_distance_km']} km total. "
        f"Average match rating: {summary['avg_match_rating']}.\n\n"
        f"✅ **Top Strengths:**\n"
        + "".join(f"  • {k}: {v}/100\n" for k, v in strengths) +
        f"\n⚠️ **Areas for Improvement:**\n"
        + "".join(f"  • {k}: {v}/100\n" for k, v in weakness) +
        f"\n🔍 **Risk Flags:**{recovery_flag}{injury_flag if injury_flag else ' None detected.'}\n\n"
        f"💡 **Recommendation:** Prioritise targeted sessions for "
        f"{weakness[0][0].lower()} ({weakness[0][1]}/100) with 2 dedicated "
        f"sessions per week while maintaining {strengths[0][0].lower()} work.\n\n"
        f"_[Demo Mode – add your GROQ_API_KEY for full AI analysis]_"
    )


def _demo_training_plan(summary: dict) -> str:
    """Rule-based 7-day training plan used in demo mode."""
    ath   = summary["athlete"]
    sport = ath["sport"]
    metrics = {
        "strength":  summary["avg_strength"],
        "agility":   summary["avg_agility"],
        "endurance": summary["avg_endurance"],
        "recovery":  summary["avg_recovery"],
    }
    weakest = min(metrics, key=metrics.get)
    focus   = {
        "strength":  "resistance training (compound lifts)",
        "agility":   "cone drills, lateral plyometrics",
        "endurance": "zone-2 cardio, tempo runs",
        "recovery":  "yoga, sleep protocol, HRV monitoring",
    }[weakest]

    return (
        f"**7-Day Training Plan – {ath['name']} ({sport})**\n\n"
        f"🎯 **Focus:** {weakest.title()} development ({metrics[weakest]}/100)\n\n"
        f"**Day 1 (Mon) – {weakest.title()} Focus** | 60–70 min | Moderate\n"
        f"  • Warm-up: 10 min dynamic stretching\n"
        f"  • Main: {focus} – 4 sets × 8–12 reps / 20 min\n"
        f"  • Cool-down: 10 min foam rolling\n\n"
        f"**Day 2 (Tue) – Sport-Specific Drills** | 75 min | High\n"
        f"  • {sport}-specific technical drills\n"
        f"  • Agility ladder + reaction time work\n"
        f"  • Sprint intervals: 6 × 30 m\n\n"
        f"**Day 3 (Wed) – Active Recovery** | 40 min | Low\n"
        f"  • Easy steady-state cardio (HR < 130 bpm)\n"
        f"  • Mobility and flexibility work\n\n"
        f"**Day 4 (Thu) – Endurance Base** | 70 min | Moderate\n"
        f"  • Zone-2 aerobic session\n"
        f"  • Breathing and pacing techniques\n\n"
        f"**Day 5 (Fri) – Power & Explosiveness** | 55 min | High\n"
        f"  • Plyometric circuit: box jumps, bounding\n"
        f"  • Core stability: 3 × 10 each movement\n\n"
        f"**Day 6 (Sat) – Rest / Light Stretch** | 20–30 min | Very Low\n"
        f"  • Passive rest or gentle yoga\n"
        f"  • Hydration and nutrition focus\n\n"
        f"**Day 7 (Sun) – Match / Practice / Time Trial** | Full session\n"
        f"  • Apply trained skills in competitive or practice setting\n\n"
        f"🥗 **Nutrition & Recovery Tips:**\n"
        f"  1. Consume 1.6–2.0 g protein/kg body weight daily\n"
        f"  2. Target 7–9 hours of sleep per night\n"
        f"  3. Hydrate with 500 mL water 2 hrs before every session\n\n"
        f"_[Demo Mode – add your GROQ_API_KEY for a fully personalised AI plan]_"
    )


def _demo_coaching_insights(summary: dict) -> str:
    """Rule-based coaching report used in demo mode."""
    ath   = summary["athlete"]
    sport = ath["sport"]
    stars = "⭐" * round(summary["avg_match_rating"] / 2) if summary["avg_match_rating"] else "N/A"

    return (
        f"**Coaching Report – {ath['name']} ({sport}, {ath['position']})**\n\n"
        f"**1. Executive Summary**\n"
        f"{ath['name']} has completed {summary['total_sessions']} training sessions "
        f"with an average match rating of {summary['avg_match_rating']} {stars}. "
        f"The athlete demonstrates a solid base with targeted development areas identified.\n\n"
        f"**2. Performance Highlights**\n"
        f"  • Total distance covered: {summary['total_distance_km']} km\n"
        f"  • Peak heart rate recorded: {summary['max_heart_rate']} bpm\n"
        f"  • Match contributions: {summary['total_goals']} goals, {summary['total_assists']} assists\n\n"
        f"**3. Areas of Concern**\n"
        f"  • Recovery score avg: {summary['avg_recovery']}/100 "
        f"{'(needs attention)' if summary['avg_recovery'] < 78 else '(satisfactory)'}\n"
        f"  • Injury incidents: {summary['total_injuries']}\n\n"
        f"**4. Tactical Observations**\n"
        f"  • Endurance: {summary['avg_endurance']}/100 | "
        f"Agility: {summary['avg_agility']}/100 | "
        f"Strength: {summary['avg_strength']}/100\n\n"
        f"**5. Short-term Goals (4 weeks)**\n"
        f"  • Raise lowest metric by 5 points through targeted sessions\n"
        f"  • Maintain match rating above {summary['avg_match_rating']}\n\n"
        f"**6. Long-term Development (3–6 months)**\n"
        f"  • Build VO₂ Max from {summary['avg_vo2_max']} toward sport-elite benchmarks\n"
        f"  • Establish structured periodisation blocks\n\n"
        f"**7. Coach Recommendation**\n"
        f"  Implement weekly HRV monitoring to better manage training load and "
        f"prevent overtraining given current recovery patterns.\n\n"
        f"_[Demo Mode – add your GROQ_API_KEY for full AI coaching report]_"
    )

# ---------------------------------------------------------------------------
# Flask Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Main dashboard page."""
    athletes = AthleteDataAgent.get_all_athletes()
    return render_main_page(athletes)


@app.route("/api/athletes")
def api_athletes():
    """Return all athletes as JSON."""
    return jsonify(AthleteDataAgent.get_all_athletes())


@app.route("/api/athlete/<athlete_id>/summary")
def api_summary(athlete_id):
    """Return full summary for one athlete."""
    data = AthleteDataAgent.get_athlete_summary(athlete_id)
    if not data:
        return jsonify({"error": "Athlete not found"}), 404
    return jsonify(data)


@app.route("/api/athlete/<athlete_id>/charts")
def api_charts(athlete_id):
    """Return Chart.js-ready time-series data."""
    data = AthleteDataAgent.get_chart_data(athlete_id)
    if not data:
        return jsonify({"error": "No chart data found"}), 404
    return jsonify(data)


@app.route("/api/comparison")
def api_comparison():
    """Return radar chart comparison data for all athletes."""
    return jsonify(AthleteDataAgent.get_comparison_data())


@app.route("/api/workflow", methods=["POST"])
def api_workflow():
    """
    Run the full 4-agent workflow.
    POST body: { "athlete_id": "A001", "query": "optional question" }
    """
    body       = request.get_json(silent=True) or {}
    athlete_id = body.get("athlete_id", "").strip()
    query      = body.get("query", "").strip()

    if not athlete_id:
        return jsonify({"error": "athlete_id is required"}), 400

    result = AthleteIQWorkflow.run(athlete_id, query)
    return jsonify(result)


@app.route("/api/agent/analysis", methods=["POST"])
def api_analysis():
    """Agent 2 – Performance analysis endpoint."""
    body       = request.get_json(silent=True) or {}
    athlete_id = body.get("athlete_id", "").strip()
    question   = body.get("question", "").strip()

    if not athlete_id:
        return jsonify({"error": "athlete_id is required"}), 400

    result = PerformanceAnalysisAgent.analyse(athlete_id, question)
    return jsonify({"result": result, "demo_mode": DEMO_MODE})


@app.route("/api/agent/training", methods=["POST"])
def api_training():
    """Agent 3 – Training recommendation endpoint."""
    body       = request.get_json(silent=True) or {}
    athlete_id = body.get("athlete_id", "").strip()
    focus      = body.get("focus", "").strip()

    if not athlete_id:
        return jsonify({"error": "athlete_id is required"}), 400

    result = TrainingRecommendationAgent.recommend(athlete_id, focus)
    return jsonify({"result": result, "demo_mode": DEMO_MODE})


@app.route("/api/agent/coaching", methods=["POST"])
def api_coaching():
    """Agent 4 – Coaching insights endpoint."""
    body       = request.get_json(silent=True) or {}
    athlete_id = body.get("athlete_id", "").strip()
    notes      = body.get("notes", "").strip()

    if not athlete_id:
        return jsonify({"error": "athlete_id is required"}), 400

    result = CoachingInsightsAgent.insights(athlete_id, notes)
    return jsonify({"result": result, "demo_mode": DEMO_MODE})


@app.route("/api/report/<athlete_id>")
def api_report(athlete_id):
    """Download a plain-text performance report for an athlete."""
    report = CoachingInsightsAgent.generate_report(athlete_id)
    if not report or report == "No data found.":
        return jsonify({"error": "Athlete not found"}), 404

    buf = io.BytesIO(report.encode("utf-8"))
    buf.seek(0)
    return send_file(
        buf,
        mimetype="text/plain",
        as_attachment=True,
        download_name=f"athleteiq_report_{athlete_id}.txt",
    )


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    Accept a CSV file upload, validate it, and insert records into the DB.
    Expected columns (at minimum): athlete_id, name, sport, session_date, session_type
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    f = request.files["file"]
    if not f.filename or not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "Please upload a .csv file"}), 400

    try:
        stream = io.StringIO(f.stream.read().decode("utf-8"))
        reader = csv.DictReader(stream)
        required = {"athlete_id", "name", "sport", "session_date", "session_type"}
        headers  = {h.strip().lower() for h in (reader.fieldnames or [])}
        missing  = required - headers
        if missing:
            return jsonify({
                "error": f"CSV is missing required columns: {missing}"
            }), 400

        rows_inserted = 0
        db = get_db()
        for row in reader:
            row = {k.strip().lower(): v.strip() for k, v in row.items()}
            # Upsert athlete
            db.execute("""
                INSERT OR IGNORE INTO athletes
                    (athlete_id, name, sport, position, age, weight_kg, height_cm)
                VALUES (?,?,?,?,?,?,?)
            """, (
                row.get("athlete_id", ""),
                row.get("name", "Unknown"),
                row.get("sport", ""),
                row.get("position", ""),
                int(row.get("age", 0) or 0),
                float(row.get("weight_kg", 0) or 0),
                float(row.get("height_cm", 0) or 0),
            ))
            # Insert session
            db.execute("""
                INSERT INTO sessions
                    (athlete_id, session_date, session_type, duration_min,
                     distance_km, avg_heart_rate, max_heart_rate, calories_burned,
                     sprint_speed_kmh, vo2_max, strength_score, agility_score,
                     endurance_score, recovery_score, match_goals, match_assists,
                     match_rating, injuries, notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                row.get("athlete_id", ""),
                row.get("session_date", ""),
                row.get("session_type", ""),
                float(row.get("duration_min", 0) or 0),
                float(row.get("distance_km", 0) or 0),
                float(row.get("avg_heart_rate", 0) or 0),
                float(row.get("max_heart_rate", 0) or 0),
                float(row.get("calories_burned", 0) or 0),
                float(row.get("sprint_speed_kmh", 0) or 0),
                float(row.get("vo2_max", 0) or 0),
                float(row.get("strength_score", 0) or 0),
                float(row.get("agility_score", 0) or 0),
                float(row.get("endurance_score", 0) or 0),
                float(row.get("recovery_score", 0) or 0),
                int(row.get("match_goals", 0) or 0),
                int(row.get("match_assists", 0) or 0),
                float(row.get("match_rating", 0) or 0),
                int(row.get("injuries", 0) or 0),
                row.get("notes", ""),
            ))
            rows_inserted += 1
        db.commit()
        return jsonify({"message": f"Successfully imported {rows_inserted} sessions."})
    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route("/api/add_athlete", methods=["POST"])
def api_add_athlete():
    """
    Add a new athlete + optional first session via a simple JSON form.
    All fields except athlete_id, name, sport are optional.
    """
    body = request.get_json(silent=True) or {}

    # ── Validate required fields ──────────────────────────────────────
    athlete_id = body.get("athlete_id", "").strip().upper()
    name       = body.get("name", "").strip()
    sport      = body.get("sport", "").strip()
    if not athlete_id or not name or not sport:
        return jsonify({"error": "athlete_id, name and sport are required."}), 400

    db = get_db()

    # ── Check duplicate ───────────────────────────────────────────────
    existing = db.execute(
        "SELECT athlete_id FROM athletes WHERE athlete_id = ?", (athlete_id,)
    ).fetchone()
    if existing:
        return jsonify({"error": f"Athlete ID '{athlete_id}' already exists."}), 409

    # ── Insert athlete ────────────────────────────────────────────────
    db.execute("""
        INSERT INTO athletes (athlete_id, name, sport, position, age, weight_kg, height_cm)
        VALUES (?,?,?,?,?,?,?)
    """, (
        athlete_id,
        name,
        sport,
        body.get("position", "").strip(),
        int(body.get("age", 0) or 0),
        float(body.get("weight_kg", 0) or 0),
        float(body.get("height_cm", 0) or 0),
    ))

    # ── Insert first session if session_date provided ─────────────────
    session_saved = False
    if body.get("session_date") and body.get("session_type"):
        db.execute("""
            INSERT INTO sessions
                (athlete_id, session_date, session_type, duration_min,
                 distance_km, avg_heart_rate, max_heart_rate, calories_burned,
                 sprint_speed_kmh, vo2_max, strength_score, agility_score,
                 endurance_score, recovery_score, match_goals, match_assists,
                 match_rating, injuries, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            athlete_id,
            body.get("session_date", ""),
            body.get("session_type", "Training"),
            float(body.get("duration_min", 0) or 0),
            float(body.get("distance_km", 0) or 0),
            float(body.get("avg_heart_rate", 0) or 0),
            float(body.get("max_heart_rate", 0) or 0),
            float(body.get("calories_burned", 0) or 0),
            float(body.get("sprint_speed_kmh", 0) or 0),
            float(body.get("vo2_max", 0) or 0),
            float(body.get("strength_score", 0) or 0),
            float(body.get("agility_score", 0) or 0),
            float(body.get("endurance_score", 0) or 0),
            float(body.get("recovery_score", 0) or 0),
            int(body.get("match_goals", 0) or 0),
            int(body.get("match_assists", 0) or 0),
            float(body.get("match_rating", 0) or 0),
            int(body.get("injuries", 0) or 0),
            body.get("notes", ""),
        ))
        session_saved = True

    db.commit()
    return jsonify({
        "message": f"Athlete '{name}' added successfully!"
                   + (" Session also saved." if session_saved else ""),
        "athlete_id": athlete_id,
    })


@app.route("/api/add_session", methods=["POST"])
def api_add_session():
    """Add a new training session for an existing athlete."""
    body       = request.get_json(silent=True) or {}
    athlete_id = body.get("athlete_id", "").strip()
    session_date = body.get("session_date", "").strip()
    session_type = body.get("session_type", "").strip()

    if not athlete_id or not session_date or not session_type:
        return jsonify({"error": "athlete_id, session_date and session_type are required."}), 400

    db = get_db()
    exists = db.execute(
        "SELECT athlete_id FROM athletes WHERE athlete_id = ?", (athlete_id,)
    ).fetchone()
    if not exists:
        return jsonify({"error": f"Athlete '{athlete_id}' not found."}), 404

    db.execute("""
        INSERT INTO sessions
            (athlete_id, session_date, session_type, duration_min,
             distance_km, avg_heart_rate, max_heart_rate, calories_burned,
             sprint_speed_kmh, vo2_max, strength_score, agility_score,
             endurance_score, recovery_score, match_goals, match_assists,
             match_rating, injuries, notes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        athlete_id, session_date, session_type,
        float(body.get("duration_min", 0) or 0),
        float(body.get("distance_km", 0) or 0),
        float(body.get("avg_heart_rate", 0) or 0),
        float(body.get("max_heart_rate", 0) or 0),
        float(body.get("calories_burned", 0) or 0),
        float(body.get("sprint_speed_kmh", 0) or 0),
        float(body.get("vo2_max", 0) or 0),
        float(body.get("strength_score", 0) or 0),
        float(body.get("agility_score", 0) or 0),
        float(body.get("endurance_score", 0) or 0),
        float(body.get("recovery_score", 0) or 0),
        int(body.get("match_goals", 0) or 0),
        int(body.get("match_assists", 0) or 0),
        float(body.get("match_rating", 0) or 0),
        int(body.get("injuries", 0) or 0),
        body.get("notes", ""),
    ))
    db.commit()
    return jsonify({"message": f"Session saved for {athlete_id} on {session_date}."})


# ---------------------------------------------------------------------------
# HTML Template (render_template_string)
# ---------------------------------------------------------------------------

def render_main_page(athletes):
    """Render the full single-page dashboard using render_template_string."""
    from flask import render_template_string
    return render_template_string(HTML_TEMPLATE, athletes=athletes, demo_mode=DEMO_MODE)


# The full HTML template is defined as a Python string below.
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AthleteIQ AI – Performance Analysis</title>

  <!-- Bootstrap 5 -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"/>
  <!-- Bootstrap Icons -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet"/>
  <!-- Chart.js -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
  <!-- Marked.js (markdown → HTML) -->
  <script src="https://cdn.jsdelivr.net/npm/marked@11.1.1/marked.min.js"></script>

  <style>
    /* ── Design tokens ─────────────────────────────────────────────── */
    :root {
      --primary:   #0d6efd;
      --accent:    #00c6ff;
      --dark-bg:   #0a0e1a;
      --card-bg:   #111827;
      --card-border: #1e2d40;
      --text:      #e2e8f0;
      --muted:     #94a3b8;
      --success:   #22c55e;
      --warning:   #f59e0b;
      --danger:    #ef4444;
      --sidebar-w: 260px;
    }

    /* ── Base ───────────────────────────────────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; }
    html, body { height: 100%; margin: 0; font-family: 'Segoe UI', system-ui, sans-serif; }
    body {
      background: var(--dark-bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      min-height: 100vh;
    }

    /* ── Top navigation ─────────────────────────────────────────────── */
    .topnav {
      background: linear-gradient(90deg, #0a0e1a 0%, #0d2040 50%, #0a0e1a 100%);
      border-bottom: 1px solid var(--card-border);
      padding: .65rem 1.5rem;
      display: flex;
      align-items: center;
      gap: 1rem;
      position: sticky;
      top: 0;
      z-index: 100;
      backdrop-filter: blur(8px);
    }
    .topnav .brand {
      font-size: 1.3rem;
      font-weight: 700;
      color: var(--accent);
      letter-spacing: .5px;
      text-decoration: none;
      white-space: nowrap;
    }
    .topnav .brand span { color: #fff; }
    .demo-badge {
      background: var(--warning);
      color: #000;
      font-size: .72rem;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 4px;
      letter-spacing: .5px;
    }
    .nav-tabs-custom {
      display: flex;
      gap: .25rem;
      margin-left: auto;
      flex-wrap: wrap;
    }
    .nav-tabs-custom .nav-btn {
      background: transparent;
      border: 1px solid transparent;
      color: var(--muted);
      padding: .35rem .85rem;
      border-radius: 6px;
      font-size: .85rem;
      cursor: pointer;
      transition: all .2s;
    }
    .nav-tabs-custom .nav-btn:hover,
    .nav-tabs-custom .nav-btn.active {
      background: #1e2d40;
      border-color: var(--primary);
      color: #fff;
    }

    /* ── Layout ─────────────────────────────────────────────────────── */
    .app-body {
      display: flex;
      flex: 1;
      overflow: hidden;
    }
    .sidebar {
      width: var(--sidebar-w);
      min-width: var(--sidebar-w);
      background: #0c1320;
      border-right: 1px solid var(--card-border);
      overflow-y: auto;
      padding: 1rem .75rem;
      display: flex;
      flex-direction: column;
      gap: .35rem;
    }
    .sidebar-label {
      font-size: .7rem;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .8px;
      padding: .5rem .5rem .2rem;
    }
    .athlete-btn {
      display: flex;
      align-items: center;
      gap: .65rem;
      background: transparent;
      border: 1px solid transparent;
      border-radius: 8px;
      color: var(--text);
      padding: .55rem .75rem;
      cursor: pointer;
      width: 100%;
      text-align: left;
      transition: all .2s;
      font-size: .88rem;
    }
    .athlete-btn:hover { background: #1a2235; border-color: var(--card-border); }
    .athlete-btn.active {
      background: linear-gradient(90deg, #0d2040, #0a1830);
      border-color: var(--primary);
      color: #fff;
    }
    .athlete-avatar {
      width: 34px; height: 34px;
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-weight: 700; font-size: .85rem;
      flex-shrink: 0;
    }
    .sport-icon { font-size: 1rem; }

    .main-content {
      flex: 1;
      overflow-y: auto;
      padding: 1.25rem 1.5rem 2rem;
    }

    /* ── Cards ─────────────────────────────────────────────────────── */
    .card-dark {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 1.25rem;
    }
    .card-dark .card-title {
      font-size: .85rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .6px;
      margin-bottom: .3rem;
    }
    .card-dark .card-value {
      font-size: 1.7rem;
      font-weight: 700;
      color: #fff;
    }
    .card-dark .card-sub {
      font-size: .78rem;
      color: var(--muted);
    }

    /* ── Charts ─────────────────────────────────────────────────────── */
    .chart-container {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 1.25rem;
    }
    .chart-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;
    }
    .chart-title { font-weight: 600; font-size: 1rem; }

    /* ── AI panels ──────────────────────────────────────────────────── */
    .ai-panel {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 1.25rem;
    }
    .ai-panel-header {
      display: flex;
      align-items: center;
      gap: .6rem;
      margin-bottom: 1rem;
      padding-bottom: .75rem;
      border-bottom: 1px solid var(--card-border);
    }
    .agent-badge {
      font-size: .7rem;
      padding: 2px 8px;
      border-radius: 20px;
      font-weight: 600;
    }
    .agent-badge.analysis  { background: #1e3a5f; color: #60a5fa; }
    .agent-badge.training  { background: #14532d; color: #4ade80; }
    .agent-badge.coaching  { background: #3b1f6e; color: #c084fc; }
    .agent-badge.workflow  { background: #431407; color: #fb923c; }

    .ai-output {
      font-size: .9rem;
      line-height: 1.7;
      color: var(--text);
      min-height: 80px;
    }
    .ai-output h2 { font-size: 1.05rem; color: var(--accent); margin-top: 1rem; }
    .ai-output h3 { font-size: .95rem; color: #93c5fd; margin-top: .8rem; }
    .ai-output strong { color: #fff; }
    .ai-output ul { padding-left: 1.2rem; }
    .ai-output li { margin-bottom: .2rem; }
    .ai-output code {
      background: #1e2d40;
      padding: 1px 5px;
      border-radius: 4px;
      font-size: .85em;
    }

    /* ── Buttons ─────────────────────────────────────────────────────── */
    .btn-ai {
      background: linear-gradient(135deg, var(--primary), #0099cc);
      border: none;
      color: #fff;
      padding: .45rem 1.1rem;
      border-radius: 8px;
      font-size: .85rem;
      font-weight: 500;
      cursor: pointer;
      transition: opacity .2s;
      display: inline-flex;
      align-items: center;
      gap: .4rem;
    }
    .btn-ai:hover  { opacity: .85; }
    .btn-ai:active { opacity: .7;  }

    /* ── Spinner ─────────────────────────────────────────────────────── */
    .spinner-sm {
      width: 1rem; height: 1rem;
      border: 2px solid #ffffff40;
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin .7s linear infinite;
      display: inline-block;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Skill meters ────────────────────────────────────────────────── */
    .skill-bar-wrap { margin-bottom: .75rem; }
    .skill-label {
      display: flex;
      justify-content: space-between;
      font-size: .8rem;
      margin-bottom: .25rem;
      color: var(--muted);
    }
    .skill-bar-bg {
      background: #1e2d40;
      border-radius: 6px;
      height: 8px;
      overflow: hidden;
    }
    .skill-bar-fill {
      height: 100%;
      border-radius: 6px;
      transition: width .6s ease;
    }

    /* ── Section tabs ────────────────────────────────────────────────── */
    .section-tabs { display: flex; gap: .4rem; margin-bottom: 1.25rem; flex-wrap: wrap; }
    .section-tab {
      background: #111827;
      border: 1px solid var(--card-border);
      color: var(--muted);
      padding: .4rem 1rem;
      border-radius: 8px;
      cursor: pointer;
      font-size: .85rem;
      transition: all .2s;
    }
    .section-tab.active { background: var(--primary); border-color: var(--primary); color: #fff; }

    /* ── Upload zone ─────────────────────────────────────────────────── */
    .upload-zone {
      border: 2px dashed var(--card-border);
      border-radius: 10px;
      padding: 2rem;
      text-align: center;
      cursor: pointer;
      transition: border-color .2s;
    }
    .upload-zone:hover { border-color: var(--primary); }

    /* ── Alert inline ────────────────────────────────────────────────── */
    .alert-inline {
      padding: .5rem .9rem;
      border-radius: 8px;
      font-size: .85rem;
      margin-top: .5rem;
    }
    .alert-inline.success { background: #14532d40; border: 1px solid #22c55e40; color: var(--success); }
    .alert-inline.danger  { background: #7f1d1d40; border: 1px solid #ef444440; color: var(--danger);  }

    /* ── Responsive ──────────────────────────────────────────────────── */
    @media (max-width: 768px) {
      .app-body    { flex-direction: column; }
      .sidebar     { width: 100%; min-width: unset; flex-direction: row; flex-wrap: wrap; padding: .5rem; }
      .athlete-btn { width: auto; flex: 1 1 120px; }
      .main-content { padding: .75rem; }
      .nav-tabs-custom { display: none; }
    }
  </style>
</head>

<body>
  <!-- ── Top Navigation ────────────────────────────────────────────── -->
  <nav class="topnav">
    <a class="brand" href="#"><i class="bi bi-lightning-charge-fill me-1" style="color:var(--accent)"></i>AthleteIQ <span>AI</span></a>
    {% if demo_mode %}
    <span title="No GROQ_API_KEY set – using rule-based responses" style="display:flex;align-items:center;gap:.35rem;font-size:.78rem;color:var(--warning);cursor:default">
      <span style="width:8px;height:8px;border-radius:50%;background:var(--warning);display:inline-block"></span>Demo Mode
    </span>
    {% else %}
    <span title="Groq AI connected" style="display:flex;align-items:center;gap:.35rem;font-size:.78rem;color:var(--success);cursor:default">
      <span style="width:8px;height:8px;border-radius:50%;background:var(--success);display:inline-block"></span>AI Connected
    </span>
    {% endif %}
    <div class="nav-tabs-custom">
      <button class="nav-btn active" onclick="showSection('dashboard',this)"><i class="bi bi-grid-1x2 me-1"></i>Dashboard</button>
      <button class="nav-btn" onclick="showSection('agents',this)"><i class="bi bi-cpu me-1"></i>AI Agents</button>
      <button class="nav-btn" onclick="showSection('comparison',this)"><i class="bi bi-bar-chart-line me-1"></i>Comparison</button>
      <button class="nav-btn" onclick="showSection('upload',this)"><i class="bi bi-person-plus me-1"></i>Add Athlete</button>
    </div>
  </nav>

  <!-- ── App body ───────────────────────────────────────────────────── -->
  <div class="app-body">

    <!-- ── Sidebar ──────────────────────────────────────────────────── -->
    <aside class="sidebar" id="sidebar">
      <div class="sidebar-label"><i class="bi bi-people me-1"></i>Athletes</div>
      {% for ath in athletes %}
      <button class="athlete-btn {% if loop.first %}active{% endif %}"
              onclick="selectAthlete('{{ ath.athlete_id }}', this)"
              data-sport="{{ ath.sport }}">
        <div class="athlete-avatar" style="background:{{ loop_color(loop.index0) }}">
          {{ ath.name[0] }}{{ ath.name.split(' ')[-1][0] if ' ' in ath.name else '' }}
        </div>
        <div>
          <div style="font-weight:600;font-size:.85rem;line-height:1.2">{{ ath.name }}</div>
          <div style="font-size:.72rem;color:var(--muted)">{{ ath.sport }} · {{ ath.position }}</div>
        </div>
      </button>
      {% else %}
      <div style="padding:.75rem;color:var(--muted);font-size:.83rem;text-align:center">
        <i class="bi bi-person-x" style="font-size:1.5rem;display:block;margin-bottom:.5rem"></i>
        No athletes yet.<br>
        <button onclick="showSection('upload', document.querySelector('.nav-btn:last-child'))"
                style="background:var(--primary);border:none;color:#fff;padding:.3rem .8rem;border-radius:6px;margin-top:.5rem;cursor:pointer;font-size:.8rem">
          Add Athlete
        </button>
      </div>
      {% endfor %}
    </aside>

    <!-- ── Main content ─────────────────────────────────────────────── -->
    <main class="main-content" id="mainContent">

      <!-- ══ Section: Dashboard ═════════════════════════════════════ -->
      <div id="sec-dashboard">

        <!-- Athlete header -->
        <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem;flex-wrap:wrap">
          <div id="ath-avatar" class="athlete-avatar" style="width:56px;height:56px;font-size:1.2rem;background:#1e3a5f"></div>
          <div>
            <h4 id="ath-name" style="margin:0;font-weight:700"></h4>
            <div id="ath-meta" style="color:var(--muted);font-size:.85rem"></div>
          </div>
          <div style="margin-left:auto;display:flex;gap:.5rem;flex-wrap:wrap">
            <button class="btn-ai" id="workflowBtn" onclick="runWorkflow(this)">
              <i class="bi bi-diagram-3"></i>Run Full Workflow
            </button>
            <button class="btn-ai" style="background:linear-gradient(135deg,#6d28d9,#4f46e5)"
                    onclick="downloadReport()">
              <i class="bi bi-file-earmark-arrow-down"></i>Download Report
            </button>
          </div>
        </div>

        <!-- KPI cards row -->
        <div class="row g-3 mb-3" id="kpiRow">
          <div class="col-6 col-md-3">
            <div class="card-dark">
              <div class="card-title"><i class="bi bi-activity me-1"></i>Sessions</div>
              <div class="card-value" id="kpi-sessions">–</div>
              <div class="card-sub">total logged</div>
            </div>
          </div>
          <div class="col-6 col-md-3">
            <div class="card-dark">
              <div class="card-title"><i class="bi bi-geo-alt me-1"></i>Distance</div>
              <div class="card-value" id="kpi-distance">–</div>
              <div class="card-sub">kilometres covered</div>
            </div>
          </div>
          <div class="col-6 col-md-3">
            <div class="card-dark">
              <div class="card-title"><i class="bi bi-heart-pulse me-1"></i>Avg HR</div>
              <div class="card-value" id="kpi-hr">–</div>
              <div class="card-sub">beats per minute</div>
            </div>
          </div>
          <div class="col-6 col-md-3">
            <div class="card-dark">
              <div class="card-title"><i class="bi bi-star me-1"></i>Match Rating</div>
              <div class="card-value" id="kpi-rating">–</div>
              <div class="card-sub">average score</div>
            </div>
          </div>
        </div>

        <!-- Charts row -->
        <div class="row g-3 mb-3">
          <!-- Performance over time -->
          <div class="col-12 col-lg-8">
            <div class="chart-container" style="height:280px">
              <div class="chart-header">
                <span class="chart-title"><i class="bi bi-graph-up me-1" style="color:var(--accent)"></i>Performance Metrics Over Time</span>
                <select id="metricSelect" onchange="updateLineChart()" style="background:#1e2d40;border:1px solid var(--card-border);color:var(--text);padding:3px 8px;border-radius:6px;font-size:.8rem">
                  <option value="endurance">Endurance</option>
                  <option value="strength">Strength</option>
                  <option value="agility">Agility</option>
                  <option value="recovery">Recovery</option>
                  <option value="heart_rate">Heart Rate</option>
                  <option value="calories">Calories</option>
                  <option value="vo2_max">VO₂ Max</option>
                </select>
              </div>
              <canvas id="lineChart" style="max-height:200px"></canvas>
            </div>
          </div>

          <!-- Skill radar -->
          <div class="col-12 col-lg-4">
            <div class="chart-container" style="height:280px">
              <div class="chart-header">
                <span class="chart-title"><i class="bi bi-pentagon me-1" style="color:#c084fc"></i>Skill Profile</span>
              </div>
              <canvas id="radarChart" style="max-height:200px"></canvas>
            </div>
          </div>
        </div>

        <!-- Session donut + workload bar -->
        <div class="row g-3 mb-3">
          <div class="col-12 col-md-5">
            <div class="chart-container">
              <div class="chart-header">
                <span class="chart-title"><i class="bi bi-pie-chart me-1" style="color:var(--warning)"></i>Session Breakdown</span>
              </div>
              <canvas id="donutChart" style="max-height:180px"></canvas>
            </div>
          </div>
          <div class="col-12 col-md-7">
            <div class="chart-container">
              <div class="chart-header">
                <span class="chart-title"><i class="bi bi-bar-chart me-1" style="color:var(--success)"></i>Weekly Workload (Calories)</span>
              </div>
              <canvas id="workloadChart" style="max-height:180px"></canvas>
            </div>
          </div>
        </div>

        <!-- Skill meters + recent sessions -->
        <div class="row g-3">
          <div class="col-12 col-md-5">
            <div class="ai-panel">
              <div class="ai-panel-header">
                <i class="bi bi-speedometer2" style="color:var(--accent)"></i>
                <strong>Skill Metrics</strong>
              </div>
              <div id="skillMeters"></div>
            </div>
          </div>
          <div class="col-12 col-md-7">
            <div class="ai-panel">
              <div class="ai-panel-header">
                <i class="bi bi-calendar3" style="color:var(--warning)"></i>
                <strong>Recent Sessions</strong>
              </div>
              <div id="recentSessions" style="font-size:.85rem"></div>
            </div>
          </div>
        </div>

        <!-- Workflow output -->
        <div class="mt-3 ai-panel" id="workflowPanel" style="display:none">
          <div class="ai-panel-header">
            <i class="bi bi-diagram-3" style="color:#fb923c"></i>
            <strong>Multi-Agent Workflow Output</strong>
            <span class="agent-badge workflow ms-auto">All 4 Agents</span>
          </div>
          <div id="workflowOutput" class="ai-output"></div>
        </div>
      </div><!-- /sec-dashboard -->

      <!-- ══ Section: AI Agents ══════════════════════════════════════ -->
      <div id="sec-agents" style="display:none">
        <h5 style="font-weight:700;margin-bottom:1.25rem">
          <i class="bi bi-cpu me-2" style="color:var(--accent)"></i>AI Agent Console
        </h5>

        <!-- Agent 2 – Performance Analysis -->
        <div class="ai-panel mb-3">
          <div class="ai-panel-header">
            <i class="bi bi-graph-up-arrow" style="color:#60a5fa"></i>
            <strong>Agent 2 – Performance Analysis</strong>
            <span class="agent-badge analysis ms-auto">Analysis</span>
          </div>
          <div class="mb-2">
            <input id="analysisQ" type="text" class="form-control form-control-sm"
                   style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"
                   placeholder="Ask a specific question (optional)…" />
          </div>
          <button class="btn-ai" onclick="runAgent('analysis',this)">
            <i class="bi bi-play-circle"></i>Analyse Performance
          </button>
          <div id="analysisOutput" class="ai-output mt-3"></div>
        </div>

        <!-- Agent 3 – Training Recommendation -->
        <div class="ai-panel mb-3">
          <div class="ai-panel-header">
            <i class="bi bi-calendar-check" style="color:#4ade80"></i>
            <strong>Agent 3 – Training Recommendation</strong>
            <span class="agent-badge training ms-auto">Training</span>
          </div>
          <div class="mb-2">
            <input id="trainingFocus" type="text" class="form-control form-control-sm"
                   style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"
                   placeholder="Specific focus area (e.g. speed, endurance)…" />
          </div>
          <button class="btn-ai" style="background:linear-gradient(135deg,#15803d,#065f46)"
                  onclick="runAgent('training',this)">
            <i class="bi bi-play-circle"></i>Generate Plan
          </button>
          <div id="trainingOutput" class="ai-output mt-3"></div>
        </div>

        <!-- Agent 4 – Coaching Insights -->
        <div class="ai-panel">
          <div class="ai-panel-header">
            <i class="bi bi-clipboard2-pulse" style="color:#c084fc"></i>
            <strong>Agent 4 – Coaching Insights</strong>
            <span class="agent-badge coaching ms-auto">Coaching</span>
          </div>
          <div class="mb-2">
            <textarea id="coachNotes" rows="2" class="form-control form-control-sm"
                      style="background:#1e2d40;border-color:var(--card-border);color:var(--text);resize:none"
                      placeholder="Additional coach notes (optional)…"></textarea>
          </div>
          <button class="btn-ai" style="background:linear-gradient(135deg,#6d28d9,#4f46e5)"
                  onclick="runAgent('coaching',this)">
            <i class="bi bi-play-circle"></i>Get Coaching Insights
          </button>
          <div id="coachingOutput" class="ai-output mt-3"></div>
        </div>
      </div><!-- /sec-agents -->

      <!-- ══ Section: Athlete Comparison ════════════════════════════ -->
      <div id="sec-comparison" style="display:none">
        <h5 style="font-weight:700;margin-bottom:1.25rem">
          <i class="bi bi-bar-chart-line me-2" style="color:var(--accent)"></i>Athlete Comparison
        </h5>
        <div class="row g-3">
          <div class="col-12 col-lg-7">
            <div class="chart-container">
              <div class="chart-header">
                <span class="chart-title">Skill Radar – All Athletes</span>
              </div>
              <canvas id="comparisonRadar" style="max-height:380px"></canvas>
            </div>
          </div>
          <div class="col-12 col-lg-5">
            <div class="chart-container">
              <div class="chart-header">
                <span class="chart-title">VO₂ Max Comparison</span>
              </div>
              <canvas id="vo2Chart" style="max-height:380px"></canvas>
            </div>
          </div>
        </div>
        <div class="mt-3">
          <div class="chart-container">
            <div class="chart-header">
              <span class="chart-title">Endurance vs Strength – All Athletes</span>
            </div>
            <canvas id="scatterChart" style="max-height:250px"></canvas>
          </div>
        </div>
      </div><!-- /sec-comparison -->

      <!-- ══ Section: Add Athlete / Session / CSV ═══════════════════ -->
      <div id="sec-upload" style="display:none">
        <h5 style="font-weight:700;margin-bottom:1.25rem">
          <i class="bi bi-person-plus me-2" style="color:var(--accent)"></i>Add Athlete Data
        </h5>

        <!-- Sub-tabs -->
        <div class="section-tabs mb-3">
          <button class="section-tab active" id="tab-newathlete"  onclick="switchAddTab('newathlete',this)"><i class="bi bi-person-plus me-1"></i>New Athlete</button>
          <button class="section-tab"         id="tab-newsession"  onclick="switchAddTab('newsession',this)"><i class="bi bi-calendar-plus me-1"></i>Add Session</button>
          <button class="section-tab"         id="tab-csvupload"   onclick="switchAddTab('csvupload',this)"><i class="bi bi-file-earmark-spreadsheet me-1"></i>Bulk CSV Upload</button>
        </div>

        <!-- ── Tab 1: New Athlete Form ─────────────────────────────── -->
        <div id="addtab-newathlete" class="ai-panel">
          <div class="ai-panel-header">
            <i class="bi bi-person-badge" style="color:var(--accent)"></i>
            <strong>Add New Athlete</strong>
            <span style="font-size:.78rem;color:var(--muted);margin-left:auto">* required</span>
          </div>

          <div class="row g-3">
            <!-- Athlete ID -->
            <div class="col-6 col-md-3">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">Athlete ID *</label>
              <input id="f-id" type="text" class="form-control form-control-sm" placeholder="e.g. A006"
                     style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
              <div style="font-size:.72rem;color:var(--muted);margin-top:2px">Unique code, no spaces</div>
            </div>
            <!-- Full Name -->
            <div class="col-6 col-md-4">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">Full Name *</label>
              <input id="f-name" type="text" class="form-control form-control-sm" placeholder="e.g. John Smith"
                     style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
            </div>
            <!-- Sport -->
            <div class="col-6 col-md-3">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">Sport *</label>
              <select id="f-sport" class="form-select form-select-sm"
                      style="background:#1e2d40;border-color:var(--card-border);color:var(--text)">
                <option value="">Select sport…</option>
                <option>Soccer</option><option>Basketball</option><option>Swimming</option>
                <option>Tennis</option><option>Cycling</option><option>Athletics</option>
                <option>Rugby</option><option>Cricket</option><option>Other</option>
              </select>
            </div>
            <!-- Position -->
            <div class="col-6 col-md-2">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">Position</label>
              <input id="f-position" type="text" class="form-control form-control-sm" placeholder="e.g. Forward"
                     style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
            </div>
            <!-- Age -->
            <div class="col-4 col-md-2">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">Age</label>
              <input id="f-age" type="number" min="10" max="60" class="form-control form-control-sm" placeholder="25"
                     style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
            </div>
            <!-- Weight -->
            <div class="col-4 col-md-2">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">Weight (kg)</label>
              <input id="f-weight" type="number" min="30" max="200" class="form-control form-control-sm" placeholder="75"
                     style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
            </div>
            <!-- Height -->
            <div class="col-4 col-md-2">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">Height (cm)</label>
              <input id="f-height" type="number" min="100" max="250" class="form-control form-control-sm" placeholder="180"
                     style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
            </div>
          </div>

          <!-- Optional: first session -->
          <div style="margin-top:1.25rem;padding-top:1rem;border-top:1px solid var(--card-border)">
            <div style="font-size:.82rem;color:var(--muted);margin-bottom:.75rem">
              <i class="bi bi-calendar-plus me-1"></i>Optional: Add first session at the same time
            </div>
            <div class="row g-3">
              <div class="col-6 col-md-3">
                <label class="form-label" style="font-size:.8rem;color:var(--muted)">Session Date</label>
                <input id="f-sdate" type="date" class="form-control form-control-sm"
                       style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
              </div>
              <div class="col-6 col-md-3">
                <label class="form-label" style="font-size:.8rem;color:var(--muted)">Session Type</label>
                <select id="f-stype" class="form-select form-select-sm"
                        style="background:#1e2d40;border-color:var(--card-border);color:var(--text)">
                  <option value="">None</option>
                  <option>Training</option><option>Match</option><option>Interval</option>
                  <option>Strength</option><option>Recovery</option><option>Endurance</option>
                  <option>Agility</option><option>Race</option><option>Practice</option>
                </select>
              </div>
              <div class="col-6 col-md-2">
                <label class="form-label" style="font-size:.8rem;color:var(--muted)">Duration (min)</label>
                <input id="f-dur" type="number" min="0" class="form-control form-control-sm" placeholder="60"
                       style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
              </div>
              <div class="col-6 col-md-2">
                <label class="form-label" style="font-size:.8rem;color:var(--muted)">Avg Heart Rate</label>
                <input id="f-hr" type="number" min="0" class="form-control form-control-sm" placeholder="150"
                       style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
              </div>
              <div class="col-6 col-md-2">
                <label class="form-label" style="font-size:.8rem;color:var(--muted)">Calories</label>
                <input id="f-cal" type="number" min="0" class="form-control form-control-sm" placeholder="500"
                       style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
              </div>
              <div class="col-12">
                <label class="form-label" style="font-size:.8rem;color:var(--muted)">Notes</label>
                <input id="f-notes" type="text" class="form-control form-control-sm" placeholder="Session notes…"
                       style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
              </div>
            </div>
          </div>

          <div style="margin-top:1.25rem;display:flex;gap:.75rem;align-items:center;flex-wrap:wrap">
            <button class="btn-ai" onclick="submitNewAthlete(this)">
              <i class="bi bi-person-check"></i>Save Athlete
            </button>
            <button onclick="clearAthleteForm()" style="background:transparent;border:1px solid var(--card-border);color:var(--muted);padding:.4rem 1rem;border-radius:8px;cursor:pointer;font-size:.85rem">
              <i class="bi bi-x-circle me-1"></i>Clear
            </button>
          </div>
          <div id="athleteFormStatus" style="margin-top:.75rem"></div>
        </div>

        <!-- ── Tab 2: Add Session Form ─────────────────────────────── -->
        <div id="addtab-newsession" class="ai-panel" style="display:none">
          <div class="ai-panel-header">
            <i class="bi bi-calendar-plus" style="color:var(--warning)"></i>
            <strong>Add Training Session</strong>
            <span style="font-size:.78rem;color:var(--muted);margin-left:auto">* required</span>
          </div>

          <div class="row g-3">
            <!-- Athlete selector -->
            <div class="col-12 col-md-4">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">Athlete *</label>
              <select id="s-athlete" class="form-select form-select-sm"
                      style="background:#1e2d40;border-color:var(--card-border);color:var(--text)">
                <option value="">Select athlete…</option>
                {% for ath in athletes %}
                <option value="{{ ath.athlete_id }}">{{ ath.name }} ({{ ath.sport }})</option>
                {% endfor %}
              </select>
            </div>
            <div class="col-6 col-md-3">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">Date *</label>
              <input id="s-date" type="date" class="form-control form-control-sm"
                     style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
            </div>
            <div class="col-6 col-md-3">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">Session Type *</label>
              <select id="s-type" class="form-select form-select-sm"
                      style="background:#1e2d40;border-color:var(--card-border);color:var(--text)">
                <option value="">Select type…</option>
                <option>Training</option><option>Match</option><option>Interval</option>
                <option>Strength</option><option>Recovery</option><option>Endurance</option>
                <option>Agility</option><option>Race</option><option>Practice</option>
              </select>
            </div>
            <div class="col-6 col-md-2">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">Duration (min)</label>
              <input id="s-dur" type="number" min="0" class="form-control form-control-sm" placeholder="60"
                     style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
            </div>
            <div class="col-6 col-md-2">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">Distance (km)</label>
              <input id="s-dist" type="number" min="0" step="0.1" class="form-control form-control-sm" placeholder="5.0"
                     style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
            </div>
            <div class="col-6 col-md-2">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">Avg HR (bpm)</label>
              <input id="s-hr" type="number" min="0" class="form-control form-control-sm" placeholder="150"
                     style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
            </div>
            <div class="col-6 col-md-2">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">Max HR (bpm)</label>
              <input id="s-maxhr" type="number" min="0" class="form-control form-control-sm" placeholder="185"
                     style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
            </div>
            <div class="col-6 col-md-2">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">Calories</label>
              <input id="s-cal" type="number" min="0" class="form-control form-control-sm" placeholder="500"
                     style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
            </div>
            <div class="col-6 col-md-2">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">Strength (0-100)</label>
              <input id="s-str" type="number" min="0" max="100" class="form-control form-control-sm" placeholder="70"
                     style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
            </div>
            <div class="col-6 col-md-2">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">Endurance (0-100)</label>
              <input id="s-end" type="number" min="0" max="100" class="form-control form-control-sm" placeholder="70"
                     style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
            </div>
            <div class="col-6 col-md-2">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">Recovery (0-100)</label>
              <input id="s-rec" type="number" min="0" max="100" class="form-control form-control-sm" placeholder="80"
                     style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
            </div>
            <div class="col-6 col-md-2">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">Agility (0-100)</label>
              <input id="s-agi" type="number" min="0" max="100" class="form-control form-control-sm" placeholder="70"
                     style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
            </div>
            <div class="col-6 col-md-2">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">VO₂ Max</label>
              <input id="s-vo2" type="number" min="0" step="0.1" class="form-control form-control-sm" placeholder="55"
                     style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
            </div>
            <div class="col-6 col-md-2">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">Match Rating</label>
              <input id="s-rating" type="number" min="0" max="10" step="0.1" class="form-control form-control-sm" placeholder="0"
                     style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
            </div>
            <div class="col-6 col-md-2">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">Goals</label>
              <input id="s-goals" type="number" min="0" class="form-control form-control-sm" placeholder="0"
                     style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
            </div>
            <div class="col-6 col-md-2">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">Assists</label>
              <input id="s-assists" type="number" min="0" class="form-control form-control-sm" placeholder="0"
                     style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
            </div>
            <div class="col-6 col-md-2">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">Injury?</label>
              <select id="s-injury" class="form-select form-select-sm"
                      style="background:#1e2d40;border-color:var(--card-border);color:var(--text)">
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
            </div>
            <div class="col-12">
              <label class="form-label" style="font-size:.8rem;color:var(--muted)">Notes</label>
              <input id="s-notes" type="text" class="form-control form-control-sm" placeholder="Session notes…"
                     style="background:#1e2d40;border-color:var(--card-border);color:var(--text)"/>
            </div>
          </div>

          <div style="margin-top:1.25rem;display:flex;gap:.75rem;align-items:center;flex-wrap:wrap">
            <button class="btn-ai" style="background:linear-gradient(135deg,#b45309,#92400e)" onclick="submitNewSession(this)">
              <i class="bi bi-calendar-check"></i>Save Session
            </button>
            <button onclick="clearSessionForm()" style="background:transparent;border:1px solid var(--card-border);color:var(--muted);padding:.4rem 1rem;border-radius:8px;cursor:pointer;font-size:.85rem">
              <i class="bi bi-x-circle me-1"></i>Clear
            </button>
          </div>
          <div id="sessionFormStatus" style="margin-top:.75rem"></div>
        </div>

        <!-- ── Tab 3: CSV Upload ───────────────────────────────────── -->
        <div id="addtab-csvupload" class="ai-panel" style="display:none">
          <div class="ai-panel-header">
            <i class="bi bi-file-earmark-spreadsheet" style="color:var(--success)"></i>
            <strong>Bulk CSV Upload</strong>
          </div>
          <p style="color:var(--muted);font-size:.88rem;margin-bottom:1rem">
            Upload a CSV file to add multiple athletes and sessions at once.<br>
            <strong style="color:var(--text)">Required columns:</strong>
            <code>athlete_id</code>, <code>name</code>, <code>sport</code>,
            <code>session_date</code>, <code>session_type</code><br>
            All other columns are optional. See <code>athletes.csv</code> for the full example.
          </p>
          <div class="upload-zone" id="uploadZone" onclick="document.getElementById('csvFile').click()">
            <i class="bi bi-cloud-upload" style="font-size:2.5rem;color:var(--muted)"></i>
            <p class="mt-2 mb-1" style="color:var(--text);font-weight:600">Click to choose a CSV file</p>
            <p style="font-size:.8rem;color:var(--muted);margin:0">Max 5 MB · .csv files only</p>
          </div>
          <input type="file" id="csvFile" accept=".csv" style="display:none" onchange="uploadCSV(this)" />
          <div id="uploadStatus"></div>
        </div>

      </div><!-- /sec-upload -->

    </main><!-- /main-content -->
  </div><!-- /app-body -->

  <!-- Bootstrap JS -->
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>

  <script>
  // ─────────────────────────────────────────────────────────────────────────
  // State
  // ─────────────────────────────────────────────────────────────────────────
  let currentAthleteId  = "";
  let chartData         = {};
  let comparisonData    = {};
  let lineChartInst     = null;
  let radarChartInst    = null;
  let donutChartInst    = null;
  let workloadChartInst = null;
  let compRadarInst     = null;
  let vo2ChartInst      = null;
  let scatterInst       = null;

  const PALETTE = [
    "#3b82f6","#22c55e","#f59e0b","#a855f7","#ef4444",
    "#06b6d4","#f97316","#84cc16","#ec4899","#14b8a6"
  ];

  // ─────────────────────────────────────────────────────────────────────────
  // Navigation
  // ─────────────────────────────────────────────────────────────────────────
  function showSection(name, btn) {
    ["dashboard","agents","comparison","upload"].forEach(s => {
      document.getElementById("sec-" + s).style.display = (s === name) ? "" : "none";
    });
    // Only reset nav-btn active state (not section-tab buttons inside panels)
    document.querySelectorAll(".nav-tabs-custom .nav-btn").forEach(b => b.classList.remove("active"));
    if (btn) btn.classList.add("active");
    if (name === "comparison") loadComparisonCharts();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Select athlete from sidebar
  // ─────────────────────────────────────────────────────────────────────────
  function selectAthlete(id, btn) {
    document.querySelectorAll(".athlete-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentAthleteId = id;
    loadAthleteDashboard(id);
  }

  async function loadAthleteDashboard(id) {
    // Show loading state on KPIs
    ["kpi-sessions","kpi-distance","kpi-hr","kpi-rating"].forEach(k => {
      const el = document.getElementById(k); if (el) el.textContent = "…";
    });
    document.getElementById("ath-name").textContent = "Loading…";
    document.getElementById("ath-meta").textContent = "";

    try {
      const [summary, charts] = await Promise.all([
        fetch("/api/athlete/" + id + "/summary").then(r => r.json()),
        fetch("/api/athlete/" + id + "/charts").then(r => r.json()),
      ]);

      if (summary.error) throw new Error(summary.error);

      chartData = charts;
      renderKPIs(summary);
      renderSkillMeters(summary);
      renderRecentSessions(summary.recent_sessions || []);
      renderLineChart();
      renderRadarChart(summary);
      renderDonutChart(summary.session_types || {});
      renderWorkloadChart();

      // Reset agent outputs when switching athlete so stale results don't show
      ["analysisOutput","trainingOutput","coachingOutput","workflowOutput"].forEach(outId => {
        const el = document.getElementById(outId);
        if (el) el.innerHTML = "";
      });
      document.getElementById("workflowPanel").style.display = "none";

    } catch (e) {
      console.error("Dashboard load error:", e);
      document.getElementById("ath-name").textContent = "Error loading athlete";
      document.getElementById("ath-meta").textContent = e.message;
      ["kpi-sessions","kpi-distance","kpi-hr","kpi-rating"].forEach(k => {
        const el = document.getElementById(k); if (el) el.textContent = "–";
      });
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // KPI cards
  // ─────────────────────────────────────────────────────────────────────────
  function renderKPIs(s) {
    document.getElementById("kpi-sessions").textContent = s.total_sessions ?? "–";
    document.getElementById("kpi-distance").textContent = (s.total_distance_km ?? "–") + " km";
    document.getElementById("kpi-hr").textContent       = (s.avg_heart_rate ?? "–") + " bpm";
    document.getElementById("kpi-rating").textContent   = s.avg_match_rating > 0 ? s.avg_match_rating : "N/A";

    const ath = s.athlete || {};
    document.getElementById("ath-name").textContent = ath.name || "–";
    document.getElementById("ath-meta").textContent =
      `${ath.sport} · ${ath.position} · Age ${ath.age} · ${ath.weight_kg} kg · ${ath.height_cm} cm`;
    const av = document.getElementById("ath-avatar");
    const initials = (ath.name || "?").split(" ").map(w => w[0]).slice(0,2).join("");
    av.textContent = initials;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Skill meters (animated progress bars)
  // ─────────────────────────────────────────────────────────────────────────
  function renderSkillMeters(s) {
    const items = [
      { label: "Strength",  val: s.avg_strength,  color: "#3b82f6" },
      { label: "Agility",   val: s.avg_agility,   color: "#22c55e" },
      { label: "Endurance", val: s.avg_endurance, color: "#f59e0b" },
      { label: "Recovery",  val: s.avg_recovery,  color: "#a855f7" },
      { label: "VO₂ Max",   val: s.avg_vo2_max,   color: "#06b6d4" },
    ];
    document.getElementById("skillMeters").innerHTML = items.map(it => `
      <div class="skill-bar-wrap">
        <div class="skill-label"><span>${it.label}</span><span>${it.val}</span></div>
        <div class="skill-bar-bg">
          <div class="skill-bar-fill" style="width:${Math.min(it.val,100)}%;background:${it.color}"></div>
        </div>
      </div>
    `).join("");
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Recent sessions table
  // ─────────────────────────────────────────────────────────────────────────
  function renderRecentSessions(sessions) {
    if (!sessions.length) {
      document.getElementById("recentSessions").innerHTML = '<p style="color:var(--muted)">No sessions found.</p>';
      return;
    }
    const rows = [...sessions].reverse().map(s => `
      <tr>
        <td>${s.session_date}</td>
        <td><span style="background:#1e2d40;padding:2px 8px;border-radius:12px;font-size:.78rem">${s.session_type}</span></td>
        <td>${s.duration_min} min</td>
        <td>${s.avg_heart_rate > 0 ? s.avg_heart_rate + " bpm" : "–"}</td>
        <td style="color:var(--muted);font-size:.78rem;max-width:140px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis">${s.notes || "–"}</td>
      </tr>
    `).join("");
    document.getElementById("recentSessions").innerHTML = `
      <div style="overflow-x:auto">
      <table style="width:100%;border-collapse:collapse">
        <thead><tr style="color:var(--muted);font-size:.78rem">
          <th style="padding:4px 8px;text-align:left">Date</th>
          <th style="padding:4px 8px;text-align:left">Type</th>
          <th style="padding:4px 8px;text-align:left">Duration</th>
          <th style="padding:4px 8px;text-align:left">HR</th>
          <th style="padding:4px 8px;text-align:left">Notes</th>
        </tr></thead>
        <tbody style="font-size:.83rem">${rows}</tbody>
      </table>
      </div>
    `;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Charts
  // ─────────────────────────────────────────────────────────────────────────
  const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { labels: { color: "#94a3b8", font: { size: 11 } } } },
  };

  function destroyChart(inst) { if (inst) inst.destroy(); return null; }

  function renderLineChart() {
    const metric = document.getElementById("metricSelect").value;
    const labels = chartData.labels || [];
    const data   = chartData[metric] || [];
    lineChartInst = destroyChart(lineChartInst);
    lineChartInst = new Chart(document.getElementById("lineChart"), {
      type: "line",
      data: {
        labels,
        datasets: [{
          label: metric.replace(/_/g," "),
          data,
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59,130,246,.12)",
          tension: 0.4,
          pointRadius: 3,
          fill: true,
        }]
      },
      options: {
        ...chartDefaults,
        scales: {
          x: { ticks: { color:"#475569", maxTicksLimit:8, font:{size:10} }, grid:{ color:"#1e2d40" } },
          y: { ticks: { color:"#475569", font:{size:10} }, grid:{ color:"#1e2d40" } },
        }
      }
    });
  }

  function updateLineChart() { renderLineChart(); }

  function renderRadarChart(s) {
    radarChartInst = destroyChart(radarChartInst);
    radarChartInst = new Chart(document.getElementById("radarChart"), {
      type: "radar",
      data: {
        labels: ["Strength","Agility","Endurance","Recovery","VO₂ Max"],
        datasets: [{
          label: s.athlete?.name || "Athlete",
          data: [s.avg_strength, s.avg_agility, s.avg_endurance, s.avg_recovery, s.avg_vo2_max],
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59,130,246,.2)",
          pointBackgroundColor: "#3b82f6",
        }]
      },
      options: {
        ...chartDefaults,
        scales: {
          r: {
            min: 40, max: 100,
            ticks: { color:"#475569", font:{size:9}, backdropColor:"transparent" },
            grid:  { color:"#1e2d40" },
            pointLabels: { color:"#94a3b8", font:{size:10} },
            angleLines: { color:"#1e2d40" },
          }
        }
      }
    });
  }

  function renderDonutChart(sessionTypes) {
    donutChartInst = destroyChart(donutChartInst);
    const labels = Object.keys(sessionTypes);
    const vals   = Object.values(sessionTypes);
    donutChartInst = new Chart(document.getElementById("donutChart"), {
      type: "doughnut",
      data: {
        labels,
        datasets: [{
          data: vals,
          backgroundColor: PALETTE.slice(0, labels.length),
          borderWidth: 2,
          borderColor: "#111827",
        }]
      },
      options: {
        ...chartDefaults,
        cutout: "65%",
        plugins: {
          legend: { position: "right", labels: { color: "#94a3b8", font:{size:11}, boxWidth:12 } }
        }
      }
    });
  }

  function renderWorkloadChart() {
    const labels   = chartData.labels || [];
    const calories = chartData.calories || [];
    workloadChartInst = destroyChart(workloadChartInst);
    workloadChartInst = new Chart(document.getElementById("workloadChart"), {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "Calories Burned",
          data: calories,
          backgroundColor: labels.map((_,i) => PALETTE[i % PALETTE.length] + "cc"),
          borderRadius: 4,
        }]
      },
      options: {
        ...chartDefaults,
        scales: {
          x: { ticks: { color:"#475569", maxTicksLimit:8, font:{size:10} }, grid:{ color:"#1e2d40" } },
          y: { ticks: { color:"#475569", font:{size:10} }, grid:{ color:"#1e2d40" } },
        }
      }
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Comparison section charts
  // ─────────────────────────────────────────────────────────────────────────
  async function loadComparisonCharts() {
    if (Object.keys(comparisonData).length === 0) {
      comparisonData = await fetch("/api/comparison").then(r => r.json());
    }
    const names   = Object.keys(comparisonData);
    const metrics = ["strength","agility","endurance","recovery","vo2_max"];

    // Radar
    compRadarInst = destroyChart(compRadarInst);
    compRadarInst = new Chart(document.getElementById("comparisonRadar"), {
      type: "radar",
      data: {
        labels: ["Strength","Agility","Endurance","Recovery","VO₂ Max"],
        datasets: names.map((n, i) => ({
          label: n,
          data: metrics.map(m => comparisonData[n][m]),
          borderColor: PALETTE[i],
          backgroundColor: PALETTE[i] + "30",
          pointBackgroundColor: PALETTE[i],
        }))
      },
      options: {
        ...chartDefaults,
        scales: {
          r: {
            min:40, max:100,
            ticks: { color:"#475569", backdropColor:"transparent" },
            grid:  { color:"#1e2d40" },
            pointLabels: { color:"#94a3b8", font:{size:11} },
            angleLines: { color:"#1e2d40" },
          }
        }
      }
    });

    // VO₂ Max bar
    vo2ChartInst = destroyChart(vo2ChartInst);
    vo2ChartInst = new Chart(document.getElementById("vo2Chart"), {
      type: "bar",
      data: {
        labels: names,
        datasets: [{
          label: "Avg VO₂ Max",
          data: names.map(n => comparisonData[n].vo2_max),
          backgroundColor: names.map((_,i) => PALETTE[i] + "cc"),
          borderRadius: 6,
        }]
      },
      options: {
        ...chartDefaults,
        indexAxis: "y",
        scales: {
          x: { ticks:{color:"#475569"}, grid:{color:"#1e2d40"}, min:40 },
          y: { ticks:{color:"#94a3b8"}, grid:{display:false} },
        }
      }
    });

    // Endurance vs Strength scatter
    scatterInst = destroyChart(scatterInst);
    scatterInst = new Chart(document.getElementById("scatterChart"), {
      type: "scatter",
      data: {
        datasets: names.map((n, i) => ({
          label: n,
          data: [{ x: comparisonData[n].endurance, y: comparisonData[n].strength }],
          backgroundColor: PALETTE[i],
          pointRadius: 8,
        }))
      },
      options: {
        ...chartDefaults,
        scales: {
          x: { title:{display:true,text:"Endurance",color:"#94a3b8"}, ticks:{color:"#475569"}, grid:{color:"#1e2d40"} },
          y: { title:{display:true,text:"Strength", color:"#94a3b8"}, ticks:{color:"#475569"}, grid:{color:"#1e2d40"} },
        }
      }
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // AI Agent calls
  // ─────────────────────────────────────────────────────────────────────────
  function setLoading(outputId, btnEl) {
    document.getElementById(outputId).innerHTML = '<span class="spinner-sm"></span> Running agent…';
    if (btnEl) btnEl.disabled = true;
  }

  function setResult(outputId, text, btnEl) {
    document.getElementById(outputId).innerHTML = marked.parse(text || "_No response_");
    if (btnEl) btnEl.disabled = false;
  }

  async function runAgent(type, btnEl) {
    if (!currentAthleteId) { alert("Please select an athlete first."); return; }
    const outputs = { analysis:"analysisOutput", training:"trainingOutput", coaching:"coachingOutput" };
    setLoading(outputs[type], btnEl);

    const getVal = id => { const el = document.getElementById(id); return el ? el.value : ""; };
    const bodies = {
      analysis: { athlete_id: currentAthleteId, question: getVal("analysisQ") },
      training: { athlete_id: currentAthleteId, focus:    getVal("trainingFocus") },
      coaching: { athlete_id: currentAthleteId, notes:    getVal("coachNotes") },
    };

    try {
      const r    = await fetch("/api/agent/" + type, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(bodies[type]),
      });
      const data = await r.json();
      setResult(outputs[type], data.result || data.error, btnEl);
    } catch (e) {
      setResult(outputs[type], "⚠️ Request failed: " + e.message, btnEl);
    }
  }

  async function runWorkflow(btnEl) {
    if (!currentAthleteId) { alert("Please select an athlete first."); return; }
    const panel = document.getElementById("workflowPanel");
    const out   = document.getElementById("workflowOutput");
    panel.style.display = "";
    out.innerHTML = '<div style="display:flex;align-items:center;gap:.6rem;color:var(--muted)"><span class="spinner-sm"></span> Running all 4 agents — this may take 10–20 seconds…</div>';
    panel.scrollIntoView({ behavior:"smooth" });
    if (btnEl) { btnEl.disabled = true; btnEl.innerHTML = '<span class="spinner-sm"></span> Running…'; }

    try {
      const r    = await fetch("/api/workflow", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ athlete_id: currentAthleteId }),
      });
      const data = await r.json();
      if (data.error) { out.innerHTML = "⚠️ " + data.error; return; }

      const combined = [
        "## 🔍 Performance Analysis\n" + (data.analysis || "_No response_"),
        "---\n## 🏋️ Training Recommendations\n" + (data.recommendations || "_No response_"),
        "---\n## 📋 Coaching Insights\n" + (data.coaching || "_No response_"),
      ].join("\n\n");
      out.innerHTML = marked.parse(combined);
    } catch (e) {
      out.innerHTML = "⚠️ Workflow failed: " + e.message;
    } finally {
      if (btnEl) { btnEl.disabled = false; btnEl.innerHTML = '<i class="bi bi-diagram-3"></i> Run Full Workflow'; }
    }
  }

  function downloadReport() {
    if (!currentAthleteId) { alert("Please select an athlete first."); return; }
    window.location.href = "/api/report/" + currentAthleteId;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CSV upload
  // ─────────────────────────────────────────────────────────────────────────
  async function uploadCSV(input) {
    const statusEl = document.getElementById("uploadStatus");
    if (!input.files.length) return;
    const form = new FormData();
    form.append("file", input.files[0]);
    statusEl.innerHTML = '<div class="alert-inline" style="background:#1e2d40;color:var(--text)"><span class="spinner-sm"></span> Uploading…</div>';

    try {
      const r    = await fetch("/api/upload", { method:"POST", body: form });
      const data = await r.json();
      if (r.ok) {
        statusEl.innerHTML = `<div class="alert-inline success"><i class="bi bi-check-circle me-1"></i>${data.message}</div>`;
        // Reload page after 2s to refresh sidebar
        setTimeout(() => location.reload(), 2000);
      } else {
        statusEl.innerHTML = `<div class="alert-inline danger"><i class="bi bi-exclamation-circle me-1"></i>${data.error}</div>`;
      }
    } catch (e) {
      statusEl.innerHTML = `<div class="alert-inline danger">Upload error: ${e.message}</div>`;
    }
    input.value = "";
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Boot: load first athlete
  // ─────────────────────────────────────────────────────────────────────────
  window.addEventListener("DOMContentLoaded", () => {
    const first = document.querySelector(".athlete-btn");
    if (first) {
      const id = first.getAttribute("onclick").match(/'([^']+)'/)[1];
      currentAthleteId = id;
      loadAthleteDashboard(id);
    }
    // Set today's date as default in session date pickers
    const today = new Date().toISOString().split("T")[0];
    const dp = document.getElementById("f-sdate");
    if (dp) dp.value = today;
    const dp2 = document.getElementById("s-date");
    if (dp2) dp2.value = today;
  });

  // ─────────────────────────────────────────────────────────────────────────
  // Add Athlete / Session sub-tab switcher
  // ─────────────────────────────────────────────────────────────────────────
  function switchAddTab(name, btn) {
    ["newathlete","newsession","csvupload"].forEach(t => {
      document.getElementById("addtab-" + t).style.display = (t === name) ? "" : "none";
    });
    // Only reset tabs inside the upload section, not all .section-tab elements on page
    document.querySelectorAll("#sec-upload .section-tab").forEach(b => b.classList.remove("active"));
    if (btn) btn.classList.add("active");
  }

  // ─────────────────────────────────────────────────────────────────────────
  // New Athlete form submit
  // ─────────────────────────────────────────────────────────────────────────
  function formVal(id)  { const el = document.getElementById(id); return el ? el.value.trim() : ""; }
  function formNum(id)  { const v = formVal(id); return v === "" ? 0 : parseFloat(v) || 0; }

  async function submitNewAthlete(btnEl) {
    const id   = formVal("f-id");
    const name = formVal("f-name");
    const sport = formVal("f-sport");
    const statusEl = document.getElementById("athleteFormStatus");

    // Client-side validation
    if (!id || !name || !sport) {
      statusEl.innerHTML = '<div class="alert-inline danger"><i class="bi bi-exclamation-circle me-1"></i>Athlete ID, Name and Sport are required.</div>';
      return;
    }
    if (/\s/.test(id)) {
      statusEl.innerHTML = '<div class="alert-inline danger"><i class="bi bi-exclamation-circle me-1"></i>Athlete ID must not contain spaces.</div>';
      return;
    }

    btnEl.disabled = true;
    statusEl.innerHTML = '<div class="alert-inline" style="background:#1e2d40;color:var(--text)"><span class="spinner-sm"></span> Saving…</div>';

    const body = {
      athlete_id: id, name, sport,
      position:   formVal("f-position"),
      age:        formNum("f-age"),
      weight_kg:  formNum("f-weight"),
      height_cm:  formNum("f-height"),
      // Optional first session
      session_date: formVal("f-sdate"),
      session_type: formVal("f-stype"),
      duration_min: formNum("f-dur"),
      avg_heart_rate: formNum("f-hr"),
      calories_burned: formNum("f-cal"),
      notes: formVal("f-notes"),
    };

    try {
      const r = await fetch("/api/add_athlete", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(body),
      });
      const d = await r.json();
      if (r.ok) {
        statusEl.innerHTML = `<div class="alert-inline success"><i class="bi bi-check-circle me-1"></i>${d.message} Page will reload in 2 seconds…</div>`;
        clearAthleteForm();
        setTimeout(() => location.reload(), 2000);
      } else {
        statusEl.innerHTML = `<div class="alert-inline danger"><i class="bi bi-exclamation-circle me-1"></i>${d.error}</div>`;
      }
    } catch(e) {
      statusEl.innerHTML = `<div class="alert-inline danger">Request failed: ${e.message}</div>`;
    }
    btnEl.disabled = false;
  }

  function clearAthleteForm() {
    ["f-id","f-name","f-position","f-sdate","f-notes"].forEach(id => {
      const el = document.getElementById(id); if (el) el.value = "";
    });
    ["f-sport","f-stype"].forEach(id => {
      const el = document.getElementById(id); if (el) el.selectedIndex = 0;
    });
    ["f-age","f-weight","f-height","f-dur","f-hr","f-cal"].forEach(id => {
      const el = document.getElementById(id); if (el) el.value = "";
    });
    document.getElementById("athleteFormStatus").innerHTML = "";
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Add Session form submit
  // ─────────────────────────────────────────────────────────────────────────
  async function submitNewSession(btnEl) {
    const aid   = formVal("s-athlete");
    const date  = formVal("s-date");
    const stype = formVal("s-type");
    const statusEl = document.getElementById("sessionFormStatus");

    if (!aid || !date || !stype) {
      statusEl.innerHTML = '<div class="alert-inline danger"><i class="bi bi-exclamation-circle me-1"></i>Athlete, Date and Session Type are required.</div>';
      return;
    }

    btnEl.disabled = true;
    statusEl.innerHTML = '<div class="alert-inline" style="background:#1e2d40;color:var(--text)"><span class="spinner-sm"></span> Saving…</div>';

    const body = {
      athlete_id:     aid,
      session_date:   date,
      session_type:   stype,
      duration_min:   formNum("s-dur"),
      distance_km:    formNum("s-dist"),
      avg_heart_rate: formNum("s-hr"),
      max_heart_rate: formNum("s-maxhr"),
      calories_burned: formNum("s-cal"),
      strength_score:  formNum("s-str"),
      endurance_score: formNum("s-end"),
      recovery_score:  formNum("s-rec"),
      agility_score:   formNum("s-agi"),
      vo2_max:         formNum("s-vo2"),
      match_rating:    formNum("s-rating"),
      match_goals:     formNum("s-goals"),
      match_assists:   formNum("s-assists"),
      injuries:        parseInt(formVal("s-injury") || "0"),
      notes:           formVal("s-notes"),
    };

    try {
      const r = await fetch("/api/add_session", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(body),
      });
      const d = await r.json();
      if (r.ok) {
        statusEl.innerHTML = `<div class="alert-inline success"><i class="bi bi-check-circle me-1"></i>${d.message}</div>`;
        clearSessionForm();
        // Refresh dashboard if this session is for the currently selected athlete
        if (aid === currentAthleteId) loadAthleteDashboard(aid);
      } else {
        statusEl.innerHTML = `<div class="alert-inline danger"><i class="bi bi-exclamation-circle me-1"></i>${d.error}</div>`;
      }
    } catch(e) {
      statusEl.innerHTML = `<div class="alert-inline danger">Request failed: ${e.message}</div>`;
    }
    btnEl.disabled = false;
  }

  function clearSessionForm() {
    ["s-date","s-notes"].forEach(id => { const el = document.getElementById(id); if (el) el.value = ""; });
    ["s-athlete","s-type"].forEach(id => { const el = document.getElementById(id); if (el) el.selectedIndex = 0; });
    ["s-dur","s-dist","s-hr","s-maxhr","s-cal","s-str","s-end","s-rec","s-agi","s-vo2","s-rating","s-goals","s-assists"].forEach(id => {
      const el = document.getElementById(id); if (el) el.value = "";
    });
    document.getElementById("sessionFormStatus").innerHTML = "";
  }

  </script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Jinja2 template helper (loop_color for avatar backgrounds)
# ---------------------------------------------------------------------------

@app.template_global()
def loop_color(idx: int) -> str:
    """Return a distinct background colour per loop index."""
    colors = [
        "#1e3a5f", "#14532d", "#3b1f6e", "#431407",
        "#164e63", "#713f12", "#4a044e", "#052e16",
    ]
    return colors[idx % len(colors)]

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  AthleteIQ AI – Athlete Performance Analysis Assistant")
    print("=" * 60)
    print(f"  Groq API Key : {'✓ Configured' if GROQ_API_KEY else '✗ Not set (running in Demo Mode)'}")
    print(f"  Groq Model   : {GROQ_MODEL}")
    print(f"  Database     : {DB_PATH}")
    print(f"  CSV Data     : {CSV_PATH}")
    print(f"  Demo Mode    : {DEMO_MODE}")
    print("-" * 60)
    print("  Initialising database…")
    init_db()
    print("  Database ready.")
    print("-" * 60)
    print("  Starting server → http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host="127.0.0.1", port=5000)
