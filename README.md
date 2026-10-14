# AthleteIQ-AI
AI-Powered Athlete Performance Analysis Agent using IBM Bob, watsonx.ai and Granite Models
# AthleteIQ AI – Athlete Performance Analysis Assistant

A full-stack AI-powered web application for analysing athlete performance data using a **4-agent multi-agent pipeline** powered by the [Groq API](https://console.groq.com).

> **Demo Mode:** The application works fully without a Groq API key using intelligent rule-based AI responses. Add your key anytime to unlock full LLM-powered analysis.

---

## ✨ Features

| Feature | Details |
|---|---|
| **4 AI Agents** | Data, Analysis, Training, Coaching in a chained workflow |
| **Dashboard** | KPI cards, time-series charts, radar skill profile |
| **Charts** | Line, Radar, Doughnut, Bar, Scatter via Chart.js |
| **AI Analysis** | Performance trends, strengths & weaknesses |
| **Training Plans** | Personalised 7-day weekly training schedules |
| **Coaching Reports** | Downloadable `.txt` performance reports |
| **Athlete Comparison** | Multi-athlete radar & scatter charts |
| **CSV Upload** | Upload new athlete data at runtime |
| **SQLite Storage** | All data persisted locally |
| **Demo Mode** | Fully functional without an API key |

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9 or higher
- pip

### 2. Clone / download the project

```bash
# If using git:
git clone <your-repo-url>
cd athleteiq-ai

# Or just place all files in a folder and cd into it
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment (optional but recommended)

```bash
cp .env.example .env
```

Open `.env` and fill in your Groq API key:

```
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama3-8b-8192
```

If you skip this step the app runs in **Demo Mode** — all features work but AI responses are rule-based rather than LLM-generated.

### 5. Run the application

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

---

## 📁 Project Structure

```
athleteiq-ai/
│
├── app.py              ← Flask application (all backend logic + HTML template)
├── athletes.csv        ← Sample data for 5 athletes (50 sessions)
├── requirements.txt    ← Python dependencies
├── .env.example        ← Environment variable template
├── README.md           ← This file
│
└── athleteiq.db        ← SQLite database (auto-created on first run)
```

---

## 🤖 The Four AI Agents

### Agent 1 – Athlete Data Agent
- Reads athlete session records from SQLite (seeded from `athletes.csv`)
- Aggregates metrics with **Pandas** (totals, averages, trends)
- Provides structured context to all downstream agents
- Endpoint: `GET /api/athlete/<id>/summary`

### Agent 2 – Performance Analysis Agent
- Identifies trends, strengths, and weaknesses from the aggregated data
- Flags risk indicators (low recovery, injuries, declining metrics)
- Provides one concrete, evidence-based recommendation
- Endpoint: `POST /api/agent/analysis`

### Agent 3 – Training Recommendation Agent
- Generates a personalised **7-day training plan**
- Focuses on the athlete's weakest metrics
- Includes exercises, intensity levels, and recovery tips
- Endpoint: `POST /api/agent/training`

### Agent 4 – Coaching Insights Agent
- Produces a formal coaching report with 7 structured sections
- Generates a downloadable plain-text performance report
- Endpoint: `POST /api/agent/coaching`

### Multi-Agent Workflow Orchestrator
- Chains all 4 agents sequentially: Data → Analysis → Training → Coaching
- Returns a combined output rendered in the dashboard
- Endpoint: `POST /api/workflow`

---

## 📊 Sample Athletes

The included `athletes.csv` contains data for 5 athletes across 5 sports:

| Athlete | Sport | Position |
|---|---|---|
| Marcus Johnson | Soccer | Forward |
| Sofia Reyes | Swimming | Freestyle |
| Jamal Williams | Basketball | Point Guard |
| Priya Sharma | Tennis | Singles |
| Lucas Meyer | Cycling | Road |

Each athlete has ~10 sessions with full metrics: heart rate, calories, VO₂ max, strength, agility, endurance, recovery, speed, and match statistics.

---

## 📤 CSV Upload Format

To upload your own athlete data, your CSV must include at minimum:

| Column | Required | Type |
|---|---|---|
| `athlete_id` | ✅ | text (e.g. `A001`) |
| `name` | ✅ | text |
| `sport` | ✅ | text |
| `session_date` | ✅ | `YYYY-MM-DD` |
| `session_type` | ✅ | text |
| `position` | optional | text |
| `age` | optional | integer |
| `weight_kg` | optional | float |
| `height_cm` | optional | float |
| `duration_min` | optional | float |
| `distance_km` | optional | float |
| `avg_heart_rate` | optional | float |
| `max_heart_rate` | optional | float |
| `calories_burned` | optional | float |
| `sprint_speed_kmh` | optional | float |
| `vo2_max` | optional | float |
| `strength_score` | optional | 0–100 |
| `agility_score` | optional | 0–100 |
| `endurance_score` | optional | 0–100 |
| `recovery_score` | optional | 0–100 |
| `match_goals` | optional | integer |
| `match_assists` | optional | integer |
| `match_rating` | optional | float (0–10) |
| `injuries` | optional | 0 or 1 |
| `notes` | optional | text |

---

## 🔑 Getting a Free Groq API Key

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to **API Keys** → **Create API Key**
4. Copy the key into your `.env` file as `GROQ_API_KEY`

Groq's free tier provides fast inference with generous rate limits — more than sufficient for this application.

---

## ⚙️ API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Main dashboard |
| `GET` | `/api/athletes` | List all athletes |
| `GET` | `/api/athlete/<id>/summary` | Full summary for one athlete |
| `GET` | `/api/athlete/<id>/charts` | Chart.js-ready time-series data |
| `GET` | `/api/comparison` | Comparison data for all athletes |
| `POST` | `/api/workflow` | Run all 4 agents |
| `POST` | `/api/agent/analysis` | Agent 2 analysis |
| `POST` | `/api/agent/training` | Agent 3 training plan |
| `POST` | `/api/agent/coaching` | Agent 4 coaching insights |
| `GET` | `/api/report/<id>` | Download text report |
| `POST` | `/api/upload` | Upload a CSV file |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3, Flask |
| AI | Groq API (`llama3-8b-8192` default) |
| Data | Pandas, SQLite |
| Frontend | Bootstrap 5, Chart.js 4, Marked.js |
| Styling | CSS custom properties, dark sports theme |
| Icons | Bootstrap Icons |

---

## 🐛 Troubleshooting

**App won't start:**
- Make sure Python 3.9+ is installed: `python --version`
- Reinstall dependencies: `pip install -r requirements.txt`

**No athletes in sidebar:**
- Ensure `athletes.csv` is in the same folder as `app.py`
- Delete `athleteiq.db` and restart to re-seed from CSV

**Demo Mode even with API key:**
- Check `.env` file exists and `GROQ_API_KEY` is set (no quotes needed)
- Verify key starts with `gsk_`

**Chart not loading:**
- Charts require a browser with JavaScript enabled
- Check browser console for any CDN loading errors

---

## 📄 License

MIT License – free to use, modify, and distribute.

---

*Built with Python · Flask · Groq API · Chart.js · Bootstrap 5*
