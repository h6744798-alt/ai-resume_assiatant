# 📄 ATS Resume Checker

Upload a resume (PDF or DOCX) and get an ATS score, a score breakdown, prioritized improvements, missing keywords, and example bullet rewrites. Built with **Streamlit** and **Google Gemini Flash**.

## How the score works

The final score blends two parts:

- **AI assessment (65%)**: Gemini rates formatting, keywords, content quality, and readability.
- **Format checks (35%)**: deterministic rules (contact info, standard section headings, length, bullet points, quantified achievements, action verbs).

Paste a job description to get keyword matching against that specific role.

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key at https://aistudio.google.com/apikey, then either:

- set it as an environment variable: `export GEMINI_API_KEY="your-key"` (Windows PowerShell: `$env:GEMINI_API_KEY="your-key"`), or
- create `.streamlit/secrets.toml` containing `GEMINI_API_KEY = "your-key"`, or
- just paste it into the sidebar when the app is running.

Start the app:

```bash
streamlit run app.py
```

## Configuration

| Name | Purpose | Default |
|------|---------|---------|
| `GEMINI_API_KEY` | Your Gemini API key | none (sidebar input is shown if missing) |
| `GEMINI_MODEL` | Gemini model name | `gemini-2.5-flash` |

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public, or private with access granted to Streamlit).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose your repo and branch, and set the main file to `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**.

## Notes

- Scanned/image-only PDFs can't be read (ATS systems can't read them either). Use a text-based PDF.
- Resume text is sent to Google's Gemini API for analysis. The app doesn't store it.
- Never commit your API key to GitHub.
