import json
import requests
import os
import datetime as dt
from typing import Any
from companies_updater import add_or_update_by_tickers

# ---------- Gemini model (hard-coded) ----------
# Pick ONE model and set it to GEMINI_MODEL.
# Common options:
# - "gemini-2.0-flash"
# - "gemini-1.5-pro"
# - "gemini-1.5-flash"
GEMINI_MODEL = "gemini-2.5-flash"

COMPANIES_PATH = "companies.json"


def _is_truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip() in {"1", "true", "TRUE", "yes", "YES"}

def _now_israel() -> dt.datetime:
    """
    GitHub Actions runs in UTC by default; we want the "last 24 hours"
    window to be based on Israel time (Asia/Jerusalem).
    """
    try:
        from zoneinfo import ZoneInfo  # py3.9+

        return dt.datetime.now(ZoneInfo("Asia/Jerusalem"))
    except Exception:
        return dt.datetime.now(dt.timezone.utc)


def _reported_within_last_24h(company: dict, *, now: dt.datetime) -> bool:
    ec = company.get("earnings_call") or {}
    if not isinstance(ec, dict):
        return False

    d = ec.get("date")
    if not isinstance(d, str) or not d.strip():
        return False

    try:
        report_date = dt.date.fromisoformat(d.strip())
    except ValueError:
        return False

    threshold_date = (now - dt.timedelta(hours=24)).date()
    return report_date >= threshold_date


def load_companies_data(path: str = COMPANIES_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_companies_data(data: dict, path: str = COMPANIES_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def refresh_companies_json(path: str = COMPANIES_PATH) -> None:
    """
    Best-effort refresh of companies.json (name/cik/exchange/earnings_call).
    If refresh fails (network/etc), continue with existing file contents.
    """
    try:
        data = load_companies_data(path)
        companies = data.get("companies") or []
        tickers = [c.get("ticker") for c in companies if isinstance(c, dict) and c.get("ticker")]
        if not tickers:
            return

        sec_identity = os.environ.get("SEC_IDENTITY", "you@example.com")
        user_agent = f"finance-podcast-bot main/1.0 (contact: {sec_identity})"
        add_or_update_by_tickers(
            companies_path=path,
            tickers=[str(t) for t in tickers],
            user_agent=user_agent,
            update_existing=True,
            dry_run=False,
        )
    except Exception as e:
        print(f"[WARN] Could not refresh {path}: {e}")

def get_company_source_text(company: dict) -> str:
    # Prefer earnings-call transcript; fallback to SEC JSON snippet if missing.
    try:
        from earningcall_parser import parse_company_earningscall_transcript
        return parse_company_earningscall_transcript(company)
    except Exception:
        return get_latest_report(company["cik"])

# ---------- SEC fetch ----------
def get_latest_report(cik):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {"User-Agent": "finance-bot your@email.com"}
    r = requests.get(url, headers=headers)
    return r.text[:4000]  # חותך למניעת עומס

# ---------- Gemini summary ----------
def summarize(client: Any, model: str, text: str, company: str, mock: bool) -> str:
    if mock:
      return """ברוכים הבאים לפודקאסט של תותי כאן העתיד הוא העבר והעבר הוא כבר ממש מיושן! הכל תודות לגברת ביבי של כוחותיה והצלחותיה בהריון המטורף שהיא עוברת
"""
    else:
      prompt = f"""
    תסכם כפודקאסט פיננסי בעברית באורך 3 דקות.
    החברה: {company}
    טקסט:
    {text}
    """
      response = client.models.generate_content(model=model, contents=prompt)
      return response.text or ""

# ---------- YouTube ----------
def get_youtube_service():
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    creds = Credentials(
        None,
        refresh_token=os.environ["YOUTUBE_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["YOUTUBE_CLIENT_ID"],
        client_secret=os.environ["YOUTUBE_CLIENT_SECRET"],
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )

    # יוצר Access Token חדש בכל ריצה
    creds.refresh(Request())

    return build("youtube", "v3", credentials=creds)


def upload_video(file_path, title):
    from googleapiclient.http import MediaFileUpload

    youtube = get_youtube_service()

    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": title,
                "description": "דוח פיננסי יומי אוטומטי",
                "categoryId": "28"
            },
            "status": {
                "privacyStatus": "public"
            }
        },
        media_body=MediaFileUpload(file_path)
    )

    response = request.execute()
    print("Uploaded video ID:", response["id"])
    return response["id"]


def main() -> None:
    mock = False
    client: Any | None = None

    refresh_companies_json(COMPANIES_PATH)
    companies_data = load_companies_data(COMPANIES_PATH)
    companies: list[dict] = companies_data["companies"]
    now_il = _now_israel()

    for company in companies:
        if not _reported_within_last_24h(company, now=now_il):
            print(
                f"Skipping {company.get('ticker', '')}: not reported in last 24h "
                f"(earnings_call.date={((company.get('earnings_call') or {}).get('date'))!r})"
            )
            continue

        if client is None and not mock:
            try:
                from google import genai as _genai  # type: ignore
            except Exception:
                try:
                    import google.genai as _genai  # type: ignore
                except Exception as e:
                    raise RuntimeError(
                        "Gemini SDK not available. Install `google-genai` to run summaries."
                    ) from e
            client = _genai.Client(api_key=os.environ["GEMINI_KEY"])

        print(f"Processing company: {company['name']}")
        source_text = get_company_source_text(company)
        print(f"Source text: {source_text}")
        script = summarize(client, GEMINI_MODEL, source_text, company["name"], mock=mock)
        print(f"Summary script: {script}")

        from offline_hebrew_tts import synthesize_hebrew_audio
        from get_earning_image import create_earnings_summary_image
        from moviepy import AudioFileClip, ImageClip

        audio_path = synthesize_hebrew_audio(script)
        audio = AudioFileClip(audio_path)
        image_path = create_earnings_summary_image(company, summary_text=script)
        image = ImageClip(image_path).with_duration(audio.duration)
        video = image.with_audio(audio)
        video.write_videofile("final.mp4", fps=24)
        audio.close()
        try:
            os.remove(audio_path)
        except OSError:
            pass
        try:
            os.remove(image_path)
        except OSError:
            pass

        if _is_truthy_env("UPLOAD_YOUTUBE"):
            print(f"Uploading video to YouTube: {company['name']}")
            title = f"סיכום דוח - {company['name']}"
            upload_video("final.mp4", title)
        
        print(f"Finished processing company: {company['name']}")


if __name__ == "__main__":
    main()