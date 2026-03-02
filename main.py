import json
import requests
import os
from google import genai
from moviepy import AudioFileClip, ImageClip
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
# from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from earningcall_parser import parse_company_earningscall_transcript
from offline_hebrew_tts import synthesize_hebrew_audio
from get_earning_image import create_earnings_summary_image

# ---------- Gemini model (hard-coded) ----------
# Pick ONE model and set it to GEMINI_MODEL.
# Common options:
# - "gemini-2.0-flash"
# - "gemini-1.5-pro"
# - "gemini-1.5-flash"
GEMINI_MODEL = "gemini-2.5-flash"


def load_companies(path: str = "companies.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)["companies"]

def get_company_source_text(company: dict) -> str:
    # Prefer earnings-call transcript; fallback to SEC JSON snippet if missing.
    try:
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
def summarize(client: genai.Client, model: str, text: str, company: str, mock: bool) -> str:
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


def main() -> None:
    mock = False
    client = None
    if not mock:
      client = genai.Client(api_key=os.environ["GEMINI_KEY"])
    companies = load_companies()

    for company in companies:
        print(f"Processing company: {company['name']}")
        source_text = get_company_source_text(company)
        print(f"Source text: {source_text}")
        script = summarize(client, GEMINI_MODEL, source_text, company["name"], mock=mock)
        print(f"Summary script: {script}")

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

        # upload_video("final.mp4", f"סיכום דוח - {company['name']}")


if __name__ == "__main__":
    main()