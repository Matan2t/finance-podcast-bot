import json
import requests
import os
import google.generativeai as genai
from gtts import gTTS
from moviepy import AudioFileClip, ImageClip
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
# from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# ---------- Load companies ----------
with open("companies.json") as f:
    companies = json.load(f)["companies"]

# ---------- SEC fetch ----------
def get_latest_report(cik):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    headers = {"User-Agent": "finance-bot your@email.com"}
    r = requests.get(url, headers=headers)
    return r.text[:4000]  # חותך למניעת עומס

# ---------- Gemini summary ----------
genai.configure(api_key=os.environ["GEMINI_KEY"])
model = genai.GenerativeModel("gemini-pro")

def summarize(text, company):
    prompt = f"""
    תסכם כפודקאסט פיננסי בעברית באורך 3 דקות.
    החברה: {company}
    טקסט:
    {text}
    """
    response = model.generate_content(prompt)
    return response.text

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

# ---------- Main loop ----------
for company in companies:
    report = get_latest_report(company["cik"])
    script = summarize(report, company["name"])

    tts = gTTS(script, lang="he")
    tts.save("audio.mp3")

    audio = AudioFileClip("audio.mp3")
    image = ImageClip("background.png").set_duration(audio.duration)
    video = image.set_audio(audio)
    video.write_videofile("final.mp4", fps=24)

    upload_video("final.mp4", f"סיכום דוח - {company['name']}")