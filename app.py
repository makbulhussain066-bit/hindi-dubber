import os
import subprocess
import uuid
import whisper
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from yt_dlp import YoutubeDL
from deep_translator import GoogleTranslator
from gtts import gTTS

app = FastAPI()

# Frontend ke liye static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

WORKSPACE = "workspace"
os.makedirs(WORKSPACE, exist_ok=True)

print("Loading AI Whisper model...")
model = whisper.load_model("base")
print("AI Model loaded.")

class VideoRequest(BaseModel):
    url: str

@app.post("/api/process")
async def process_video(request: VideoRequest):
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(WORKSPACE, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    try:
        # STEP 1: Video Download
        print(f"[{job_id}] Downloading video...")
        video_path = os.path.join(job_dir, "original_video.mp4")
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            'outtmpl': video_path,
            'quiet': True
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([request.url])

        # STEP 2: Audio nikalna
        print(f"[{job_id}] Extracting audio...")
        audio_path = os.path.join(job_dir, "original_audio.wav")
        subprocess.run([
            "ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path, "-y"
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # STEP 3: English sun kar Text banana
        print(f"[{job_id}] Transcribing audio...")
        result = model.transcribe(audio_path)
        english_text = result["text"]

        # STEP 4: Hindi mein Translate karna
        print(f"[{job_id}] Translating to Hindi...")
        hindi_text = GoogleTranslator(source='en', target='hi').translate(english_text)

        # STEP 5: Hindi Awaaz banana
        print(f"[{job_id}] Generating Hindi audio...")
        hindi_audio_path = os.path.join(job_dir, "hindi_audio.mp3")
        tts = gTTS(text=hindi_text, lang='hi', slow=False)
        tts.save(hindi_audio_path)

        # STEP 6: Nayi awaaz ko Video mein jodna
        print(f"[{job_id}] Merging video and audio...")
        final_video_path = os.path.join(job_dir, "final_video.mp4")
        subprocess.run([
            "ffmpeg", "-i", video_path, "-i", hindi_audio_path,
            "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", 
            "-shortest", 
            final_video_path, "-y"
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return {
            "status": "success",
            "video_url": f"/api/download/{job_id}/final_video.mp4",
            "audio_url": f"/api/download/{job_id}/hindi_audio.mp3"
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    file_path = os.path.join(WORKSPACE, job_id, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/octet-stream", filename=filename)
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')
