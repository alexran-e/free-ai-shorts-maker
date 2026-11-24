import os
import sys
import uuid
import zipfile
import threading
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from youtube_handler import download_video_and_audio, validate_youtube_url
from transcriber import transcribe_audio
from ai_analyzer import select_best_clips
from video_editor import render_short_clip

JOBS_ROOT = os.path.join(BASE_DIR, "temp_jobs")
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))
os.makedirs(JOBS_ROOT, exist_ok=True)

app = FastAPI(title="Free AI Shorts Maker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS = {}


class GenerateRequest(BaseModel):
    youtube_url: str
    clip_duration: int = 60
    num_clips: int = 3
    subtitle_enabled: bool = True
    subtitle_language: str = "original"
    smart_crop: bool = True


def _update_job(job_id: str, status: str, progress: int, message: str, clips=None):
    if job_id in JOBS:
        JOBS[job_id]["status"] = status
        JOBS[job_id]["progress"] = progress
        JOBS[job_id]["message"] = message
        if clips is not None:
            JOBS[job_id]["clips"] = clips


def _process_job(job_id: str, req: GenerateRequest):
    job_dir = os.path.join(JOBS_ROOT, job_id)
    os.makedirs(job_dir, exist_ok=True)

    try:
        _update_job(job_id, "downloading", 15, "YouTube ভিডিও ও অডিও প্রসেস করা হচ্ছে...")
        media = download_video_and_audio(req.youtube_url, job_dir)

        _update_job(job_id, "transcribing", 40, "AI (Whisper) দিয়ে কথাবার্তা টেক্সটে রূপান্তর হচ্ছে...")
        transcript = transcribe_audio(media["audio_path"], req.subtitle_language)
        import gc; gc.collect()

        if not transcript["segments"]:
            raise RuntimeError("ভিডিওতে কোনো স্পষ্ট কথাবার্তা পাওয়া যায়নি।")

        _update_job(job_id, "analyzing", 62, "AI দিয়ে ভিডিওর সেরা ও আকর্ষণীয় মুহূর্ত খোঁজা হচ্ছে...")
        selected_clips = select_best_clips(
            transcript["segments"],
            media["audio_path"],
            target_duration=req.clip_duration,
            num_clips=req.num_clips,
        )

        if not selected_clips:
            raise RuntimeError("উপযুক্ত কোনো ক্লিপ নির্বাচন করা সম্ভব হয়নি।")

        rendered_clips = []
        total = len(selected_clips)

        for idx, clip_data in enumerate(selected_clips):
            pct = 65 + int((idx / total) * 30)
            _update_job(
                job_id,
                "editing",
                pct,
                f"Shorts #{idx + 1}/{total} এডিট করা হচ্ছে (9:16 + Smart Crop + Subtitle)...",
            )
            out_name = f"short_{idx + 1}.mp4"
            actual_dur = render_short_clip(
                video_path=media["video_path"],
                clip_data=clip_data,
                output_filename=out_name,
                job_dir=job_dir,
                subtitle_enabled=req.subtitle_enabled,
                smart_crop=req.smart_crop,
            )

            rendered_clips.append(
                {
                    "id": idx + 1,
                    "title": clip_data["title"],
                    "score": clip_data["score"],
                    "reason": clip_data["reason"],
                    "duration": actual_dur,
                    "video_url": f"/api/clip/{job_id}/{out_name}",
                    "download_url": f"/api/download/{job_id}/{out_name}",
                }
            )

        _update_job(job_id, "completed", 100, "সবগুলো AI Shorts সফলভাবে তৈরি হয়েছে!", clips=rendered_clips)

    except Exception as exc:
        _update_job(job_id, "error", 0, f"সমস্যা হয়েছে: {str(exc)}")


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/generate")
def generate_shorts(req: GenerateRequest):
    if not validate_youtube_url(req.youtube_url):
        raise HTTPException(status_code=400, detail="সঠিক YouTube ভিডিও লিংক দিন।")

    if req.clip_duration not in (30, 60, 90):
        raise HTTPException(status_code=400, detail="Clip duration অবশ্যই 30, 60 বা 90 হতে হবে।")

    if req.num_clips not in (1, 2, 3, 5, 10):
        raise HTTPException(status_code=400, detail="Number of clips অবশ্যই 3, 5 বা 10 হতে হবে।")

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "status": "queued",
        "progress": 5,
        "message": "কাজটি কিউতে যুক্ত হয়েছে...",
        "clips": [],
    }

    worker = threading.Thread(target=_process_job, args=(job_id, req), daemon=True)
    worker.start()

    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def get_job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job খুঁজে পাওয়া যায়নি।")
    return JOBS[job_id]


@app.get("/api/clip/{job_id}/{filename}")
def stream_clip(job_id: str, filename: str):
    file_path = os.path.join(JOBS_ROOT, job_id, os.path.basename(filename))
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="ভিডিও ফাইল পাওয়া যায়নি।")
    return FileResponse(file_path, media_type="video/mp4")


@app.get("/api/download/{job_id}/{filename}")
def download_clip(job_id: str, filename: str):
    safe_name = os.path.basename(filename)
    file_path = os.path.join(JOBS_ROOT, job_id, safe_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="ভিডিও ফাইল পাওয়া যায়নি।")
    return FileResponse(file_path, media_type="video/mp4", filename=f"ai_{safe_name}")


@app.get("/api/download-all/{job_id}")
def download_all_clips(job_id: str):
    job_dir = os.path.join(JOBS_ROOT, job_id)
    if not os.path.exists(job_dir):
        raise HTTPException(status_code=404, detail="Job খুঁজে পাওয়া যায়নি।")

    zip_path = os.path.join(job_dir, "all_ai_shorts.zip")
    mp4_files = sorted(
        f for f in os.listdir(job_dir) if f.startswith("short_") and f.endswith(".mp4")
    )
    if not mp4_files:
        raise HTTPException(status_code=404, detail="ডাউনলোড করার মতো ক্লিপ পাওয়া যায়নি।")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in mp4_files:
            zf.write(os.path.join(job_dir, fname), arcname=fname)

    return FileResponse(zip_path, media_type="application/zip", filename="all_ai_shorts.zip")


if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")