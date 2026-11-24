import os
import re
import subprocess
import yt_dlp


YOUTUBE_REGEX = re.compile(
    r"^(https?://)?(www\.|m\.)?(youtube\.com/(watch\?v=|shorts/|live/)|youtu\.be/)[\w\-]+"
)


def validate_youtube_url(url: str) -> bool:
    return bool(YOUTUBE_REGEX.match(url.strip()))


def download_video_and_audio(youtube_url: str, job_dir: str) -> dict:
    """
    YouTube থেকে ভিডিও ডাউনলোড করে এবং 16kHz mono WAV অডিও এক্সট্রাক্ট করে।
    """
    if not validate_youtube_url(youtube_url):
        raise ValueError("অবৈধ YouTube URL। অনুগ্রহ করে সঠিক লিংক দিন।")

    os.makedirs(job_dir, exist_ok=True)
    video_template = os.path.join(job_dir, "source.%(ext)s")
    final_video_path = os.path.join(job_dir, "source.mp4")
    audio_path = os.path.join(job_dir, "audio.wav")

    js_runtimes = {"deno": {}, "node": {}}
    deno_bin = os.path.expanduser("~/.deno/bin/deno")
    if os.path.exists(deno_bin):
        js_runtimes["deno"] = {"path": deno_bin}

    ydl_opts = {
        "format": "bv*[height<=1080]+ba/b[height<=1080]/best",
        "merge_output_format": "mp4",
        "outtmpl": video_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "force_ipv4": True,
        "sleep_requests": 1,
        "js_runtimes": js_runtimes,
        "remote_components": ["ejs:github"],
        "max_filesize": 1500 * 1024 * 1024,
    }

    cookies_file = os.path.join(os.path.dirname(__file__), "cookies.txt")
    if os.path.exists(cookies_file):
        ydl_opts["cookiefile"] = cookies_file

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        title = info.get("title", "YouTube Video")
        duration = float(info.get("duration") or 0)

    if not os.path.exists(final_video_path):
        for fname in os.listdir(job_dir):
            if fname.startswith("source.") and not fname.endswith(".part"):
                src = os.path.join(job_dir, fname)
                subprocess.run(
                    ["ffmpeg", "-y", "-i", src, "-c:v", "libx264", "-c:a", "aac", final_video_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                break

    if not os.path.exists(final_video_path):
        raise RuntimeError("ভিডিও ডাউনলোড ব্যর্থ হয়েছে। ভিডিওটি প্রাইভেট বা রেস্ট্রিক্টেড হতে পারে।")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            final_video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            audio_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return {
        "title": title,
        "duration": duration,
        "video_path": final_video_path,
        "audio_path": audio_path,
    }