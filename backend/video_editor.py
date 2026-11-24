import os
import subprocess
import cv2
import numpy as np


def _format_ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        secs += 1
        cs -= 100
    return f"{hrs}:{mins:02d}:{secs:02d}.{cs:02d}"


def _extract_active_subclips(segments: list, max_video_dur: float = 0.0) -> tuple:
    """
    ক্লিপের শুরু-শেষ টাইট করে এবং বড় নীরবতা (>3.5s) বাদ দিয়ে
    সর্বোচ্চ ২টি অংশে ভাগ করে রিম্যাপ করা টাইমস্ট্যাম্প তৈরি করে।
    """
    all_words = []
    for seg in segments:
        all_words.extend(seg.get("words", []))

    if not all_words:
        s0 = segments[0]["start"]
        e0 = segments[-1]["end"]
        if max_video_dur > 0:
            e0 = min(e0, max_video_dur - 0.05)
        return [(s0, max(s0 + 0.5, e0))], []

    chunks = []
    cur_start = max(0.0, all_words[0]["start"] - 0.15)
    cur_end = all_words[0]["end"]
    cur_words = [all_words[0]]

    for w in all_words[1:]:
        gap = w["start"] - cur_end
        if gap > 3.5 and len(chunks) < 1:
            chunks.append((cur_start, cur_end + 0.15, cur_words))
            cur_start = max(0.0, w["start"] - 0.10)
            cur_words = [w]
        else:
            cur_words.append(w)
        cur_end = max(cur_end, w["end"])

    chunks.append((cur_start, cur_end + 0.20, cur_words))

    time_ranges = []
    remapped_words = []
    timeline_offset = 0.0

    for c_start, c_end, c_words in chunks:
        if max_video_dur > 0:
            c_end = min(c_end, max_video_dur - 0.05)
        if c_end - c_start < 0.5:
            continue

        time_ranges.append((c_start, c_end))
        for w in c_words:
            w_s = max(0.0, w["start"] - c_start + timeline_offset)
            w_e = max(w_s + 0.05, w["end"] - c_start + timeline_offset)
            remapped_words.append(
                {
                    "word": w["word"],
                    "start": w_s,
                    "end": w_e,
                }
            )
        timeline_offset += (c_end - c_start)

    if not time_ranges:
        s0 = max(0.0, segments[0]["start"])
        e0 = segments[-1]["end"]
        if max_video_dur > 0:
            e0 = min(e0, max_video_dur - 0.05)
        return [(s0, max(s0 + 0.5, e0))], []

    return time_ranges, remapped_words


def _detect_smart_crop_x(video_path: str, start_t: float, end_t: float, width: int, height: int, crop_w: int) -> int:
    """
    OpenCV দিয়ে স্পিকারের চেহারা শনাক্ত করে 9:16 ক্রপ উইন্ডোর X-পজিশন নির্ণয় করে।
    """
    default_x = max(0, (width - crop_w) // 2)
    try:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        face_cascade = cv2.CascadeClassifier(cascade_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return default_x

        face_centers = []
        sample_times = np.linspace(start_t, max(start_t + 0.5, end_t), num=5)

        for t in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(40, 40))
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_centers.append(x + w / 2.0)

        cap.release()
        if not face_centers:
            return default_x

        median_cx = float(np.median(face_centers))
        crop_x = int(median_cx - crop_w / 2.0)
        return max(0, min(crop_x, width - crop_w))
    except Exception:
        return default_x


def _generate_highlighted_ass(words: list, ass_path: str):
    """
    গুরুত্বপূর্ণ ও চলমান শব্দ হলুদ রঙে হাইলাইট করে ASS সাবটাইটেল ফাইল তৈরি করে।
    """
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 720
PlayResY: 1280
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Noto Sans,44,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,40,40,200,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    chunk_size = 4
    for i in range(0, len(words), chunk_size):
        group = words[i : i + chunk_size]
        for active_idx, active_word in enumerate(group):
            ev_start = active_word["start"]
            if active_idx + 1 < len(group):
                ev_end = group[active_idx + 1]["start"]
            else:
                ev_end = active_word["end"] + 0.12

            if ev_end <= ev_start:
                ev_end = ev_start + 0.15

            styled_tokens = []
            for j, tok in enumerate(group):
                safe_word = tok["word"].replace("{", "").replace("}", "")
                is_important = len(safe_word) >= 7 or any(ch.isdigit() for ch in safe_word)
                if j == active_idx:
                    styled_tokens.append(f"{{\\c&H0000FFFF&}}{safe_word}{{\\c&H00FFFFFF&}}")
                elif is_important:
                    styled_tokens.append(f"{{\\c&H00FFFF80&}}{safe_word}{{\\c&H00FFFFFF&}}")
                else:
                    styled_tokens.append(safe_word)

            dialogue_text = " ".join(styled_tokens)
            lines.append(
                f"Dialogue: 0,{_format_ass_time(ev_start)},{_format_ass_time(ev_end)},Default,,0,0,0,,{dialogue_text}"
            )

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(lines) + "\n")


def render_short_clip(
    video_path: str,
    clip_data: dict,
    output_filename: str,
    job_dir: str,
    subtitle_enabled: bool = True,
    smart_crop: bool = True,
) -> float:
    """
    চূড়ান্ত 9:16 HD MP4 Shorts দ্রুত রেন্ডার করে এবং আসল দৈর্ঘ্য রিটার্ন করে।
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0.0
    max_video_dur = (frame_count / fps) if (frame_count > 0 and fps > 0) else 0.0
    cap.release()

    target_ratio = 9.0 / 16.0
    if (width / height) > target_ratio:
        crop_h = height - (height % 2)
        crop_w = int(crop_h * target_ratio)
        crop_w -= crop_w % 2
        crop_y = 0
        if smart_crop:
            crop_x = _detect_smart_crop_x(
                video_path, clip_data["start"], clip_data["end"], width, height, crop_w
            )
        else:
            crop_x = max(0, (width - crop_w) // 2)
    else:
        crop_w = width - (width % 2)
        crop_h = int(crop_w / target_ratio)
        crop_h -= crop_h % 2
        crop_x = 0
        crop_y = max(0, (height - crop_h) // 2)

    time_ranges, remapped_words = _extract_active_subclips(clip_data["segments"], max_video_dur)
    total_duration = sum(e - s for s, e in time_ranges)

    ass_filename = output_filename.replace(".mp4", ".ass")
    ass_path = os.path.join(job_dir, ass_filename)
    use_ass = subtitle_enabled and len(remapped_words) > 0
    if use_ass:
        _generate_highlighted_ass(remapped_words, ass_path)

    crop_scale_filter = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale=720:1280"
    if use_ass:
        video_post = f"{crop_scale_filter},ass={ass_filename}"
    else:
        video_post = crop_scale_filter

    if len(time_ranges) == 1:
        s, e = time_ranges[0]
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{s:.3f}",
            "-to",
            f"{e:.3f}",
            "-i",
            "source.mp4",
            "-vf",
            video_post,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast", "-threads", "2",
            "-crf",
            "24",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            output_filename,
        ]
    else:
        filter_parts = []
        concat_inputs = []
        for idx, (s, e) in enumerate(time_ranges):
            filter_parts.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{idx}];")
            filter_parts.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{idx}];")
            concat_inputs.append(f"[v{idx}][a{idx}]")

        concat_str = "".join(concat_inputs)
        filter_parts.append(f"{concat_str}concat=n={len(time_ranges)}:v=1:a=1[vc][ac];")
        filter_parts.append(f"[vc]{video_post}[vout]")
        filter_complex = "".join(filter_parts)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            "source.mp4",
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-map",
            "[ac]",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast", "-threads", "2",
            "-crf",
            "24",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            output_filename,
        ]

    res = subprocess.run(cmd, cwd=job_dir, capture_output=True, text=True)
    if res.returncode != 0:
        err_tail = (res.stderr or "")[-250:]
        raise RuntimeError(f"ভিডিও এডিট ত্রুটি: {err_tail}")

    return total_duration