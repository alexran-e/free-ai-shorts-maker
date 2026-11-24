import wave
import json
import numpy as np
import requests


HOOK_KEYWORDS = [
    # English hooks: Q&A, storytelling, educational, funny, emotional
    "why", "how", "secret", "truth", "mistake", "never", "always", "important",
    "story", "remember", "lesson", "question", "answer", "actually", "imagine",
    "because", "best", "worst", "amazing", "shocking", "learn", "tip", "funny",
    "laugh", "heart", "cry", "life", "success", "fail",
    # Bengali hooks
    "কেন", "কিভাবে", "কীভাবে", "গল্প", "গুরুত্বপূর্ণ", "রহস্য", "সত্য", "ভুল",
    "মনে রাখবেন", "শিক্ষা", "প্রশ্ন", "উত্তর", "আসলে", "সবচেয়ে", "সবচেয়ে",
    "দারুণ", "মজার", "হাসি", "কষ্ট", "আবেগ", "সফল", "জীবন", "কারণ",
    "জানেন", "দেখুন", "শুনুন",
]


def _compute_audio_energy_profile(audio_path: str, bin_sec: float = 1.0) -> np.ndarray:
    with wave.open(audio_path, "rb") as wf:
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

    if len(samples) == 0:
        return np.ones(1, dtype=np.float32)

    samples_per_bin = max(1, int(rate * bin_sec))
    num_bins = max(1, len(samples) // samples_per_bin)
    trimmed = samples[: num_bins * samples_per_bin].reshape(num_bins, samples_per_bin)
    rms = np.sqrt(np.mean(trimmed ** 2, axis=1))
    max_rms = np.max(rms) if np.max(rms) > 0 else 1.0
    return rms / max_rms


def _build_candidate_windows(segments: list, target_duration: int) -> list:
    """
    বাক্যের শুরু ও শেষ ঠিক রেখে target_duration-এর কাছাকাছি ক্যান্ডিডেট ক্লিপ তৈরি করে।
    """
    if not segments:
        return []

    min_dur = max(15.0, target_duration * 0.65)
    max_dur = target_duration * 1.18

    candidates = []
    n = len(segments)
    i = 0

    while i < n:
        start_time = segments[i]["start"]
        window_segs = []
        j = i

        while j < n:
            seg = segments[j]
            current_dur = seg["end"] - start_time
            if current_dur > max_dur and window_segs:
                break
            window_segs.append(seg)
            if current_dur >= target_duration:
                break
            j += 1

        if window_segs:
            end_time = window_segs[-1]["end"]
            dur = end_time - start_time
            if dur >= min_dur or (i == 0 and n <= 3):
                candidates.append(
                    {
                        "start": start_time,
                        "end": end_time,
                        "duration": dur,
                        "segments": window_segs,
                        "text": " ".join(s["text"] for s in window_segs),
                    }
                )

        step_target = start_time + (target_duration * 0.35)
        next_i = i + 1
        while next_i < n and segments[next_i]["start"] < step_target:
            next_i += 1
        i = next_i

    return candidates


def _score_candidate_heuristic(cand: dict, energy_profile: np.ndarray) -> tuple:
    text = cand["text"]
    text_lower = text.lower()
    dur = max(1.0, cand["duration"])

    # ১. প্রশ্ন ও আবেগ চিহ্ন স্কোর
    question_count = text.count("?")
    exclaim_count = text.count("!")
    punct_score = min(20, question_count * 8 + exclaim_count * 5)

    # ২. হুক ও গুরুত্বপূর্ণ বিষয় স্কোর
    keyword_hits = sum(1 for kw in HOOK_KEYWORDS if kw in text_lower)
    keyword_score = min(30, keyword_hits * 5)

    # ৩. কথার ঘনত্ব (অপ্রয়োজনীয় নীরবতা কম থাকলে বেশি স্কোর)
    word_count = len(text.split())
    wps = word_count / dur
    density_score = min(20, max(0, (wps - 0.8) * 10))

    # ৪. অডিও এনার্জি (হাসি, উত্তেজনা বা আবেগপূর্ণ বক্তব্য)
    s_idx = max(0, int(cand["start"]))
    e_idx = min(len(energy_profile), int(cand["end"]) + 1)
    if e_idx > s_idx:
        slice_energy = energy_profile[s_idx:e_idx]
        energy_score = float(np.mean(slice_energy) * 18 + np.std(slice_energy) * 20)
        energy_score = min(30, energy_score)
    else:
        energy_score = 10

    total_score = int(min(99, max(35, punct_score + keyword_score + density_score + energy_score)))

    reasons = []
    if question_count > 0:
        reasons.append("প্রশ্ন-উত্তর ও কৌতূহলপূর্ণ অংশ")
    if keyword_hits >= 2:
        reasons.append("গুরুত্বপূর্ণ বক্তব্য ও আকর্ষণীয় গল্প/শিক্ষণীয় বিষয়")
    if energy_score >= 18:
        reasons.append("উচ্চ আবেগ ও প্রাণবন্ত ভয়েস ডেলিভারি")
    if not reasons:
        reasons.append("সাবলীল আলোচনা ও দর্শক ধরে রাখার মতো অংশ")

    return total_score, " • ".join(reasons)


def _try_ollama_rerank(candidates: list) -> dict:
    """
    যদি সার্ভারে ফ্রি ওপেন-সোর্স Ollama LLM চালু থাকে, তবে সেটি দিয়ে আরও উন্নত র‍্যাঙ্কিং করে।
    """
    try:
        ping = requests.get("http://localhost:11434/api/tags", timeout=1.5)
        if ping.status_code != 200:
            return {}
        models = ping.json().get("models", [])
        if not models:
            return {}
        model_name = models[0]["name"]

        summaries = [
            {"id": idx, "text": c["text"][:450]}
            for idx, c in enumerate(candidates[:15])
        ]
        prompt = (
            "You are a viral video editor. Rate each transcript clip from 1-100 for Shorts potential "
            "(story, funny, educational, Q&A, emotional) and give a short catchy title in the clip's language. "
            "Return JSON list: [{\"id\": 0, \"score\": 88, \"title\": \"...\", \"reason\": \"...\"}].\n"
            f"Clips: {json.dumps(summaries, ensure_ascii=False)}"
        )

        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False, "format": "json"},
            timeout=25,
        )
        if resp.status_code == 200:
            raw = resp.json().get("response", "[]")
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return {item["id"]: item for item in parsed if isinstance(item, dict) and "id" in item}
    except Exception:
        pass
    return {}


def select_best_clips(
    segments: list,
    audio_path: str,
    target_duration: int = 60,
    num_clips: int = 3,
) -> list:
    if not segments:
        return []

    energy_profile = _compute_audio_energy_profile(audio_path)
    candidates = _build_candidate_windows(segments, target_duration)

    if not candidates:
        total_start = segments[0]["start"]
        total_end = segments[-1]["end"]
        return [
            {
                "title": "Highlight Clip #1",
                "score": 80,
                "reason": "মূল বক্তব্যের সারসংক্ষেপ",
                "start": total_start,
                "end": total_end,
                "duration": total_end - total_start,
                "segments": segments,
            }
        ]

    ollama_results = _try_ollama_rerank(candidates)

    scored = []
    for idx, cand in enumerate(candidates):
        h_score, h_reason = _score_candidate_heuristic(cand, energy_profile)
        first_sentence = cand["text"].split(".")[0].split("।")[0].strip()
        short_title = (first_sentence[:48] + "...") if len(first_sentence) > 48 else first_sentence

        if idx in ollama_results:
            llm_item = ollama_results[idx]
            final_score = int((h_score * 0.4) + (int(llm_item.get("score", h_score)) * 0.6))
            title = llm_item.get("title") or short_title or f"Highlight #{idx + 1}"
            reason = llm_item.get("reason") or h_reason
        else:
            final_score = h_score
            title = short_title or f"Highlight #{idx + 1}"
            reason = h_reason

        scored.append(
            {
                **cand,
                "score": min(99, max(40, final_score)),
                "title": title,
                "reason": reason,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    selected = []
    for cand in scored:
        overlaps = any(
            not (cand["end"] <= sel["start"] + 3.0 or cand["start"] >= sel["end"] - 3.0)
            for sel in selected
        )
        if not overlaps:
            selected.append(cand)
        if len(selected) >= num_clips:
            break

    return selected