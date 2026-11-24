from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator

_MODEL_INSTANCE = None


def get_whisper_model():
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        _MODEL_INSTANCE = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
            cpu_threads=4,
        )
    return _MODEL_INSTANCE


def transcribe_audio(
    audio_path: str,
    target_subtitle_lang: str = "original",
    total_duration: float = 0.0,
    progress_cb=None,
) -> dict:
    """
    অডিও থেকে দ্রুত Word-level টাইমস্ট্যাম্পসহ ট্রান্সক্রিপশন তৈরি করে।
    """
    model = get_whisper_model()
    task = "translate" if target_subtitle_lang == "en" else "transcribe"

    # beam_size=1 ও best_of=1 দেওয়ায় স্পিড প্রায় ৩ গুণ বেড়ে যাবে
    segments_gen, info = model.transcribe(
        audio_path,
        task=task,
        beam_size=1,
        best_of=1,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    detected_lang = info.language
    segments = []

    translator_bn = None
    if target_subtitle_lang == "bn" and detected_lang != "bn":
        translator_bn = GoogleTranslator(source="auto", target="bn")

    for seg in segments_gen:
        text = seg.text.strip()
        if not text:
            continue

        if progress_cb and total_duration > 0:
            ratio = min(1.0, float(seg.end) / total_duration)
            pct = 40 + int(ratio * 20)
            progress_cb(
                pct,
                f"AI কথা শুনছে ও লিখছে... ({int(seg.end)}s / {int(total_duration)}s)",
            )

        words_data = []
        if seg.words:
            for w in seg.words:
                w_text = w.word.strip()
                if w_text:
                    words_data.append(
                        {
                            "word": w_text,
                            "start": float(w.start),
                            "end": float(w.end),
                        }
                    )

        if translator_bn is not None:
            try:
                translated_text = translator_bn.translate(text)
                if translated_text:
                    text = translated_text.strip()
                    words_data = _map_translated_words(text, float(seg.start), float(seg.end))
            except Exception:
                pass

        if not words_data:
            words_data = _map_translated_words(text, float(seg.start), float(seg.end))

        segments.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
                "words": words_data,
            }
        )

    return {
        "detected_language": detected_lang,
        "segments": segments,
    }


def _map_translated_words(text: str, start: float, end: float) -> list:
    tokens = text.split()
    if not tokens:
        return []
    total_dur = max(0.2, end - start)
    step = total_dur / len(tokens)
    mapped = []
    for i, tok in enumerate(tokens):
        mapped.append(
            {
                "word": tok,
                "start": start + i * step,
                "end": start + (i + 1) * step,
            }
        )
    return mapped