"""Language detection for crawled pages."""

from __future__ import annotations


def detect_language(text: str) -> tuple[str, float]:
    """Best-effort language detection. Returns (language_code, confidence).

    Falls back to ("en", 0.0) if the text is too short to detect reliably or
    if langdetect isn't installed - callers should treat a 0.0 confidence as
    "unknown" rather than a confident English classification.
    """
    text = text.strip()
    if len(text) < 20:
        return "en", 0.0

    try:
        from langdetect import DetectorFactory, detect_langs

        DetectorFactory.seed = 0  # deterministic results
        candidates = detect_langs(text)
        if not candidates:
            return "en", 0.0
        best = candidates[0]
        return best.lang, float(best.prob)
    except Exception:
        return "en", 0.0
