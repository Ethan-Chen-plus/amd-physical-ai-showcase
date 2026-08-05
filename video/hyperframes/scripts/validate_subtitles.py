#!/usr/bin/env python3
"""Validate subtitle timing, phrase granularity, and 16:9 safe-caption limits."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILM_DURATION = 299.0
TIMESTAMP = re.compile(
    r"^(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> "
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})$"
)


def seconds(groups: tuple[str, ...]) -> float:
    hours, minutes, secs, millis = map(int, groups)
    return hours * 3600 + minutes * 60 + secs + millis / 1000


def parse(path: Path) -> list[dict[str, object]]:
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig").strip())
    cues = []
    for expected_index, block in enumerate(blocks, 1):
        lines = block.splitlines()
        if len(lines) < 3 or int(lines[0]) != expected_index:
            raise ValueError(f"{path.name}: malformed cue {expected_index}")
        match = TIMESTAMP.match(lines[1])
        if not match:
            raise ValueError(f"{path.name}: invalid timestamp in cue {expected_index}")
        values = match.groups()
        cues.append(
            {
                "start": seconds(values[:4]),
                "end": seconds(values[4:]),
                "lines": lines[2:],
            }
        )
    return cues


def main() -> None:
    paths = {
        "en": ROOT / "subtitles" / "en.srt",
        "zh": ROOT / "subtitles" / "zh.srt",
        "bilingual": ROOT / "subtitles" / "bilingual.srt",
    }
    parsed = {name: parse(path) for name, path in paths.items()}
    counts = {name: len(cues) for name, cues in parsed.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"cue count mismatch: {counts}")

    maximum_duration = 0.0
    maximum_en = 0
    maximum_zh = 0
    previous_end = -1.0
    for index, (en, zh, bilingual) in enumerate(
        zip(parsed["en"], parsed["zh"], parsed["bilingual"], strict=True), 1
    ):
        timing = (en["start"], en["end"])
        if timing != (zh["start"], zh["end"]) or timing != (
            bilingual["start"],
            bilingual["end"],
        ):
            raise ValueError(f"cue {index}: language timing mismatch")
        if en["start"] < previous_end:
            raise ValueError(f"cue {index}: overlaps previous cue")
        if en["start"] < 0 or en["end"] > FILM_DURATION or en["end"] <= en["start"]:
            raise ValueError(f"cue {index}: invalid film bounds")
        cue_duration = en["end"] - en["start"]
        if cue_duration < 0.75 or cue_duration > 9.0:
            raise ValueError(
                f"cue {index}: duration {cue_duration:.3f}s outside 0.75-9.0s"
            )
        if (
            len(en["lines"]) != 1
            or len(zh["lines"]) != 1
            or len(bilingual["lines"]) != 2
        ):
            raise ValueError(f"cue {index}: expected 1/1/2 subtitle lines")
        if bilingual["lines"] != [en["lines"][0], zh["lines"][0]]:
            raise ValueError(f"cue {index}: bilingual text mismatch")

        # A cue is one spoken sentence or clause. Multiple sentence endings
        # indicate that unrelated spoken sentences were combined.
        sentence_endings = len(re.findall(r"[.!?](?:\s|$)", en["lines"][0]))
        if sentence_endings > 1:
            raise ValueError(f"cue {index}: combines multiple spoken sentences")

        en_length = len(en["lines"][0])
        zh_length = len(zh["lines"][0])
        if en_length > 90:
            raise ValueError(
                f"cue {index}: English line exceeds safe text budget ({en_length})"
            )
        if zh_length > 42:
            raise ValueError(
                f"cue {index}: Chinese line exceeds safe text budget ({zh_length})"
            )
        if en_length / cue_duration > 28:
            raise ValueError(f"cue {index}: English reading speed is too high")
        if zh_length / cue_duration > 18:
            raise ValueError(f"cue {index}: Chinese reading speed is too high")
        previous_end = en["end"]
        maximum_duration = max(maximum_duration, cue_duration)
        maximum_en = max(maximum_en, en_length)
        maximum_zh = max(maximum_zh, zh_length)

    report = {
        "film_duration_seconds": FILM_DURATION,
        "cue_count": counts["en"],
        "languages": ["English", "Chinese", "English + Chinese"],
        "timings_identical": True,
        "overlaps": 0,
        "maximum_cue_duration_seconds": round(maximum_duration, 3),
        "maximum_english_characters": maximum_en,
        "maximum_chinese_characters": maximum_zh,
        "caption_safe_margins_pixels": {"left": 220, "right": 220, "bottom": 52},
        "narration_language": "English",
        "chinese_delivery_caption_mode": "English + Chinese",
    }
    output = ROOT / "subtitles" / "validation-report.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("SUBTITLE_VALIDATION_OK")


if __name__ == "__main__":
    main()
