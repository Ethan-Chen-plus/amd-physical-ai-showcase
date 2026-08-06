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
    cues = parse(ROOT / "subtitles" / "en.srt")

    maximum_duration = 0.0
    maximum_en = 0
    previous_end = -1.0
    for index, cue in enumerate(cues, 1):
        start = float(cue["start"])
        end = float(cue["end"])
        if start < previous_end:
            raise ValueError(f"cue {index}: overlaps previous cue")
        if start < 0 or end > FILM_DURATION or end <= start:
            raise ValueError(f"cue {index}: invalid film bounds")
        cue_duration = end - start
        if cue_duration < 0.75 or cue_duration > 9.0:
            raise ValueError(
                f"cue {index}: duration {cue_duration:.3f}s outside 0.75-9.0s"
            )
        if len(cue["lines"]) != 1:
            raise ValueError(f"cue {index}: expected one subtitle line")

        # A cue is one spoken sentence or clause. Multiple sentence endings
        # indicate that unrelated spoken sentences were combined.
        sentence_endings = len(re.findall(r"[.!?](?:\s|$)", cue["lines"][0]))
        if sentence_endings > 1:
            raise ValueError(f"cue {index}: combines multiple spoken sentences")

        en_length = len(cue["lines"][0])
        if en_length > 90:
            raise ValueError(
                f"cue {index}: English line exceeds safe text budget ({en_length})"
            )
        if en_length / cue_duration > 28:
            raise ValueError(f"cue {index}: English reading speed is too high")
        previous_end = end
        maximum_duration = max(maximum_duration, cue_duration)
        maximum_en = max(maximum_en, en_length)

    report = {
        "film_duration_seconds": FILM_DURATION,
        "cue_count": len(cues),
        "language": "English",
        "overlaps": 0,
        "maximum_cue_duration_seconds": round(maximum_duration, 3),
        "maximum_english_characters": maximum_en,
        "caption_safe_margins_pixels": {"left": 220, "right": 220, "bottom": 52},
        "narration_language": "English",
        "caption_language": "English",
    }
    output = ROOT / "subtitles" / "validation-report.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("SUBTITLE_VALIDATION_OK")


if __name__ == "__main__":
    main()
