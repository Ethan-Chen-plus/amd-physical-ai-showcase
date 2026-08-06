#!/usr/bin/env python3
"""Generate phrase-aligned SRT files from narration WAV pause boundaries."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "subtitles" / "cues.json"
OUT_DIR = ROOT / "subtitles"


@dataclass(frozen=True)
class Silence:
    start: float
    end: float

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2

    @property
    def duration(self) -> float:
        return self.end - self.start


def media_duration(path: Path) -> float:
    value = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        text=True,
    ).strip()
    return float(value)


def detect_silences(path: Path, threshold_db: int, minimum: float) -> list[Silence]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i",
        str(path),
        "-af",
        f"silencedetect=noise={threshold_db}dB:d={minimum}",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())

    pending: list[float] = []
    silences: list[Silence] = []
    for line in result.stderr.splitlines():
        start_match = re.search(r"silence_start:\s*([0-9.]+)", line)
        if start_match:
            pending.append(float(start_match.group(1)))
        end_match = re.search(r"silence_end:\s*([0-9.]+)", line)
        if end_match and pending:
            silences.append(Silence(pending.pop(0), float(end_match.group(1))))
    return silences


def spoken_weight(text: str) -> float:
    words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text)
    return max(1.0, sum(max(1.0, len(word) / 5.0) for word in words))


def select_boundaries(
    cues: list[dict[str, str]], silences: list[Silence], wav_duration: float
) -> list[Silence]:
    count = len(cues) - 1
    if count == 0:
        return []

    weights = [spoken_weight(cue["en"]) for cue in cues]
    total = sum(weights)
    ideals: list[float] = []
    running = 0.0
    for weight in weights[:-1]:
        running += weight
        ideals.append(wav_duration * running / total)

    # Add interpolated phrase boundaries as deterministic fallbacks. A natural
    # pause wins when it is close to the expected text position; interpolation
    # is used when a spoken phrase has no measurable silence.
    candidates = list(silences)
    for ideal in ideals:
        if all(abs(silence.midpoint - ideal) > 0.14 for silence in candidates):
            candidates.append(Silence(ideal, ideal))
    silences = sorted(candidates, key=lambda silence: silence.midpoint)
    if len(silences) < count:
        raise ValueError(f"need {count} phrase boundaries, found {len(silences)}")

    # Pick ordered pauses near cumulative spoken-text positions. Sentence
    # boundaries in the deterministic TTS render generally have longer pauses,
    # so pause duration receives a small preference.
    states: list[list[tuple[float, list[int]] | None]] = [
        [None] * len(silences) for _ in range(count)
    ]
    for index, silence in enumerate(silences):
        if silence.midpoint < 0.65 or wav_duration - silence.midpoint < 0.65:
            continue
        interpolation_penalty = 0.22 if silence.duration == 0 else 0.0
        cost = (
            (silence.midpoint - ideals[0]) ** 2
            - min(silence.duration, 0.9) * 0.45
            + interpolation_penalty
        )
        states[0][index] = (cost, [index])

    for boundary_index in range(1, count):
        for index, silence in enumerate(silences):
            best: tuple[float, list[int]] | None = None
            for previous in range(index):
                state = states[boundary_index - 1][previous]
                if state is None:
                    continue
                prior = silences[previous]
                if silence.midpoint - prior.midpoint < 0.55:
                    continue
                interpolation_penalty = 0.22 if silence.duration == 0 else 0.0
                cost = (
                    state[0]
                    + (silence.midpoint - ideals[boundary_index]) ** 2
                    - min(silence.duration, 0.9) * 0.45
                    + interpolation_penalty
                )
                if best is None or cost < best[0]:
                    best = (cost, state[1] + [index])
            states[boundary_index][index] = best

    final_states = [state for state in states[-1] if state is not None]
    if not final_states:
        raise ValueError("could not select ordered pause boundaries")
    indices = min(final_states, key=lambda state: state[0])[1]
    return [silences[index] for index in indices]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def srt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(path: Path, cues: list[dict[str, object]]) -> None:
    blocks: list[str] = []
    for index, cue in enumerate(cues, 1):
        blocks.append(
            f'{index}\n{srt_time(float(cue["start"]))} --> '
            f'{srt_time(float(cue["end"]))}\n{cue["en"]}'
        )
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    output: list[dict[str, object]] = []
    report: list[dict[str, object]] = []

    for chapter in manifest["chapters"]:
        chapter_id = chapter["id"]
        chapter_start = float(chapter["start"])
        wav = ROOT / "audio" / "generated" / f"{chapter_id}.wav"
        narration = ROOT / "audio" / "narration" / f"{chapter_id}.txt"
        wav_duration = media_duration(wav)
        silences = detect_silences(
            wav,
            int(manifest["silence_threshold_db"]),
            float(manifest["silence_min_duration"]),
        )

        manual_fields = [
            "start" in cue or "end" in cue for cue in chapter["cues"]
        ]
        if any(manual_fields) and not all(manual_fields):
            raise ValueError(
                f"{chapter_id}: manual timing requires start and end on every cue"
            )
        if all(manual_fields):
            starts = [float(cue["start"]) for cue in chapter["cues"]]
            ends = [float(cue["end"]) for cue in chapter["cues"]]
            for index, (local_start, local_end) in enumerate(
                zip(starts, ends, strict=True), 1
            ):
                if local_start < 0 or local_end > wav_duration + 0.02:
                    raise ValueError(
                        f"{chapter_id}: cue {index} exceeds narration bounds"
                    )
                if local_end <= local_start:
                    raise ValueError(
                        f"{chapter_id}: cue {index} has invalid manual timing"
                    )
                if index > 1 and local_start < ends[index - 2]:
                    raise ValueError(
                        f"{chapter_id}: cue {index} overlaps the previous cue"
                    )
            boundaries: list[Silence] = []
            alignment_mode = "manual"
        else:
            boundaries = select_boundaries(chapter["cues"], silences, wav_duration)
            starts = [0.0] + [silence.end for silence in boundaries]
            ends = [silence.start for silence in boundaries] + [wav_duration]
            alignment_mode = "silence-detected"

        reconstructed = normalize_text(" ".join(cue["en"] for cue in chapter["cues"]))
        source = normalize_text(narration.read_text(encoding="utf-8"))
        if reconstructed != source:
            raise ValueError(f"{chapter_id}: cues do not reconstruct the narration source")

        selected = []
        for cue, local_start, local_end in zip(
            chapter["cues"], starts, ends, strict=True
        ):
            absolute_start = chapter_start + local_start
            absolute_end = chapter_start + local_end
            output.append(
                {
                    "chapter": chapter_id,
                    "start": absolute_start,
                    "end": absolute_end,
                    "en": cue["en"],
                }
            )
            selected.append(
                {
                    "start": round(absolute_start, 3),
                    "end": round(absolute_end, 3),
                    "en": cue["en"],
                }
            )
        report.append(
            {
                "chapter": chapter_id,
                "chapter_start": chapter_start,
                "wav_duration": round(wav_duration, 6),
                "alignment_mode": alignment_mode,
                "detected_pause_count": len(silences),
                "selected_pause_count": len(boundaries),
                "cues": selected,
            }
        )

    write_srt(OUT_DIR / "en.srt", output)
    (OUT_DIR / "alignment-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Generated {len(output)} aligned cues across {len(report)} chapters.")


if __name__ == "__main__":
    main()
