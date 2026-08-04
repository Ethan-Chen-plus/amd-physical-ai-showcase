#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$ROOT/generated"
VOICE="${HYPERFRAMES_TTS_VOICE:-af_heart}"
SPEED="${HYPERFRAMES_TTS_SPEED:-0.96}"
PYTHON="${HYPERFRAMES_PYTHON:-/data/Data14TB/envs/hyperframes-audio/bin/python}"
SAMPLE_RATE=48000
DURATION=299

mkdir -p "$OUT"

# Starts are derived from the rendered chapter durations with a 0.5s breathing
# gap. Keep this schedule synchronized with the HTML composition and subtitles.
starts_ms=(0 17990 36960 59090 81730 107150 131740 157290 182280 203690 229680 253370 279130)
texts=("$ROOT"/narration/*.txt)
voice_inputs=()
voice_filters=()
voice_labels=()

for i in "${!texts[@]}"; do
  text_file="${texts[$i]}"
  stem="$(basename "$text_file" .txt)"
  wav="$OUT/$stem.wav"
  if [[ ! -s "$wav" || "$text_file" -nt "$wav" || "${FORCE_TTS_RENDER:-0}" == "1" ]]; then
    HYPERFRAMES_PYTHON="$PYTHON" npx hyperframes tts "$text_file" \
      --voice "$VOICE" --speed "$SPEED" --lang en-us --output "$wav"
  fi
  delay_ms="${starts_ms[$i]}"
  voice_inputs+=( -i "$wav" )
  voice_filters+=( "[$i:a]adelay=${delay_ms}|${delay_ms},volume=1.0[v$i]" )
  voice_labels+=( "[v$i]" )
done

python3 - "$ROOT" <<'PY'
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
starts = [value / 1000 for value in [0, 17990, 36960, 59090, 81730, 107150, 131740, 157290, 182280, 203690, 229680, 253370, 279130]]
names = [
    "01-hook", "02-claim", "03-robocasa", "04-long-horizon", "05-dexjoco",
    "06-protocol", "07-architecture", "08-discoverse", "09-rendering",
    "10-amd", "11-result", "12-failure", "13-close",
]
durations = []
for name in names:
    path = root / "generated" / f"{name}.wav"
    value = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        text=True,
    ).strip()
    durations.append(float(value))

errors = []
for index, (start, duration) in enumerate(zip(starts, durations)):
    end = start + duration
    if index + 1 < len(starts):
        gap = starts[index + 1] - end
        if gap < -0.25:
            errors.append(f"{names[index]} overlaps next chapter by {-gap:.2f}s")
        elif gap > 3.0:
            errors.append(f"{names[index]} voice gap is {gap:.2f}s")
    else:
        gap = 299.0 - end
        if gap > 3.0:
            errors.append(f"{names[index]} voice gap before film end is {gap:.2f}s")
    print(f"{names[index]:18s} start={start:6.2f}s duration={duration:6.2f}s end={end:6.2f}s")

if errors:
    raise SystemExit("Narration timing gate failed:\n- " + "\n- ".join(errors))
print("Narration timing gate passed: every voice gap is <= 3 seconds.")
PY

voiceover="$OUT/narration-${VOICE}.wav"
filter="$(IFS=';'; echo "${voice_filters[*]}");$(IFS=''; echo "${voice_labels[*]}")amix=inputs=${#texts[@]}:duration=longest:dropout_transition=0:normalize=0,alimiter=limit=0.92[out]"
ffmpeg -hide_banner -loglevel error -y "${voice_inputs[@]}" \
  -filter_complex "$filter" -map '[out]' -ar "$SAMPLE_RATE" -ac 2 -c:a pcm_s16le "$voiceover"

music="$OUT/music-bed.wav"
ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "aevalsrc=0.028*sin(2*PI*55*t)*(0.72+0.28*sin(2*PI*t/24))+0.014*sin(2*PI*110*t)+0.009*sin(2*PI*220*t)+0.006*sin(2*PI*277.18*t)+0.004*sin(2*PI*329.63*t):s=${SAMPLE_RATE}:d=${DURATION}" \
  -af "lowpass=f=1400,afade=t=in:st=0:d=8,afade=t=out:st=252:d=8,volume=0.72" \
  -ar "$SAMPLE_RATE" -ac 2 -c:a pcm_s16le "$music"

mix="$OUT/amd-physical-ai-demo-audio.wav"
ffmpeg -hide_banner -loglevel error -y -i "$voiceover" -i "$music" \
  -filter_complex "[0:a]volume=1.0[voice];[1:a]volume=0.58[music];[voice][music]amix=inputs=2:duration=longest:dropout_transition=2,alimiter=limit=0.95[out]" \
  -map '[out]' -t "$DURATION" -ar "$SAMPLE_RATE" -ac 2 -c:a pcm_s16le "$mix"

ffprobe -v error -show_entries format=duration,size -of default=noprint_wrappers=1 "$mix"
printf 'Audio written to %s\n' "$mix"
