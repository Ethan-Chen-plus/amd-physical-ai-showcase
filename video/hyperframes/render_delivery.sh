#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE="$(cd "$ROOT/../.." && pwd)"
DIST="$ROOT/dist"
MASTER="$DIST/amd-physical-ai-demo-en.mp4"
AUDIO="$ROOT/audio/generated/amd-physical-ai-demo-audio.wav"

mkdir -p "$DIST"
cd "$ROOT"

export HYPERFRAMES_RUN_ID="${HYPERFRAMES_RUN_ID:-amd-demo-film-20260803}"
export TMPDIR="${HF_TMPDIR:-/tmp/hyperframes-amd-demo-20260803}"
mkdir -p "$TMPDIR"

# Prefer an installed Chromium binary so rendering does not trigger a browser download.
if [[ -z "${HYPERFRAMES_BROWSER_PATH:-}" ]]; then
  for candidate in \
    /snap/chromium/current/usr/lib/chromium-browser/chrome \
    /usr/bin/google-chrome \
    /usr/bin/chromium; do
    if [[ -x "$candidate" ]]; then
      export HYPERFRAMES_BROWSER_PATH="$candidate"
      break
    fi
  done
fi

if [[ "${SKIP_AUDIO_RENDER:-0}" != "1" ]]; then
  HYPERFRAMES_PYTHON="${HYPERFRAMES_PYTHON:-/data/Data14TB/envs/hyperframes-audio/bin/python}" \
    "$ROOT/audio/render_audio.sh"
else
  test -s "$AUDIO"
fi

npx hyperframes lint
npx hyperframes check --samples 13 --at-transitions --no-browser-gpu --timeout 15000
if [[ "${SKIP_MASTER_RENDER:-0}" != "1" ]]; then
  npx hyperframes render --output "$MASTER" --fps 30 --quality standard \
    --workers "${HYPERFRAMES_WORKERS:-2}" --no-browser-gpu \
    --protocol-timeout "${HYPERFRAMES_PROTOCOL_TIMEOUT:-600000}" \
    --browser-timeout "${HYPERFRAMES_BROWSER_TIMEOUT:-300}" \
    --player-ready-timeout "${HYPERFRAMES_PLAYER_READY_TIMEOUT:-120000}"
else
  test -s "$MASTER"
fi

burn() {
  local input="$1"; local subs="$2"; local output="$3"
  ffmpeg -hide_banner -loglevel error -y -i "$input" \
    -vf "subtitles=$(printf '%s' "$subs" | sed "s/'/'\\\\''/g"):force_style='FontName=DejaVu Sans,FontSize=14,Outline=1,Shadow=0,MarginV=16,Alignment=2,BorderStyle=1,WrapStyle=2'" \
    -map 0:v:0 -map 0:a:0 -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
    -c:a aac -b:a 192k -t 299 -movflags +faststart "$output"
}

burn "$MASTER" "$ROOT/subtitles/en.srt" "$DIST/amd-physical-ai-demo-en-subtitled.mp4"
burn "$MASTER" "$ROOT/subtitles/zh.srt" "$DIST/amd-physical-ai-demo-zh.mp4"
burn "$MASTER" "$ROOT/subtitles/bilingual.srt" "$DIST/amd-physical-ai-demo-bilingual.mp4"

cp "$DIST/amd-physical-ai-demo-en-subtitled.mp4" "$SITE/assets/videos/amd-physical-ai-demo-en.mp4"
cp "$DIST/amd-physical-ai-demo-zh.mp4" "$SITE/assets/videos/amd-physical-ai-demo-zh.mp4"
cp "$DIST/amd-physical-ai-demo-bilingual.mp4" "$SITE/assets/videos/amd-physical-ai-demo-bilingual.mp4"
cp "$ROOT/subtitles/en.srt" "$SITE/assets/videos/amd-physical-ai-demo.en.srt"
cp "$ROOT/subtitles/zh.srt" "$SITE/assets/videos/amd-physical-ai-demo.zh.srt"
cp "$ROOT/subtitles/bilingual.srt" "$SITE/assets/videos/amd-physical-ai-demo.zh-en.srt"

for output in "$DIST"/amd-physical-ai-demo-*.mp4; do
  ffprobe -v error -show_entries format=duration,size -show_entries stream=width,height,codec_name,r_frame_rate \
    -of default=noprint_wrappers=1 "$output"
done

printf 'Delivered to %s/assets/videos\n' "$SITE"
