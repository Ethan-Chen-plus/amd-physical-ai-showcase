#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE="$(cd "$ROOT/../.." && pwd)"
DIST="$ROOT/dist"
MASTER="$DIST/amd-physical-ai-demo-en.mp4"

mkdir -p "$DIST"
cd "$ROOT"

export HYPERFRAMES_RUN_ID="${HYPERFRAMES_RUN_ID:-amd-demo-film-20260803}"
export TMPDIR="${HF_TMPDIR:-/tmp/hyperframes-amd-demo-20260803}"
mkdir -p "$TMPDIR"

npx hyperframes lint
npx hyperframes check --samples 13 --at-transitions --no-browser-gpu --timeout 15000
if [[ "${SKIP_MASTER_RENDER:-0}" != "1" ]]; then
  npx hyperframes render --output "$MASTER" --fps 30 --quality standard
else
  test -s "$MASTER"
fi

burn() {
  local input="$1"; local subs="$2"; local output="$3"
  ffmpeg -hide_banner -loglevel error -y -i "$input" \
    -vf "subtitles=$(printf '%s' "$subs" | sed "s/'/'\\\\''/g"):force_style='FontName=Noto Sans CJK SC,FontSize=16,Outline=1,Shadow=0,MarginV=24,Alignment=2,BorderStyle=1,WrapStyle=2'" \
    -an -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart "$output"
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
