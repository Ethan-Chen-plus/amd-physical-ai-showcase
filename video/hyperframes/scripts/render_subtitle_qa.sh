#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/dist"
MASTER="$DIST/amd-physical-ai-demo-master.mp4"
LEGACY_MASTER="$DIST/amd-physical-ai-demo-en.mp4"
SOURCE="$MASTER"

if [[ ! -s "$SOURCE" ]]; then
  SOURCE="$LEGACY_MASTER"
fi
test -s "$SOURCE"

mkdir -p "$DIST"
STYLE="FontName=DejaVu Sans,FontSize=7,PrimaryColour=&H00FFFFFF,OutlineColour=&H00101010,Outline=1,Shadow=0,MarginL=44,MarginR=44,MarginV=14,Alignment=2,BorderStyle=1,WrapStyle=0"
QA="$DIST/qa-bilingual-first-10s.mp4"

ffmpeg -hide_banner -loglevel error -y \
  -i "$SOURCE" -i "$ROOT/audio/generated/amd-physical-ai-demo-audio.wav" -t 10 \
  -vf "subtitles=$ROOT/subtitles/bilingual.srt:charenc=UTF-8:force_style='$STYLE'" \
  -map 0:v:0 -map 1:a:0 -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \
  -c:a aac -b:a 192k -movflags +faststart "$QA"

ffmpeg -hide_banner -loglevel error -y -ss 1.5 -i "$QA" \
  -frames:v 1 "$DIST/qa-bilingual-frame-01.png"
ffmpeg -hide_banner -loglevel error -y -ss 5.0 -i "$QA" \
  -frames:v 1 "$DIST/qa-bilingual-frame-02.png"

ffprobe -v error -show_entries format=duration,size \
  -show_entries stream=width,height,codec_name,r_frame_rate \
  -of default=noprint_wrappers=1 "$QA"
