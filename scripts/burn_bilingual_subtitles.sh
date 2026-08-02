#!/usr/bin/env bash
set -euo pipefail

# Burn an English/Chinese SRT track into a web-safe MP4.

if [[ $# -ne 3 ]]; then
  printf 'usage: %s INPUT.mp4 BILINGUAL.srt OUTPUT.mp4\n' "$0" >&2
  exit 2
fi

INPUT="$1"
SUBS="$2"
OUTPUT="$3"
[[ -f "$INPUT" ]] || { printf 'missing input: %s\n' "$INPUT" >&2; exit 2; }
[[ -f "$SUBS" ]] || { printf 'missing subtitles: %s\n' "$SUBS" >&2; exit 2; }

mkdir -p "$(dirname "$OUTPUT")"
ffmpeg -hide_banner -loglevel error -y -i "$INPUT" \
  -vf "subtitles=$(printf '%s' "$SUBS" | sed "s/'/'\\\\''/g"):force_style='FontName=Noto Sans CJK SC,FontSize=22,Outline=2,Shadow=1,MarginV=34,Alignment=2'" \
  -an -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart "$OUTPUT"
ffprobe -v error -show_entries format=duration,size -show_entries stream=width,height,codec_name \
  -of default=noprint_wrappers=1 "$OUTPUT"
printf 'Wrote %s\n' "$OUTPUT"
