#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$ROOT/generated"
VOICE="${HYPERFRAMES_TTS_VOICE:-af_heart}"
SPEED="${HYPERFRAMES_TTS_SPEED:-0.96}"
PYTHON="${HYPERFRAMES_PYTHON:-/data/Data14TB/envs/hyperframes-audio/bin/python}"
SAMPLE_RATE=48000
DURATION=260

mkdir -p "$OUT"

starts=(0 12 28 48 68 92 110 134 158 174 194 218 246)
texts=("$ROOT"/narration/*.txt)
voice_inputs=()
voice_filters=()
voice_labels=()

for i in "${!texts[@]}"; do
  text_file="${texts[$i]}"
  stem="$(basename "$text_file" .txt)"
  wav="$OUT/$stem.wav"
  if [[ ! -s "$wav" ]]; then
    HYPERFRAMES_PYTHON="$PYTHON" npx hyperframes tts "$text_file" \
      --voice "$VOICE" --speed "$SPEED" --lang en-us --output "$wav"
  fi
  delay_ms=$(( starts[$i] * 1000 ))
  voice_inputs+=( -i "$wav" )
  voice_filters+=( "[$i:a]adelay=${delay_ms}|${delay_ms},volume=1.0[v$i]" )
  voice_labels+=( "[v$i]" )
done

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
