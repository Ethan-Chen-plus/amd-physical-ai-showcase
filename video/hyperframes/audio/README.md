# Film Audio

This directory contains the reproducible audio source for the AMD Physical AI showcase film.

`narration/*.txt` holds the English chapter script. `render_audio.sh` renders each chapter with the Kokoro `af_heart` English voice, places the chapters on the 299-second timeline, rejects any voice gap longer than three seconds, creates a restrained original score with FFmpeg, and mixes the two tracks into the WAV consumed by `index.html`.

The voice model and generated WAV files stay outside Git. On the authoring machine, the audio environment is kept under `/data/Data14TB/envs/hyperframes-audio`; a fresh environment can install `kokoro-onnx==0.5.0` and `soundfile`, then run the script after the HyperFrames TTS model cache is available.

```bash
HYPERFRAMES_PYTHON=/data/Data14TB/envs/hyperframes-audio/bin/python \
  ./audio/render_audio.sh
```

The delivery script invokes this step automatically unless `SKIP_AUDIO_RENDER=1` is set.
