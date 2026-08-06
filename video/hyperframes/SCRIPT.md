# AMD Physical AI Demo Film

Delivery target: 4 minutes 59 seconds, 1920x1080, 30 fps.

The deterministic master uses real experiment footage, on-screen English editorial copy, and a generated editorial blueprint visual for the opening thesis card. The competition delivery adds phrase-aligned English captions from a UTF-8 SRT track. No generated robot footage is used.

## Pacing contract

- Narration drives the edit. Each chapter is scheduled from its rendered voice duration.
- The breathing gap after a sentence is 0.5 seconds in the master and never exceeds 3 seconds by the timing gate.
- A successful or especially informative task may keep its full trajectory. An incomplete or low-value attempt is summarized in voice and cut to the next evidence clip.
- The final boundary-case chapter receives the same narration as the success chapters.

## Audio Delivery

- Voice: Kokoro-82M `af_heart` English voice, rendered one chapter at a time at speed `0.96`.
- Score: deterministic FFmpeg-generated low-frequency pad with fade-in and fade-out; no external copyrighted music is bundled.
- Source: `audio/render_audio.sh` and `audio/narration/*.txt`.
- Final delivery: HyperFrames embeds the mixed WAV into the master, then FFmpeg adds English captions to the competition MP4.
- Reproduction: run `HYPERFRAMES_PYTHON=/data/Data14TB/envs/hyperframes-audio/bin/python ./audio/render_audio.sh` before rendering. The downloaded model cache is external to the repository and generated WAVs are ignored.

## Narration copy

1. Datawhale-EAI builds a reproducible AMD Physical AI stack across dexterous hands, household manipulation, simulation, and rendering.
2. One evidence contract follows data, policy, physics, and proof across the complete project.
3. RoboCasa365 gives the project a household manipulation benchmark with fixed tasks, seeds, videos, and a mobile observation contract.
4. Long-horizon success traces make the full action sequence inspectable.
5. DexJoCo brings contact-rich dexterous tasks into the same protocol with native JAX on AMD ROCm.
6. Each result carries its own protocol and denominator: official seeds set the benchmark, recovery searches answer a separate engineering question, and diagnostics explain a specific behavior.
7. Data becomes policy, policy meets physics, and every result becomes proof through a shared artifact contract.
8. DISCOVERSE preserves expert paths, policy experiments, multi-view rendering, and MP4 output after migration.
9. 3D Gaussian Splatting and dynamic replay turn the renderer itself into visible migration evidence.
10. ROCm versions, GPU identity, memory, throughput, training time, JSON, MP4, and SHA travel with every result.
11. Success leads the story, with every source artifact attached behind it.
12. PAC-MAN predictive control tracks a projectile around Unitree G1, predicts its future path, and redirects motion to preserve clearance on AMD. A compact boundary sequence then maps navigation, contact, release, and recovery into the next engineering targets.
13. Build it, run it, and show the proof through code, notebooks, weights, reports, videos, and reproducible commands.

## Editorial rule

The film places verified success footage first, DexJoCo early, migration engineering in the middle, and PAC-MAN predictive safety control before the compact boundary chapter. Every clip stays attached to its protocol and evidence record.
