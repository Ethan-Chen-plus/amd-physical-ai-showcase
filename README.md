# AMD Physical AI Showcase

Public evidence website for the Datawhale-EAI AMD Physical AI project.

The site intentionally contains only public-safe assets:

- SHA-verified aggregate results;
- representative success and failure videos;
- public upstream, Notebook, and Hugging Face links;
- clearly separated formal, diagnostic, and in-progress claims.

The private competition adapters, credentials, machine addresses, raw datasets,
and unpublished training code are not included.

The migration evidence page is available at
`migrations.html`; it records the six-layer acceptance gates, DISCOVERSE and
RoboCasa365 evidence, policy boundaries, and the unresolved DexDojo/DexJoCo
identity without exposing private machine paths.

The RoboCasa365 gallery also includes a separate high-resolution, four-view
showcase rollout. It is presentation evidence only; the formal 16-task x
50-episode score remains unchanged.

The DISCOVERSE section includes four isolated official MMK2 example replays on
AMD Radeon: box pick, drawer open, cabinet door open, and cup to plate. Each task has three native 1920x1080
camera videos (`cam_0`, `cam_1`, `cam_2`), plus a web-friendly three-view
composite and a SHA-256 result manifest. The composite is presentation-only;
it does not alter training data or formal evaluation denominators.

## Local preview

```bash
python3 -m http.server 4173 --directory .
```

Then open <http://127.0.0.1:4173/>.
