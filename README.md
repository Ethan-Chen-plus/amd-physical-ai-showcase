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

## Local preview

```bash
python3 -m http.server 4173 --directory .
```

Then open <http://127.0.0.1:4173/>.
