# xwalk quickstart

A five-term target vocabulary (`targets.csv`) and four mentions (`sources.csv`), one of
which has no correct target. Created by `xwalk init`.

```
xwalk validate --job job.yaml                      # strict, offline, zero model calls
xwalk match --job job.yaml --out run --max-calls 50
xwalk inspect --run run
xwalk explain --run run s2
xwalk export --run run --view reviewed --out reviewed.csv
```

`match` needs the endpoint in `job.yaml` (the OpenAI API by default, key read from
`OPENAI_API_KEY`) and its calls are billed by the provider; `--max-calls` caps them. To
run without any endpoint, validate with `xwalk validate --job job.yaml --no-credentials`
and run the job from Python with a scripted model, as in the quickstart of the xwalk
README (https://github.com/jan3657/xwalk#readme).
