# Hosting the explorer as a website

`xwalk ui --public` turns the local explorer into a site other people can use, and the
repository's `Dockerfile` packages it with **full ontologies already parsed and indexed**
while the image is built. Visitors pick an ontology and map onto it right away. Nothing
is parsed or indexed on their time, and nothing is downloaded on their request.

```bash
docker build -t xwalk-explorer .                     # downloads + parses the ontologies
docker run -p 8080:8080 xwalk-explorer               # http://localhost:8080
docker build --build-arg ONTOLOGIES="hp mondo chebi" -t xwalk-explorer .
```

`ONTOLOGIES` takes catalog ids from `xwalk.ui.library.CATALOG`: `hp`, `mondo`, `doid`,
`go-basic`, `chebi`, `uberon`, `cl`, `pato`, `uo`, `envo`. The default is all of them
except ChEBI, which is large. Each is downloaded from its OBO Foundry permanent URL by
`scripts/build_ontology_library.py`, which also builds its search index. Measured
outside the image on this machine: HPO (19,894 terms) took 3 s and Mondo (58,690 terms)
took 10 s, including the index.

## What public mode changes

| | Local (`xwalk ui`) | Public (`xwalk ui --public`) |
|---|---|---|
| Workspace | the directory you start it in | a private one per visitor (cookie), deleted after `--session-hours` idle (24) |
| Ontologies | built-in samples, your imports, downloads | the same plus `--library` directories, read-only and shared, with shared indexes (`--lookup-cache`) |
| Downloads from a URL | allowed | refused: the server never fetches a URL a visitor names |
| Model keys | the process environment (Settings or `export`) | each visitor's own, kept in that session's memory only; the server's environment is never used for a visitor |
| Model endpoints | any | `https` only, to hosts that resolve to public addresses (no localhost, private ranges or cloud metadata) |
| Runs | unbounded | one per visitor, `--max-tasks` (4) across the site |
| Paths in messages | absolute | hidden |

Visitors can still upload files (`--max-upload-mb`), import their own ontologies into
their workspace, build projects, run them with the offline stand-in, review, evaluate
and download. Sessions live in the server's memory: after a restart, visitors start
again with a new session (and old session directories are deleted). For the same
reason, run **one instance**.

It is still a small demo server (Python's `http.server`), not hardened multi-tenant
infrastructure. Put it behind the platform's HTTPS. Tell visitors not to upload
confidential data; the start page says so.

## Where to host it for free

Verified in October 2026; free tiers change often, so check before relying on one.

| Option | Free resources | Fit |
|---|---|---|
| **Google Cloud Run** | monthly free tier: 180,000 vCPU-seconds, 360,000 GiB-seconds and 2 million requests (request-based billing), or 240,000 vCPU-s / 450,000 GiB-s with instance-based billing; scales to zero | **Recommended.** Runs this Dockerfile as is. Needs a Google Cloud billing account (card), but costs nothing within the free tier |
| **Oracle Cloud Always Free** | an Arm VM (reported as 4 OCPU / 24 GB, reportedly being reduced to 2 OCPU / 12 GB), 200 GB block storage | Most room for big ontologies (ChEBI), always on, but you run a VM (Docker, HTTPS, updates) |
| Render (free web service) | 512 MB RAM, 0.1 CPU, sleeps after 15 min idle, ephemeral disk | Only with small ontologies (`ONTOLOGIES="hp uo pato"`) |
| Hugging Face Spaces | free CPU hardware is 2 vCPU / 16 GB, but creating a Docker Space now needs a paid plan (PRO) | Good if you have PRO; see below |
| Streamlit Community Cloud | about 0.7–2.7 GB RAM, Streamlit apps only | Not used: the explorer would have to be rewritten in Streamlit, and full ontologies need more memory |

### Google Cloud Run (recommended)

From the repository root, with the [gcloud CLI](https://cloud.google.com/sdk/docs/install)
signed in to a project with billing enabled:

```bash
gcloud run deploy xwalk-explorer \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 4Gi --cpu 1 \
  --max-instances 1 \
  --no-cpu-throttling \
  --timeout 3600
```

`--source .` builds the Dockerfile in Cloud Build and deploys it. The URL is printed at
the end.

- `--max-instances 1`: sessions live in one process's memory.
- `--no-cpu-throttling`: runs continue in the background between the page's progress
  polls. With the default request-based billing, CPU is given only while a request is
  being answered, so background runs would crawl. This uses instance-based billing,
  whose free allowance is counted for as long as the instance is up. An idle instance
  is stopped after a while.
- `--memory 4Gi`: Cloud Run's writable disk is held in memory, so visitors' uploads
  and runs count against it. Keep `--max-upload-mb` modest (the image sets 100).

Without `--no-cpu-throttling`, everything that answers within a request still works at
full speed: quick-map candidates, search, browsing, previews and project creation. Only
background runs slow down.

### An Oracle Cloud (or any) VM

```bash
git clone https://github.com/jan3657/xwalk && cd xwalk
docker build --build-arg ONTOLOGIES="hp mondo doid go-basic chebi uberon cl" -t xwalk-explorer .
docker run -d --restart unless-stopped -p 127.0.0.1:8080:8080 xwalk-explorer
```

Put an HTTPS reverse proxy in front: Caddy with `reverse_proxy 127.0.0.1:8080` gets a
certificate automatically. It forwards `X-Forwarded-Proto`, which makes the session
cookie `Secure`.

### Hugging Face Spaces (paid plan)

Create a **Docker** Space and push this repository to it, with this front matter at the
top of the Space's `README.md`:

```yaml
---
title: xwalk explorer
sdk: docker
app_port: 8080
---
```

Spaces show the app inside a frame on huggingface.co. Add
`--frame-ancestors https://huggingface.co` to the command in the Dockerfile so the page
may be framed; its session cookie then becomes `SameSite=None; Secure`.

## Running public mode without Docker

Docker is optional for local testing. From the repository root, install xwalk into a
Python virtual environment, build the same nine full ontologies as the Dockerfile,
then start the server on loopback:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[chatgpt]'
.venv/bin/python scripts/build_ontology_library.py \
  --out .local-explorer/library --cache .local-explorer/cache \
  --only hp mondo doid go-basic uberon cl pato uo envo
mkdir -p .local-explorer/site
.venv/bin/xwalk ui --public --host 127.0.0.1 --port 8080 \
  --root .local-explorer/site --library .local-explorer/library \
  --lookup-cache .local-explorer/cache --chatgpt-login --no-browser
```

Open `http://127.0.0.1:8080`. The library builder keeps existing entries; subsequent
starts reuse parsed terms and indexes. These commands enable BM25 over labels and
synonyms. To add general and biomedical dense retrieval, install `.[chatgpt,dense]`
and add `--dense` to both the library builder and `xwalk ui`. On Apple Silicon, add
`--dense-device mps` to both commands. The builder downloads the two embedding models
and precomputes their indexes; mapping runs reuse the shared library indexes. On
macOS, dense search uses exact NumPy scoring to avoid native OpenMP conflicts between
PyTorch and FAISS.

### Sign in with ChatGPT locally

With `xwalk[chatgpt]` installed, `--chatgpt-login` adds **Continue with ChatGPT** in
Start, Quick map, and Settings. Eligible Plus and Pro users can authorize ChatGPT plan
usage. After connecting, Quick map and new projects select the ChatGPT provider
automatically and show its available models, without API-key or endpoint fields.
Choose **With ChatGPT** / **job endpoint** to run it. Offline runs and retrieval-only
search do not use the subscription. Requests count toward the account's existing plan
limits; **Manage usage** opens ChatGPT's usage settings.

The implementation follows OpenAI's OAuth authorization-code flow with PKCE and verified
ID tokens, then sends streamed, non-stored requests to the public Responses API. It
refreshes credentials and checks the granted plan permission. Credentials never appear
in job files, run manifests, browser storage, or the process environment. In `--public`
mode they are held only in the visitor's server session memory. Without `--public`,
local account credentials persist in owner-only files below
`${XDG_CONFIG_HOME:-~/.config}/xwalk/`, outside the workspace. Sign out attempts remote
session revocation and clears local credentials, retaining the account registration
for a later sign-in.

This option requires **`--host 127.0.0.1`**, and the browser must use that address rather
than `localhost`. The open-source authorization flow returns to an HTTP loopback
callback on the user's computer. It cannot be added to the shared Cloud Run deployment
unchanged: OpenAI currently directs remotely hosted apps to request access separately.
The Dockerfile therefore continues to offer visitor API keys. ChatGPT-backed project
jobs use `llm.kind: chatgpt` and currently require this local explorer for model runs;
they can still run with the offline stand-in.

Sources (checked October 7, 2026): [eligibility and hosted access](https://developers.openai.com/siwc/token-sharing-open-source),
[registration and loopback callbacks](https://developers.openai.com/siwc/token-sharing-open-source/sign-in),
[models and inference](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference),
[preview request restrictions](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations).

For a hosted deployment with API keys:

```bash
python scripts/build_ontology_library.py --out library --cache cache --only hp mondo
xwalk ui --public --host 0.0.0.0 --port 8080 --root site \
  --library library --lookup-cache cache --max-upload-mb 100 --no-browser
```

| Flag | Default | Meaning |
|---|---|---|
| `--public` | — | the mode above |
| `--library DIR` | — | a directory of pre-parsed ontologies (repeatable) |
| `--lookup-cache DIR` | `<root>/cache` | shared search indexes for library-only targets |
| `--session-hours` | `24` | delete a visitor's workspace after this long idle |
| `--max-tasks` | `4` | runs at once across all visitors |
| `--frame-ancestors` | — | origins allowed to show the page in a frame |

`GET /healthz` answers `ok` for platform health checks.

Sources: [Cloud Run pricing](https://cloud.google.com/run/pricing),
[Oracle Always Free resources](https://docs.oracle.com/en-us/iaas/Content/FreeTier/resourceref.htm),
[Render free instances](https://render.com/docs/free),
[Hugging Face Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview),
[Docker SDK marked paid (HF forum)](https://discuss.huggingface.co/t/docker-sdk-now-marked-as-paid-when-creating-a-new-space/177580),
[Streamlit resource limits](https://docs.streamlit.io/knowledge-base/deploy/resource-limits).
