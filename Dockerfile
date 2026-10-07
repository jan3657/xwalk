# The xwalk explorer as a public website: `xwalk ui --public` with full ontologies parsed
# and indexed while the image is built. See docs/guide/hosting.md.
#
#   docker build -t xwalk-explorer .
#   docker run -p 8080:8080 xwalk-explorer            # then open http://localhost:8080
#
# Choose the ontologies with a build argument (catalog ids from xwalk.ui.library.CATALOG;
# ChEBI is large and left out by default):
#   docker build --build-arg ONTOLOGIES="hp mondo go-basic chebi" -t xwalk-explorer .
FROM python:3.12-slim

ARG ONTOLOGIES="hp mondo doid go-basic uberon cl pato uo envo"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /opt/xwalk/src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts/build_ontology_library.py ./scripts/
RUN pip install --no-cache-dir . \
    && useradd --create-home --uid 1000 xwalk \
    && mkdir -p /opt/xwalk/library /opt/xwalk/cache /data \
    && python scripts/build_ontology_library.py \
        --out /opt/xwalk/library --cache /opt/xwalk/cache --keep-going --only ${ONTOLOGIES} \
    && chown -R xwalk /opt/xwalk/cache /data

USER xwalk
EXPOSE 8080
HEALTHCHECK CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz')"
CMD ["sh", "-c", "exec xwalk ui --public --host 0.0.0.0 --port \"$PORT\" --root /data --library /opt/xwalk/library --lookup-cache /opt/xwalk/cache --max-upload-mb 100 --max-calls-cap 200 --no-browser"]
