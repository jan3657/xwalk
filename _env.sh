#!/usr/bin/env bash
# _env.sh — shared environment for all cluster jobs. `source` this.
# Edit the CLUSTER-SPECIFIC exports below for your site, then never again.
set -euo pipefail

# ---- CLUSTER-SPECIFIC (edit these) -------------------------------------------
export SCRATCH="/ceph/grid/home/jd3099/projects/xwalk/data"
export PROJECT_DIR="/ceph/grid/home/jd3099/projects/xwalk"

# The project venv. xwalk requires Python >= 3.10 (Tantivy's floor), so this is a
# project-local 3.11 venv rather than /ceph/grid/home/jd3099/venvs/xwalk, which is
# 3.9 and cannot import the package. `.venv/` is gitignored.
#   uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
#
# Set unconditionally, NOT `${VENV:-...}`: a stale VENV is commonly already exported
# in the ambient environment, and defaulting around it silently reactivates the 3.9
# venv. Use XWALK_VENV to point somewhere else on purpose.
export VENV="${XWALK_VENV:-$PROJECT_DIR/.venv}"

# module load cuda/13.2 2>/dev/null || true   # uncomment / adjust for your site

# ---- Network & Proxy Setup ---------------------------------------------------
export http_proxy=http://www-proxy.ijs.si:8080
export https_proxy=http://www-proxy.ijs.si:8080
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export all_proxy="$http_proxy"
export ALL_PROXY="$http_proxy"
export NO_PROXY="127.0.0.1,localhost,$(hostname -s),$(hostname -f)"
export no_proxy="$NO_PROXY"

# ---- Hugging Face large-file download safety --------------------------------
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

# ---- Performance & Hardware Optimization ------------------------------------
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=64
export NCCL_P2P_DISABLE=1

# ---- Cache & Storage Safety -------------------------------------------------
# TMPDIR must NOT sit on Ceph. Several tests build real Tantivy indexes under
# pytest's tmp_path, which follows TMPDIR; on network storage the retrieval suite
# takes minutes instead of well under a second. Prefer tmpfs, then local disk,
# and only fall back to the Ceph cache if neither is writable.
_pick_tmpdir() {
    local candidate
    for candidate in "/dev/shm/xwalk-${USER:-$(whoami)}" "/tmp/xwalk-${USER:-$(whoami)}"; do
        if mkdir -p "$candidate" 2>/dev/null && [ -w "$candidate" ]; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    printf '%s' "/ceph/grid/home/${USER:-$(whoami)}/.cache/tmp"
}
export TMPDIR="${TMPDIR_OVERRIDE:-$(_pick_tmpdir)}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
mkdir -p "$TMPDIR"

cd "$PROJECT_DIR"

# ---- Local secrets -----------------------------------------------------------
# pytest does not read .env on its own and xwalk has no dotenv dependency, so the
# live-provider test would skip forever without this. `set -a` exports every
# assignment; the file is gitignored.
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$PROJECT_DIR/.env"
    set +a
fi

# Activate the project venv, with a readable error rather than a cryptic one.
if [ ! -x "$VENV/bin/python" ]; then
    echo "[_env] ERROR: no venv at $VENV" >&2
    echo "[_env]   uv venv --python 3.11 .venv" >&2
    echo "[_env]   uv pip install --python .venv/bin/python -e \".[dev]\"" >&2
    set +euo pipefail
    return 1 2>/dev/null || exit 1
fi
source "$VENV/bin/activate"

_pyver=$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
case "$_pyver" in
    3.9|3.8|3.7|2.*) echo "[_env] ERROR: Python $_pyver too old; xwalk needs >= 3.10" >&2 ;;
esac

# Offline: model/dataset cache setup. Phase 1 uses no Hugging Face models — these
# matter from Phase 3 (dense retrieval) onwards.
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export XWALK_DATA_ROOT="${XWALK_DATA_ROOT:-$SCRATCH}"
export PYTHONUNBUFFERED=1

if [ -n "${XWALK_TEST_API_KEY:-}" ] && [ -n "${XWALK_TEST_BASE_URL:-}" ] \
   && [ -n "${XWALK_TEST_MODEL:-}" ]; then
    _provider="$XWALK_TEST_MODEL @ $XWALK_TEST_BASE_URL"
else
    _provider="unset (integration test will skip)"
fi

echo "[_env] PROJECT_DIR=$PROJECT_DIR  python=$_pyver  TMPDIR=$TMPDIR"
echo "[_env] live provider: $_provider"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

# Restore normal shell behavior for interactive terminals (prevent terminal exit on
# command failure)
set +euo pipefail
