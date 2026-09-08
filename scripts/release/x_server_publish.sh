#!/usr/bin/env bash
# Installed entrypoint: /home/ccl/Desktop/forPYPI/automation/x_server_publish.sh
set -euo pipefail
umask 077
export GIT_TERMINAL_PROMPT=0 GCM_INTERACTIVE=Never
export GIT_ASKPASS=/usr/bin/false SSH_ASKPASS=/usr/bin/false
ROOT=/home/ccl/Desktop/forPYPI
REPO="$ROOT/FedOps_github/FedOps"
STATE="$ROOT/automation"
PYTHON="$ROOT/release-test-env/bin/python"
TWINE=/home/ccl/anaconda3/envs/pypi_fedops/bin/python
export PATH="$ROOT/release-test-env/bin:/usr/local/bin:/usr/bin:/bin"
mkdir -p "$STATE"
exec 9>"$STATE/publish.lock"
flock -n 9 || exit 0
git -C "$REPO" fetch origin main
revision=$(git -C "$REPO" rev-parse origin/main)
if [[ -f "$STATE/last-successful-revision" ]] && [[ $(<"$STATE/last-successful-revision") == "$revision" ]]; then
  exit 0
fi

# Public API: no credential extraction. Require this exact main push's CI.
"$PYTHON" - "$revision" <<'PY'
import json, sys
from urllib.request import Request, urlopen
revision = sys.argv[1]
url = ('https://api.github.com/repos/gachon-CCLab/FedOps/actions/workflows/'
       f'pypi-release.yml/runs?head_sha={revision}&event=push&per_page=5')
with urlopen(Request(url, headers={'Accept': 'application/vnd.github+json',
                                 'User-Agent': 'FedOps-X-PyPI-Publisher'}), timeout=30) as response:
    runs = json.load(response)['workflow_runs']
matching = [run for run in runs if run['head_sha'] == revision and run['head_branch'] == 'main'
            and run['event'] == 'push' and run['head_repository']['full_name'] == 'gachon-CCLab/FedOps']
if not matching or matching[0]['status'] != 'completed' or matching[0]['conclusion'] != 'success':
    raise SystemExit('Waiting for successful exact-revision main CI; nothing published')
print(f'CI approved revision {revision}')
PY

# Isolated worktree: never reset the operator's existing checkout or caches.
checkout=$(mktemp -d "$STATE/checkout.XXXXXX")
git -C "$REPO" worktree add --detach "$checkout" "$revision"
cleanup() { git -C "$REPO" worktree remove --force "$checkout"; }
trap cleanup EXIT
cd "$checkout"
"$PYTHON" -m unittest discover -s scripts/release -p 'test_*.py'
(cd src/python && PYTHONPATH=. "$PYTHON" -m unittest discover -s tests)
"$PYTHON" scripts/release/pypi_release.py stamp
"$PYTHON" -m build src/python --outdir dist
"$TWINE" -m twine check dist/*
plan=$("$PYTHON" scripts/release/pypi_release.py plan)
version=$(sed -n 's/^version=//p' <<< "$plan")
publish=$(sed -n 's/^publish=//p' <<< "$plan")
[[ "$version" =~ ^[0-9]+(\.[0-9]+)+$ ]] || { echo 'Invalid release version'; exit 1; }
artifact_dir="$ROOT/releases/$version/$revision"
mkdir -p "$artifact_dir"
cp dist/* "$artifact_dir/"
sha256sum dist/*
if [[ "$publish" == true ]]; then
  # Existing X-server config is consumed by Twine, never read or copied here.
  "$TWINE" -m twine upload --non-interactive --disable-progress-bar \
    --config-file /home/ccl/.pypirc --repository pypi dist/*
  EXACT_ARTIFACTS=true "$PYTHON" scripts/release/pypi_release.py verify
else
  "$PYTHON" scripts/release/pypi_release.py verify
fi
"$PYTHON" -m venv "$checkout/pypi-smoke"
"$checkout/pypi-smoke/bin/python" -m pip install --no-deps --no-cache-dir \
  --index-url https://pypi.org/simple "fedops==$version"
"$checkout/pypi-smoke/bin/fedops" --help
"$checkout/pypi-smoke/bin/fedops" run agent-studio --help
printf '%s\n' "$revision" > "$STATE/last-successful-revision.tmp"
mv "$STATE/last-successful-revision.tmp" "$STATE/last-successful-revision"
printf 'VERIFIED fedops=%s revision=%s\n' "$version" "$revision"
