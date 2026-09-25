# Both pinned OCI indexes contain linux/amd64 and linux/arm64 manifests.
FROM dhi.io/python@sha256:dd14beb38cfee3decf279bf0e7e438aaac23f00a32c7cd6f2e999e19c29b2d0c AS build

WORKDIR /build
COPY docker/requirements.lock /build/requirements.lock
# Bootstrap build tooling from the same hashed lock before installing sdists.
# Disabling build isolation prevents unpinned build-time dependency downloads.
RUN python -m venv /opt/venv && python - <<'PY'
import pathlib
import re
import subprocess

lock = pathlib.Path('/build/requirements.lock').read_text()
blocks = re.split(r'(?=^[a-zA-Z][a-zA-Z0-9_.-]*==)', lock, flags=re.MULTILINE)
bootstrap = ''.join(block for block in blocks if block.startswith(('pip==', 'setuptools==', 'wheel==', 'packaging==')))
pathlib.Path('/build/bootstrap.lock').write_text(bootstrap)
subprocess.run(['/opt/venv/bin/python', '-m', 'pip', 'install', '--quiet', '--no-cache-dir', '--require-hashes', '-r', '/build/bootstrap.lock'], check=True)
PY
RUN /opt/venv/bin/python -m pip install --quiet --no-cache-dir --require-hashes --no-build-isolation -r /build/requirements.lock

COPY pyproject.toml README.md LICENSE.md /build/
COPY licenses/code_eval-APACHE-2.0.txt /build/licenses/code_eval-APACHE-2.0.txt
COPY lm_eval /build/lm_eval
COPY benchmark_runner /build/benchmark_runner
RUN /opt/venv/bin/python -m pip wheel --quiet --no-deps --no-build-isolation --wheel-dir /wheels /build && \
    /opt/venv/bin/python -m pip install --quiet --no-index --no-deps /wheels/lm_eval-*.whl && \
    /opt/venv/bin/python -m pip check

# IFEval needs punkt_tab. Fetch at build time only, pinning both source and bytes.
RUN python - <<'PY'
import hashlib
import io
import urllib.request
import zipfile

url = 'https://raw.githubusercontent.com/nltk/nltk_data/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/packages/tokenizers/punkt_tab.zip'
with urllib.request.urlopen(url, timeout=60) as response:
    data = response.read()
assert hashlib.sha256(data).hexdigest() == 'e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106'
with zipfile.ZipFile(io.BytesIO(data)) as archive:
    archive.extractall('/opt/nltk_data/tokenizers')
PY

# Hardened runtime has neither a shell nor an OS package manager.
FROM dhi.io/python@sha256:ff47bcf5c9a1918cf4ad19368202410a2189c69d902c0637ff4277e1ca3009e4
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/nltk_data /opt/nltk_data
COPY --from=build /build/requirements.lock /opt/lm-eval/requirements.lock
ENV PATH="/opt/venv/bin:/usr/bin" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/opt/nltk_data \
    HF_HOME=/tmp/huggingface
USER 65532:65532
WORKDIR /tmp
CMD ["python", "-m", "lm_eval", "--help"]
