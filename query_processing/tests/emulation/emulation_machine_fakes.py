"""Stand-ins on ``PATH`` for running ``emulation/machine/`` and ``common.sh`` locally.

A fake ``curl`` answers the metadata server's token URL, treats a folder as the
cache bucket (Cloud Storage's XML API: GET gives the file or a 404, ``-T`` stores
it; the JSON API lists it), serves the files of ``downloads/`` at
``https://downloads.test/`` and plays a small Elasticsearch whose snapshots are
files in another folder. It records every call's argv, and the Elasticsearch calls
as ``METHOD path body`` lines. ``chown`` and ``qp-served-models`` are stand-ins
too, so nothing needs root, the ``ubuntu`` user or the project installed. Lives
here, not in ``tests/helpers``, because it goes away with the emulation.

Example::

    machine = make_machine(tmp_path)
    run_machine_script(machine, CACHE_BUCKET_SH, "models-save")
    machine.bucket_objects()  # ["models/openvino/embedder.tar"]
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from repo_paths import REPO_ROOT

MACHINE_DIR = REPO_ROOT / "emulation" / "machine"
CACHE_BUCKET_SH = MACHINE_DIR / "cache_bucket.sh"
INDEX_SNAPSHOT_SH = MACHINE_DIR / "index_snapshot.sh"
COMMON_SH = REPO_ROOT / "emulation" / "terraform" / "scripts" / "common.sh"
BUCKET = "test-cache"
ES_URL = "http://es.test:9200"
TOKEN = "fake-token"
EMBEDDER = "Snowflake/snowflake-arctic-embed-l-v2.0"
RERANKER = "antoinelouis/crossencoder-camembert-base-mmarcoFR"
# What qp-served-models prints for those two: one line per served model, its
# name, type, source id and alias. The reranker comes first here, so that a
# script must pick the embedder by its type, not by its place. A test holds
# these lines to the real command's.
SERVED_MODEL_LINES = (
    f"reranker reranker {RERANKER} crossencoder-camembert-base-mmarcoFR",
    f"embedder embedder {EMBEDDER} snowflake-arctic-embed-l-v2.0",
)
PASSAGES = 1592

FAKE_CURL = r"""
import json, os, shutil, sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

VALUED = {"-H", "-o", "-w", "-T", "-X", "--data", "--connect-timeout", "--max-time", "--retry"}
STORAGE = "https://storage.googleapis.com/"
SNAPSHOTS = "/_snapshot/rag-emulation-cache"
DOWNLOADS = "https://downloads.test/"


def parse(argv):
    options, flags, url, args = {}, "", "", iter(argv)
    for arg in args:
        if arg in VALUED:
            options.setdefault(arg, []).append(next(args))
        elif arg.startswith("-"):
            flags += arg
        else:
            url = arg
    return options, flags, url


def storage(url, options, bucket_root):
    parts = urlsplit(url)
    if parts.path.startswith("/storage/v1/b/"):
        bucket = bucket_root / parts.path.split("/")[4]
        prefix = parse_qs(parts.query).get("prefix", [""])[0]
        names = sorted(str(p.relative_to(bucket)) for p in bucket.rglob("*") if p.is_file())
        items = [{"name": n, "size": str((bucket / n).stat().st_size), "updated": "2026-11-14T10:00:00Z"} for n in names if n.startswith(prefix)]
        return 200, json.dumps({"kind": "storage#objects", "items": items} if items else {"kind": "storage#objects"})
    target = bucket_root / parts.path.lstrip("/")
    if "-T" in options:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(options["-T"][0], target)
        return 200, ""
    if not target.is_file():
        return 404, "<Error><Code>NoSuchKey</Code></Error>"
    return 200, target.read_bytes()


def elasticsearch(method, path, body, state):
    with open(state / "es.log", "a") as log:
        log.write(f"{method} {path} {body}".rstrip() + "\n")
    snapshots = state / "snapshots"
    snapshots.mkdir(exist_ok=True)
    route = path.split("?")[0]
    if route == SNAPSHOTS + "/_all":
        found = [{"snapshot": p.name, "state": "SUCCESS", "end_time": "2026-11-14T10:00:00Z", "metadata": json.loads(p.read_text() or "{}")} for p in sorted(snapshots.iterdir())]
        return 200, json.dumps({"snapshots": found})
    if route.startswith(SNAPSHOTS + "/"):
        name = route[len(SNAPSHOTS) + 1:].split("/")[0]
        exists = (snapshots / name).exists()
        if method == "PUT":
            (snapshots / name).write_text(json.dumps(json.loads(body)["metadata"]))
            return 200, json.dumps({"snapshot": {"snapshot": name, "state": "SUCCESS"}})
        if method == "DELETE" and exists:
            (snapshots / name).unlink()
        return (200, json.dumps({"snapshots": [{"snapshot": name}]})) if exists else (404, '{"error": "snapshot_missing_exception"}')
    if route.endswith("/_count"):
        return 200, json.dumps({"count": int(os.environ["FAKE_ES_PASSAGES"])})
    if method == "DELETE":
        return 404, '{"error": "index_not_found_exception"}'
    return 200, '{"acknowledged": true}'


def answer(url, options):
    state = Path(os.environ["FAKE_STATE_DIR"])
    if url.endswith("/service-accounts/default/token"):
        return 200, json.dumps({"access_token": os.environ["FAKE_TOKEN"], "expires_in": 3599})
    if url.startswith(STORAGE):
        return storage(url, options, state / "bucket")
    if url.startswith(DOWNLOADS):
        target = state / "downloads" / url[len(DOWNLOADS):]
        return (200, target.read_bytes()) if target.is_file() else (404, "not found")
    if url.startswith(os.environ["FAKE_ES_URL"]):
        method = options.get("-X", ["GET"])[0]
        return elasticsearch(method, url[len(os.environ["FAKE_ES_URL"]):], options.get("--data", [""])[0], state)
    return 599, f"no stand-in answers {url}"


def main():
    options, flags, url = parse(sys.argv[1:])
    state = Path(os.environ["FAKE_STATE_DIR"])
    with open(state / "curl.log", "a") as log:
        log.write(" ".join(sys.argv[1:]) + "\n")
    status, body = answer(url, options)
    body = body.encode() if isinstance(body, str) else body
    if "f" in flags.replace("--", "") and status >= 400:
        sys.exit(22)
    if "-o" in options:
        Path(options["-o"][0]).write_bytes(body)
    else:
        sys.stdout.buffer.write(body)
    if "-w" in options:
        sys.stdout.write(str(status))


main()
"""


@dataclass(frozen=True)
class FakeMachine:
    """A machine of the emulation in ``tmp_path``: its PATH, bucket and dataset."""

    root: Path
    bin_dir: Path
    dataset: Path
    cache_env: Path

    @property
    def bucket(self) -> Path:
        """The folder that plays the cache bucket's objects."""
        return self.root / "bucket" / BUCKET

    def bucket_objects(self) -> list[str]:
        """The bucket's object names, sorted."""
        found = self.bucket.rglob("*")
        return sorted(str(p.relative_to(self.bucket)) for p in found if p.is_file())

    def es_calls(self) -> list[str]:
        """The Elasticsearch calls, ``METHOD path [body]`` each, in order."""
        log = self.root / "es.log"
        return log.read_text().splitlines() if log.exists() else []

    def curl_argv(self) -> str:
        """Every curl command line, one per line."""
        return (self.root / "curl.log").read_text()


def write_stand_in(path: Path, text: str) -> None:
    """An executable script at ``path``."""
    path.write_text(text)
    path.chmod(0o755)


def make_machine(tmp_path: Path) -> FakeMachine:
    """Stand-ins for curl, chown and qp-served-models, an empty bucket and dataset."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    write_stand_in(bin_dir / "curl", f"#!{sys.executable}\n{FAKE_CURL}")
    write_stand_in(
        bin_dir / "chown", f'#!/usr/bin/env bash\necho "$@" >> "{tmp_path}/chown.log"\n'
    )
    write_stand_in(
        bin_dir / "qp-served-models",
        "#!/usr/bin/env bash\ncat <<'LINES'\n"
        + "\n".join(SERVED_MODEL_LINES)
        + "\nLINES\n",
    )
    (tmp_path / "bucket" / BUCKET).mkdir(parents=True)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    cache_env = tmp_path / "cache.env"
    cache_env.write_text(f"CACHE_BUCKET={BUCKET}\nDATASET_DIR={dataset}\n")
    return FakeMachine(tmp_path, bin_dir, dataset, cache_env)


def machine_env(machine: FakeMachine, **env: str) -> dict[str, str]:
    """The stand-ins first on PATH, the fakes' settings, no QP_* from the shell."""
    inherited = {k: v for k, v in os.environ.items() if not k.startswith("QP_")}
    return {
        **inherited,
        "PATH": f"{machine.bin_dir}:{os.environ['PATH']}",
        "TMPDIR": str(machine.root),
        "QP_EMULATION_CACHE_ENV": str(machine.cache_env),
        "FAKE_STATE_DIR": str(machine.root),
        "FAKE_ES_URL": ES_URL,
        "FAKE_ES_PASSAGES": str(PASSAGES),
        "FAKE_TOKEN": TOKEN,
        **env,
    }


def run_machine_script(
    machine: FakeMachine, script: Path, *args: str, **env: str
) -> subprocess.CompletedProcess[str]:
    """Run one of ``emulation/machine/`` with bash, as the Makefile does."""
    return subprocess.run(
        ["bash", str(script), *args],
        env=machine_env(machine, **env),
        capture_output=True,
        text=True,
        timeout=60,
    )


def run_common_sh_function(
    machine: FakeMachine, function: str, *args: str
) -> subprocess.CompletedProcess[str]:
    """One function of ``common.sh``, sourced as a machine script sources it.

    Example::

        run_common_sh_function(machine, "fetch_ovms_bundle", str(repo), str(bundle))
    """
    command = '. "$1" && shift && "$@"'
    return subprocess.run(
        ["bash", "-c", command, "_", str(COMMON_SH), function, *args],
        env=machine_env(machine, QP_EMULATION_SOURCED="1"),
        capture_output=True,
        text=True,
        timeout=60,
    )


def restore_from_common_sh(
    machine: FakeMachine, prefix: str
) -> subprocess.CompletedProcess[str]:
    """``restore_cached_models`` from ``common.sh``, as a startup script calls it."""
    return run_common_sh_function(
        machine, "restore_cached_models", BUCKET, prefix, str(machine.dataset)
    )
