"""Stand-in binaries for running ``domino/app.sh`` without installing anything.

``ovms``, ``nginx``, ``qp-*``, ``uv``, ``sudo`` and ``apt-get`` are replaced by
scripts that record their argv (and the ``QP_*`` variables they were started
with) under ``tmp_path``; ``qp-ovms-config`` also prints the config it would
write (``served_config`` reads it back). Used by the launch script's own tests
and by the deploy tests that run every entry script through it.

Example::

    bin_dir = make_stand_ins(tmp_path)
    subprocess.run(["bash", str(APP_SH)], env=stand_in_env(tmp_path, bin_dir))
    log(tmp_path, "qp-serve")   # "--host 0.0.0.0 --port 8888 --domino\\n"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from repo_paths import TOOLKIT_ROOT

APP_SH = TOOLKIT_ROOT / "domino" / "app.sh"
FAKES = ("ovms", "nginx", "qp-serve", "qp-model-server", "apt-get")
VENV_FAKES = ("qp-serve", "qp-model-server")  # what ``uv venv`` puts in a venv
PROXY_PAIR = {"ovms": "nginx", "nginx": "ovms"}  # run together behind the prefix proxy


def write_fake(path: Path, log_file: Path, extra: str = "") -> None:
    """An executable that appends its argv to ``log_file``, then runs ``extra``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/usr/bin/env bash\necho "$@" >> "{log_file}"\n{extra}exit 0\n')
    path.chmod(0o755)


def stay_until_the_other_is_done(tmp_path: Path, name: str) -> str:
    """Shell lines ending the fake ``ovms`` and ``nginx``; "" for any other fake.

    Behind the prefix proxy app.sh runs the two together and stops both as soon as
    one exits. A fake that left first got the other killed before it had logged
    anything, which failed these tests on a loaded machine. Each one marks itself
    done, then stays until the other is done too (at most 10 s).
    """
    if name not in PROXY_PAIR:
        return ""
    other_done = tmp_path / f"{PROXY_PAIR[name]}.done"
    return (
        f'touch "{tmp_path}/{name}.done"\n'
        'if [ "${QP_OVMS_PREFIX_PROXY:-}" = 1 ]; then\n'
        f'  for _ in $(seq 100); do [ -e "{other_done}" ] && break; sleep 0.1; done\n'
        "fi\n"
    )


def write_ovms_config_stand_in(
    path: Path, tmp_path: Path, name: str = "qp-ovms-config"
) -> None:
    """A ``qp-ovms-config`` that logs its argv and prints the config it would write.

    It prints ``<serving dir>/config_<part>.json``, as the real one does, and
    appends it to ``<name>.out`` (argv to ``<name>.log``); only shell builtins,
    so it runs on any PATH. Asked ``--server-flags``, it logs to
    ``<name>-server-flags.log`` instead and prints ``$FAKE_OVMS_SERVER_FLAGS``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = --server-flags ]; then\n'
        f'  echo "$@" >> "{tmp_path}/{name}-server-flags.log"\n'
        '  echo "${FAKE_OVMS_SERVER_FLAGS:-}"\n'
        "  exit 0\n"
        "fi\n"
        f'echo "$@" >> "{tmp_path}/{name}.log"\n'
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in --serve) part=$2 ;; --serving-dir) dir=$2 ;; esac\n'
        "  shift\n"
        "done\n"
        f'echo "$dir/config_$part.json" >> "{tmp_path}/{name}.out"\n'
        'echo "$dir/config_$part.json"\n'
    )
    path.chmod(0o755)


def put_the_real_toolkit_command_on_path(bin_dir: Path, name: str) -> None:
    """Replace the stand-in ``name`` (``qp-ovms-config``, ...) by the real command.

    It runs this interpreter on ``query_processing.entrypoints.<module>``, the
    module named like the command, so it reads the settings the script's
    variables and the workflow file give, as the App's own command would.
    """
    module = name.replace("-", "_")
    run_main = (
        f"from query_processing.entrypoints.{module} import main; "
        "raise SystemExit(main())"
    )
    script = bin_dir / name
    script.write_text(
        f"#!/usr/bin/env bash\nexec '{sys.executable}' -c '{run_main}' \"$@\"\n"
    )
    script.chmod(0o755)


def served_config(tmp_path: Path, name: str = "qp-ovms-config") -> str:
    """The config path the ``qp-ovms-config`` stand-in printed last, "" if none."""
    out = tmp_path / f"{name}.out"
    return out.read_text().splitlines()[-1] if out.exists() else ""


def make_stand_ins(tmp_path: Path) -> Path:
    """Fake binaries that append their argv (and nginx its config) to <name>.log.

    Each also appends its ``QP_*`` variables to <name>.env and the directory
    it was started in to <name>.cwd; a fake installed in a venv by ``uv venv``
    logs to venv-<name>.log, venv-<name>.env and venv-<name>.cwd instead.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in FAKES:
        script = bin_dir / name
        script.write_text(
            "#!/usr/bin/env bash\n"
            f'echo "$@" >> "{tmp_path}/{name}.log"\n'
            f'env | grep "^QP_" | sort >> "{tmp_path}/{name}.env" || true\n'
            f'pwd >> "{tmp_path}/{name}.cwd"\n'
            + (
                f'[ "$1" = -c ] && cp "$2" "{tmp_path}/nginx.conf"\n'
                if name == "nginx"
                else ""
            )
            + stay_until_the_other_is_done(tmp_path, name)
            + "exit 0\n"
        )
        script.chmod(0o755)
    write_ovms_config_stand_in(bin_dir / "qp-ovms-config", tmp_path)
    # sudo runs its command; ``uv venv <dir>`` makes a venv whose qp-* log apart.
    write_fake(bin_dir / "sudo", tmp_path / "sudo.log", 'exec "$@"\n')
    venv_fakes = "".join(
        "printf '#!/usr/bin/env bash\\n"
        f'echo "$@" >> {tmp_path}/venv-{name}.log\\n'
        f'env | grep "^QP_" | sort >> {tmp_path}/venv-{name}.env\\n'
        f"pwd >> {tmp_path}/venv-{name}.cwd\\n' "
        f'> "$2/bin/{name}"; chmod +x "$2/bin/{name}"\n'
        for name in VENV_FAKES
    )
    write_fake(
        bin_dir / "uv",
        tmp_path / "uv.log",
        f'if [ "$1" = venv ]; then mkdir -p "$2/bin"\n{venv_fakes}fi\n',
    )
    return bin_dir


def stand_in_env(tmp_path: Path, bin_dir: Path, **env: str) -> dict[str, str]:
    """The environment to run app.sh in: the stand-ins first on PATH, no QP_*."""
    # A QP_* variable from the developer's shell would leak into the role under test.
    inherited = {k: v for k, v in os.environ.items() if not k.startswith("QP_")}
    home, temporary = tmp_path / "home", tmp_path / "tmp"
    home.mkdir(exist_ok=True)
    temporary.mkdir(exist_ok=True)  # mktemp's folders land here, not in /tmp
    path = f"{bin_dir}:{os.environ['PATH']}"
    paths = {"PATH": path, "HOME": str(home), "TMPDIR": str(temporary)}
    return {**inherited, **paths, **env}


def model_server_env(kind: str, component: str) -> dict[str, str]:
    """The variables that make app.sh a model server of ``component`` by ``kind``."""
    return {
        "QP_ROLE": "model-server",
        "QP_MODEL_SERVER_KIND": kind,
        "QP_MODEL_SERVER_COMPONENT": component,
    }


def log(tmp_path: Path, name: str) -> str:
    """What the stand-in ``name`` logged, "" when it never ran."""
    path = tmp_path / f"{name}.log"
    return path.read_text() if path.exists() else ""


def started_env(tmp_path: Path, name: str) -> list[str]:
    """The QP_* variables the stand-in ``name`` was started with, ``NAME=value`` each."""
    return (tmp_path / f"{name}.env").read_text().splitlines()


def started_in(tmp_path: Path, name: str) -> Path:
    """The directory the stand-in ``name`` was last started in."""
    return Path((tmp_path / f"{name}.cwd").read_text().splitlines()[-1])


def models_dir(tmp_path: Path, weight_format: str = "fp32") -> Path:
    """A models folder holding both OVMS configs, so no OVMS model server exports.

    Its ``weight-format`` record says ``weight_format``; fp32 is what app.sh asks
    for when QP_LOCAL_MODELS__OPENVINO_WEIGHT_FORMAT is unset.
    """
    models = tmp_path / "models"
    models.mkdir()
    (models / "config_reranker.json").write_text("{}")
    (models / "config_embedder.json").write_text("{}")
    (models / "weight-format").write_text(f"{weight_format}\n")
    return models


def ovms_bundle(tmp_path: Path) -> Path:
    """An unpacked OVMS bundle whose ``bin/ovms`` logs argv, LD_LIBRARY_PATH and cwd."""
    bundle = tmp_path / "bundle" / "ovms"
    (bundle / "lib").mkdir(parents=True)
    extra = (
        f'echo "$LD_LIBRARY_PATH" >> "{tmp_path}/bundle-ovms.ld"\n'
        f'pwd >> "{tmp_path}/bundle-ovms.cwd"\n'
    )
    write_fake(bundle / "bin" / "ovms", tmp_path / "bundle-ovms.log", extra)
    return bundle
