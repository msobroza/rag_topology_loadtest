"""Stand-in binaries for running ``domino/app.sh`` without installing anything.

``ovms``, ``nginx``, ``qp-*``, ``uv``, ``sudo`` and ``apt-get`` are replaced by
scripts that record their argv (and the ``QP_*`` variables they were started
with) under ``tmp_path``; ``qp-ovms-config`` also prints the config it would
write (``served_config`` reads it back). ``vllm`` is on PATH only where a test
puts one (``write_vllm_stand_in``), or in the virtualenv ``uv venv`` makes for
it; ``qp-vllm-config`` prints the words of the tests' workflow file. Used
by the launch script's own tests and by the deploy tests that run every entry
script through it.

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
# Where an App that installs finds doclang's core, which no index has, and the
# name a build of it gives its wheel.
WHEELS_VARIABLE = "QP_BOOTSTRAP_WHEELS"
DOCLANG_WHEEL = "doclang_domain_core-0.1.0-py3-none-any.whl"


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
    logs to venv-<name>.log, venv-<name>.env and venv-<name>.cwd instead
    (``venv_vllm_fakes`` says what each new venv gets for vLLM). ``uv``
    appends the directory each of its commands ran in to uv.cwd.
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
    # Shadows a real one on the developer's PATH, as qp-ovms-config's does.
    write_vllm_config_stand_in(bin_dir / "qp-vllm-config", tmp_path)
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
        f'pwd >> "{tmp_path}/uv.cwd"\n'
        f'if [ "$1" = venv ]; then mkdir -p "$2/bin"\n'
        f"{venv_fakes}{venv_vllm_fakes(tmp_path)}fi\n",
    )
    return bin_dir


def venv_vllm_fakes(tmp_path: Path) -> str:
    """Shell lines of the ``uv`` stand-in giving a new virtualenv its vLLM command.

    The virtualenv made for vLLM (``*/vllm-venv``) gets a ``vllm`` that logs as
    ``venv-vllm``; any other, the project's, gets a ``qp-vllm-config``
    (``write_vllm_config_stand_in``), which an App uses when none is on PATH.
    """
    vllm, config = tmp_path / "venv-vllm-stand-in", tmp_path / "venv-qp-vllm-config"
    write_vllm_stand_in(vllm, tmp_path, "venv-vllm")
    write_vllm_config_stand_in(config, tmp_path)
    return (
        'case "$2" in\n'
        f'  */vllm-venv) cp "{vllm}" "$2/bin/vllm" ;;\n'
        f'  *) cp "{config}" "$2/bin/qp-vllm-config" ;;\n'
        "esac\n"
    )


def write_vllm_stand_in(path: Path, tmp_path: Path, name: str = "vllm") -> None:
    """A ``vllm`` that logs what it was started with, under ``name``.

    Its arguments go to <name>.log one a line, so a test sees whether one
    holding spaces reached vLLM whole (``vllm_argv`` reads them back); its
    ``QP_*`` and ``VLLM_*`` variables to <name>.env, ``LD_PRELOAD`` to
    <name>.preload and its directory to <name>.cwd. Shell builtins only, so it
    runs on any PATH.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env bash\n"
        f'printf \'%s\\n\' "$@" >> "{tmp_path}/{name}.log"\n'
        "for variable in $(compgen -e); do\n"
        '  case $variable in QP_* | VLLM_*) echo "$variable=${!variable}" ;; esac\n'
        f'done >> "{tmp_path}/{name}.env"\n'
        f'echo "${{LD_PRELOAD:-}}" >> "{tmp_path}/{name}.preload"\n'
        f'pwd >> "{tmp_path}/{name}.cwd"\n'
    )
    path.chmod(0o755)


def vllm_argv(tmp_path: Path, name: str = "vllm") -> list[str]:
    """The arguments the ``vllm`` stand-in ``name`` was started with, [] if it never ran."""
    return log(tmp_path, name).splitlines()


def write_vllm_config_stand_in(path: Path, tmp_path: Path) -> None:
    """A ``qp-vllm-config`` that logs its argv to qp-vllm-config.log and prints a line.

    ``--serve NAME`` prints ``fixture-org/fixture-NAME``, ``--served-model-name
    fixture-NAME`` and the default cache flag, as the real one does on the
    tests' workflow file; ``--environment NAME`` prints an empty line, as at the
    defaults. Shell builtins only, so it runs on any PATH.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "$@" >> "{tmp_path}/qp-vllm-config.log"\n'
        'if [ "$1" = --environment ]; then echo; exit 0; fi\n'
        'echo "fixture-org/fixture-$2 --served-model-name fixture-$2 '
        f'--kv-cache-memory-bytes {8 * 2**30}"\n'
    )
    path.chmod(0o755)


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


def wheels_env(tmp_path: Path) -> dict[str, str]:
    """``QP_BOOTSTRAP_WHEELS`` naming a directory that holds a wheel of doclang's core.

    The wheel is an empty file under the name a build of doclang's core gives
    it: the ``uv`` stand-in installs nothing, so only its name is read. The
    directory plays the one on the Dataset that the deploy setting names.

    Example::

        run_app(tmp_path, bin_dir, QP_BOOTSTRAP="1", **wheels_env(tmp_path))
    """
    wheels = tmp_path / "wheels"
    wheels.mkdir(exist_ok=True)
    (wheels / DOCLANG_WHEEL).touch()
    return {WHEELS_VARIABLE: str(wheels)}


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
