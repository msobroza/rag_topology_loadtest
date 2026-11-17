"""A stand-in ``ovms`` executable, and an export like OVMS's exporter writes.

``write_ovms_stand_in`` writes an ``ovms`` that takes the real binary's options
(``--rest_bind_address``, ``--rest_port``, ``--config_path``), finds the graph
the config names, records its argv, its ``plugin_config`` and the files beside
the graph as one JSON line, then does what ``behaviour`` says:

- ``answers``: serves ``/v3/rerank`` (Cohere's shape) and ``/v3/embeddings``
  (OpenAI's) for the alias the config names (404 for another model). It holds
  ``NUM_STREAMS`` slots and each request keeps one for ``request_s``, so more
  streams answer several requests at once sooner, as OVMS would;
- ``exits``: exits with code 3 at once;
- ``silent``: never listens;
- ``stubborn``: answers, but ignores SIGTERM, so only a kill stops it.

``write_ovms_export`` writes ``config_<component>.json`` and ``<alias>/`` with a
``graph.pbtxt`` holding the exporter's one ``plugin_config`` line, and the
folder's ``weight-format`` record, as ``ovms/export_models.sh`` writes it.

Example::

    bin_dir = write_ovms_stand_in(tmp_path / "bin", tmp_path / "ovms.jsonl")
    write_ovms_export(tmp_path / "models", "reranker", "my-alias")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# The calculator options of the exporter's template, trimmed to what matters.
GRAPH_TEMPLATE = """input_stream: "REQUEST_PAYLOAD:input"
output_stream: "RESPONSE_PAYLOAD:output"
node {
  name: "%(alias)s"
  calculator: "%(calculator)s"
  node_options: {
    [type.googleapis.com / mediapipe.%(calculator)sOptions]: {
      models_path: "./",
      target_device: "CPU",
      plugin_config: '{"NUM_STREAMS": "1" }',
    }
  }
}
"""
MODEL_FILES = ("openvino_model.xml", "openvino_model.bin", "tokenizer.json")

_SCRIPT = """#!%(python)s
import argparse, json, re, signal, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BEHAVIOUR, RECORD, REQUEST_S = %(behaviour)r, %(record)r, %(request_s)r
parser = argparse.ArgumentParser()
parser.add_argument("--rest_bind_address", default="0.0.0.0")
parser.add_argument("--rest_port", type=int, required=True)
parser.add_argument("--config_path", required=True)
args = parser.parse_args()
config_path = Path(args.config_path)
model = json.loads(config_path.read_text())["model_config_list"][0]["config"]
folder = config_path.parent / model["base_path"]
graph = (folder / "graph.pbtxt").read_text()
plugin = json.loads(re.search(r"plugin_config: '([^']*)'", graph).group(1))
line = {"argv": sys.argv[1:], "plugin_config": plugin,
        "files": sorted(p.name for p in folder.iterdir())}
with open(RECORD, "a") as record:
    record.write(json.dumps(line) + "\\n")
print("stand-in ovms: starting", flush=True)
if BEHAVIOUR == "exits":
    sys.exit(3)
if BEHAVIOUR == "silent":
    time.sleep(600)
if BEHAVIOUR == "stubborn":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
slots = threading.Semaphore(int(plugin["NUM_STREAMS"]))


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("model") != model["name"]:
            return self.answer(404, {"error": "unknown model"})
        with slots:
            time.sleep(REQUEST_S)
        if self.path == "/v3/rerank":
            results = [{"index": i, "relevance_score": 1.0 / (i + 1)}
                       for i in range(len(body["documents"]))]
            return self.answer(200, {"results": results})
        if self.path == "/v3/embeddings":
            data = [{"index": i, "embedding": [0.1, 0.2]}
                    for i in range(len(body["input"]))]
            return self.answer(200, {"data": data})
        return self.answer(404, {"error": "no such route"})

    def answer(self, status, payload):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


ThreadingHTTPServer((args.rest_bind_address, args.rest_port), Handler).serve_forever()
"""


def write_ovms_stand_in(
    bin_dir: Path,
    record: Path,
    *,
    behaviour: str = "answers",
    request_s: float = 0.05,
) -> Path:
    """Write the stand-in ``ovms`` into ``bin_dir``; return ``bin_dir``."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "ovms"
    script.write_text(
        _SCRIPT
        % {
            "python": sys.executable,
            "behaviour": behaviour,
            "record": str(record),
            "request_s": request_s,
        }
    )
    script.chmod(0o755)
    return bin_dir


def ovms_starts(record: Path) -> list[dict[str, object]]:
    """Every start of the stand-in, in order: argv, plugin_config, files."""
    if not record.exists():
        return []
    return [json.loads(line) for line in record.read_text().splitlines()]


def write_ovms_export(
    models_dir: Path, component: str, alias: str, weight_format: str = "fp32"
) -> Path:
    """Write one model's export as OVMS's exporter lays it out; return its folder.

    The models folder's ``weight-format`` record says ``weight_format``: fp32 is
    what the settings ask for by default.
    """
    calculator = (
        "RerankCalculatorOV" if component == "reranker" else "EmbeddingsCalculatorOV"
    )
    folder = models_dir / alias
    folder.mkdir(parents=True)
    (folder / "graph.pbtxt").write_text(
        GRAPH_TEMPLATE % {"alias": alias, "calculator": calculator}
    )
    for name in MODEL_FILES:
        (folder / name).write_text(f"{name} of {alias}")
    config = {"model_config_list": [{"config": {"name": alias, "base_path": alias}}]}
    (models_dir / f"config_{component}.json").write_text(json.dumps(config))
    (models_dir / "weight-format").write_text(f"{weight_format}\n")
    return folder
