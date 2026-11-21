#!/usr/bin/env bash
# The Elasticsearch index as a snapshot in the cache bucket, through Elasticsearch's
# own snapshot API and its built-in `gcs` repository type (emulation/README.md, "The
# cache bucket"). No credential is configured: on Compute Engine the repository uses
# the Elasticsearch machine's service account. Run in the Workspace's clone, from
# the use case's directory (make -C emulation corpus-save), where qp-served-models
# finds the use case's workflow file:
#   index_snapshot.sh save <prefix>      snapshot the index as <prefix>-<embedder>
#   index_snapshot.sh restore <prefix>   replace the index with that snapshot
#   index_snapshot.sh list               each snapshot's name, state, end time and metadata
# Reads QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL (required),
# QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__INDEX (default rag_passages) and the bucket
# from /etc/rag-emulation/cache.env (QP_EMULATION_CACHE_ENV names another, for tests).
set -euo pipefail

# common.sh is the startup scripts' prelude; with this set it only defines functions.
export QP_EMULATION_SOURCED=1
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source-path=SCRIPTDIR
# shellcheck source=../terraform/scripts/common.sh
. "$HERE/../terraform/scripts/common.sh"

# Fixed by the cache's layout (README): the repository's name in Elasticsearch, and
# the folder of the bucket that holds it.
SNAPSHOT_REPOSITORY=rag-emulation-cache
SNAPSHOT_BASE_PATH=elasticsearch
# curl's status for an Elasticsearch 404 (curl's own codes stop at 99).
ES_NOT_FOUND=100

ES_URL=${QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL:-}
[ -n "$ES_URL" ] \
  || die "QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__URL is not set; expected Elasticsearch's URL (http://<address>:9200), as the Workspace's shell exports it"
ES_INDEX=${QP_COMPONENTS__VECTOR_STORE__ELASTICSEARCH__INDEX:-rag_passages}
CACHE_BUCKET=$(cache_config_value CACHE_BUCKET)

# es_call <method> <path> [json body]: Elasticsearch's answer on stdout for a 2xx,
# ES_NOT_FOUND for a 404; any other answer ends the script, showing it. A snapshot
# or a restore waits for completion, hence the long --max-time.
es_call() {
  local answer status
  answer=$(mktemp)
  status=$(curl -sS --connect-timeout 10 --max-time 900 --retry 3 -X "$1" \
    -H 'Content-Type: application/json' ${3:+--data "$3"} \
    -o "$answer" -w '%{http_code}' "$ES_URL$2") || true
  case $status in
    2??) cat "$answer"; rm -f "$answer" ;;
    404) rm -f "$answer"; return "$ES_NOT_FOUND" ;;
    *) die "Elasticsearch answered HTTP $status to $1 $2: $(cat "$answer"; rm -f "$answer")" ;;
  esac
}

# es_require <method> <path> [json body]: es_call, where a 404 is an error too.
es_require() {
  es_call "$@" || die "Elasticsearch answered HTTP 404 to $1 $2"
}

# es_delete_if_present <path>
es_delete_if_present() {
  es_call DELETE "$1" >/dev/null || [ $? -eq "$ES_NOT_FOUND" ]
}

# json_field <dotted.path>: one field of the JSON on stdin.
json_field() {
  python3 -c '
import json, sys
value = json.load(sys.stdin)
for key in sys.argv[1].split("."):
    value = value[key]
print(value)' "$1"
}

register_repository() {
  es_require PUT "/_snapshot/$SNAPSHOT_REPOSITORY" \
    "{\"type\": \"gcs\", \"settings\": {\"bucket\": \"$CACHE_BUCKET\", \"base_path\": \"$SNAPSHOT_BASE_PATH\"}}" >/dev/null
}

# snapshot_name <prefix>: <prefix>-<embedder source model>, lowercased, every
# character outside [a-z0-9._-] made "-". An index is only valid for the embedder
# and the analyzer it was built with: a changed embedder must miss the cache, not
# restore a stale index. qp-served-models prints one line per served model, its
# words shell-quoted: name, type, source id, alias. The first embedder's line is
# read with awk and its quotes dropped, never evaluated. awk reads every line
# (no early exit), so the command never writes into a closed pipe.
snapshot_name() {
  local embedder
  embedder=$(qp-served-models | awk '$2 == "embedder" && !seen { print $3; seen = 1 }' | tr -d "'")
  [ -n "$embedder" ] \
    || die "qp-served-models printed no embedder line (<name> embedder <source id> <alias>); expected the project's virtualenv on PATH, as in the Workspace's shell, and the use case's directory as the working directory"
  printf '%s-%s\n' "$1" "$embedder" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9._-]/-/g'
}

# snapshot_request: the snapshot's body, with the clone's commit and the count of
# vector-store records. The metadata keeps its key "passages": the cache bucket
# holds snapshots written with it.
snapshot_request() {
  local commit records
  commit=$(git -C "$HERE" rev-parse HEAD)
  records=$(es_require GET "/$ES_INDEX/_count" | json_field count)
  python3 -c '
import json, sys
index, commit, records = sys.argv[1:]
metadata = {"commit": commit, "passages": int(records)}
print(json.dumps({"indices": index, "include_global_state": False, "metadata": metadata}))
' "$ES_INDEX" "$commit" "$records"
}

save_snapshot() {
  local name=$1 request answer state
  request=$(snapshot_request)
  es_delete_if_present "/_snapshot/$SNAPSHOT_REPOSITORY/$name"
  answer=$(es_require PUT "/_snapshot/$SNAPSHOT_REPOSITORY/$name?wait_for_completion=true" "$request")
  state=$(printf '%s' "$answer" | json_field snapshot.state) || state=unknown
  [ "$state" = SUCCESS ] || die "snapshot $name ended in state $state, expected SUCCESS; Elasticsearch answered: $answer"
  echo "saved: snapshot $name of index $ES_INDEX in gs://$CACHE_BUCKET/$SNAPSHOT_BASE_PATH"
}

restore_snapshot() {
  local name=$1 records
  es_call GET "/_snapshot/$SNAPSHOT_REPOSITORY/$name" >/dev/null \
    || die "no snapshot $name in gs://$CACHE_BUCKET/$SNAPSHOT_BASE_PATH; make -C emulation corpus corpus-save creates it (same corpus, analyzer and embedder)"
  es_delete_if_present "/$ES_INDEX"
  es_require POST "/_snapshot/$SNAPSHOT_REPOSITORY/$name/_restore?wait_for_completion=true" \
    "{\"indices\": \"$ES_INDEX\", \"include_global_state\": false}" >/dev/null
  records=$(es_require GET "/$ES_INDEX/_count" | json_field count)
  echo "restored: snapshot $name; index $ES_INDEX holds $records vector-store records"
}

list_snapshots() {
  es_require GET "/_snapshot/$SNAPSHOT_REPOSITORY/_all" | python3 -c '
import json, sys
for snapshot in json.load(sys.stdin)["snapshots"]:
    fields = ("snapshot", "state", "end_time")
    print(*(snapshot.get(field, "-") for field in fields), json.dumps(snapshot.get("metadata", {})), sep="\t")
'
}

# main <command> [prefix]: the snapshot's name is settled before any call, so a
# failing qp-served-models stops the script instead of naming a snapshot "".
main() {
  local name=""
  case ${1:-} in
    save | restore) name=$(snapshot_name "${2:?$1 needs the snapshot prefix (<corpus>-<analyzer>)}") ;;
    list) ;;
    *) die "unknown command '${1:-}'; expected save <prefix>, restore <prefix> or list" ;;
  esac
  register_repository
  case $1 in
    save) save_snapshot "$name" ;;
    restore) restore_snapshot "$name" ;;
    list) list_snapshots ;;
  esac
}

main "$@"
