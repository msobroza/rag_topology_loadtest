#!/usr/bin/env bash
# One-time setup in Google Cloud Shell, which no longer ships Terraform: installs a
# pinned Terraform into ~/bin (kept across Cloud Shell sessions, on its PATH) and
# creates the private bucket that holds Terraform's state (it contains the SSH key
# the fake Domino uses). Safe to run again.
#
#   emulation/cloudshell/setup.sh <state bucket name> [region]
#
# TERRAFORM_VERSION overrides the pinned version.
set -euo pipefail

TERRAFORM_VERSION=${TERRAFORM_VERSION:-1.12.2}
BUCKET=${1:?usage: $0 <state bucket name> [region]}
REGION=${2:-europe-west1}
PROJECT=$(gcloud config get-value project 2>/dev/null)
[ -n "$PROJECT" ] || { echo "setup.sh: no project set; run: gcloud config set project <project id>" >&2; exit 1; }

fetch() {
  curl -fsSL --connect-timeout 10 --max-time 300 --retry 3 -o "$2" "$1"
}

install_terraform() {
  if [ "$("$HOME/bin/terraform" version -json 2>/dev/null | python3 -c 'import json, sys; print(json.load(sys.stdin)["terraform_version"])' 2>/dev/null)" = "$TERRAFORM_VERSION" ]; then
    echo "terraform $TERRAFORM_VERSION already in ~/bin"
    return 0
  fi
  local base="https://releases.hashicorp.com/terraform/$TERRAFORM_VERSION" zip="terraform_${TERRAFORM_VERSION}_linux_amd64.zip" dir
  dir=$(mktemp -d)
  fetch "$base/$zip" "$dir/$zip"
  fetch "$base/terraform_${TERRAFORM_VERSION}_SHA256SUMS" "$dir/SHA256SUMS"
  (cd "$dir" && grep " $zip\$" SHA256SUMS | sha256sum --check --status)
  mkdir -p "$HOME/bin"
  unzip -o -q "$dir/$zip" terraform -d "$HOME/bin"
  rm -rf "$dir"
  echo "installed terraform $TERRAFORM_VERSION in ~/bin"
}

create_state_bucket() {
  if gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1; then
    echo "state bucket gs://$BUCKET exists"
    return 0
  fi
  gcloud storage buckets create "gs://$BUCKET" --project "$PROJECT" --location "$REGION" \
    --uniform-bucket-level-access --public-access-prevention
  gcloud storage buckets update "gs://$BUCKET" --versioning
  echo "created state bucket gs://$BUCKET"
}

install_terraform
create_state_bucket
echo "next: make -C emulation init BUCKET=$BUCKET   (from the repository root)"
