# Terraform and provider versions, and where the state lives.
#
# The state holds the private key the fake Domino uses to reach the pool machines
# (tls_private_key), so it lives in a private GCS bucket, named at init time:
#   terraform init -backend-config="bucket=<state bucket>" -backend-config="prefix=rag-emulation"
# (emulation/cloudshell/setup.sh creates that bucket).

terraform {
  # optional() object attributes with defaults need 1.3; 1.5 is a conservative floor.
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 6.0, < 8.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }

  backend "gcs" {}
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}
