# The Google Cloud APIs this configuration calls. Left enabled on destroy: other
# work in the project may use them too.

locals {
  apis = [
    "compute.googleapis.com",       # network, NAT, firewall, machines, addresses
    "iap.googleapis.com",           # SSH and port forwards without public addresses
    "iam.googleapis.com",           # the machines' service accounts
    "secretmanager.googleapis.com", # the SSH key and the Project environment variables
  ]
}

resource "google_project_service" "apis" {
  for_each = toset(local.apis)

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}
