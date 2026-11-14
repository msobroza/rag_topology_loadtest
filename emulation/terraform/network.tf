# One network, one subnet, outbound internet through Cloud NAT, and no public
# address anywhere: people come in through IAP, machines go out through NAT
# (apt, pip, the Hugging Face Hub, GitHub, Docker images, an LLM API).

resource "google_compute_network" "emulation" {
  name                    = "${var.name}-net"
  auto_create_subnetworks = false

  depends_on = [google_project_service.apis]
}

resource "google_compute_subnetwork" "emulation" {
  name          = "${var.name}-subnet"
  region        = var.region
  network       = google_compute_network.emulation.id
  ip_cidr_range = var.subnet_cidr

  # Google APIs (Secret Manager at boot) without a public address.
  private_ip_google_access = true
}

resource "google_compute_router" "emulation" {
  name    = "${var.name}-router"
  region  = var.region
  network = google_compute_network.emulation.id
}

resource "google_compute_router_nat" "emulation" {
  name                               = "${var.name}-nat"
  router                             = google_compute_router.emulation.name
  region                             = google_compute_router.emulation.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# Reserved internal addresses: the pool machines need the Workspace's address
# (their forwarder to the fake Domino) and the Workspace needs the pool's, so
# fixing these two breaks the cycle and keeps deploy.yaml valid across re-creation.
resource "google_compute_address" "workspace" {
  name         = "${var.name}-workspace"
  region       = var.region
  subnetwork   = google_compute_subnetwork.emulation.id
  address_type = "INTERNAL"
  labels       = local.labels
}

resource "google_compute_address" "elasticsearch" {
  name         = "${var.name}-es"
  region       = var.region
  subnetwork   = google_compute_subnetwork.emulation.id
  address_type = "INTERNAL"
  labels       = local.labels
}
