# Ingress rules. From outside the network only IAP's range gets in (SSH, and the
# port forwards to the fake Domino and Locust); inside, each flow is allowed
# between the machines that need it, by network tag.

resource "google_compute_firewall" "iap_ingress" {
  name          = "${var.name}-allow-iap"
  network       = google_compute_network.emulation.id
  direction     = "INGRESS"
  source_ranges = [local.iap_range]
  target_tags   = values(local.tags)

  allow {
    protocol = "tcp"
    ports    = ["22", tostring(local.domino_port), tostring(local.locust_port)]
  }
}

# The fake Domino places Apps over SSH and proxies every call to them on 8888.
resource "google_compute_firewall" "workspace_to_pool" {
  name        = "${var.name}-workspace-to-pool"
  network     = google_compute_network.emulation.id
  direction   = "INGRESS"
  source_tags = [local.tags.workspace]
  target_tags = [local.tags.pool]

  allow {
    protocol = "tcp"
    ports    = ["22", tostring(local.app_port)]
  }
}

# Apps call the Domino API and fetch their run token through the pool machine's
# forwarder, which reaches the fake on the Workspace machine.
resource "google_compute_firewall" "pool_to_workspace" {
  name        = "${var.name}-pool-to-workspace"
  network     = google_compute_network.emulation.id
  direction   = "INGRESS"
  source_tags = [local.tags.pool]
  target_tags = [local.tags.workspace]

  allow {
    protocol = "tcp"
    ports    = [tostring(local.domino_port)]
  }
}

# Ingestion (Workspace) and the workflow App (pool) query Elasticsearch.
resource "google_compute_firewall" "to_elasticsearch" {
  name        = "${var.name}-to-es"
  network     = google_compute_network.emulation.id
  direction   = "INGRESS"
  source_tags = [local.tags.workspace, local.tags.pool]
  target_tags = [local.tags.elasticsearch]

  allow {
    protocol = "tcp"
    ports    = ["9200"]
  }
}
