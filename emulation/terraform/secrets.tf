# The key pair the fake Domino uses to reach the pool machines. The public half
# goes into the pool machines' ssh-keys metadata; the private half into Secret
# Manager, which the Workspace machine reads at boot, so it never sits in instance
# metadata (readable by anything on the machine). It is also in Terraform's state:
# keep the state bucket private (ADR 0001).

resource "tls_private_key" "fake_domino" {
  algorithm = "ED25519"
}

resource "google_secret_manager_secret" "ssh_key" {
  secret_id = local.ssh_key_secret
  labels    = local.labels

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "ssh_key" {
  secret      = google_secret_manager_secret.ssh_key.id
  secret_data = tls_private_key.fake_domino.private_key_openssh
}
