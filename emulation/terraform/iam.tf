# One service account per kind of machine, with no project-wide role: only the
# Workspace's may read secrets, and only the emulation's own (see secrets.tf).

resource "google_service_account" "workspace" {
  account_id   = "${var.name}-ws"
  display_name = "${var.name} fake Domino and Workspace"

  depends_on = [google_project_service.apis]
}

resource "google_service_account" "pool" {
  account_id   = "${var.name}-pool"
  display_name = "${var.name} hardware tier pools"

  depends_on = [google_project_service.apis]
}

resource "google_service_account" "elasticsearch" {
  account_id   = "${var.name}-es"
  display_name = "${var.name} Elasticsearch"

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_iam_member" "workspace_reads_ssh_key" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.ssh_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.workspace.email}"
}

# The Project environment variables' secret is the owner's (created outside
# Terraform, it holds API keys); Terraform only lets the Workspace read it.
resource "google_secret_manager_secret_iam_member" "workspace_reads_project_env" {
  count = var.project_env_secret == "" ? 0 : 1

  project   = var.project_id
  secret_id = var.project_env_secret
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.workspace.email}"

  depends_on = [google_project_service.apis]
}
