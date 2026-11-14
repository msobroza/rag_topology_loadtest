# The cache bucket: what outlives the machines, so that a later sitting starts in
# minutes instead of re-exporting the models and re-embedding the corpus. It holds
# the exported models (models/), Elasticsearch's index snapshots (elasticsearch/)
# and the result archives (results/); emulation/machine/ writes them, the startup
# scripts restore the models. force_destroy: `destroy` removes the emulation
# entirely, cache included.

resource "google_storage_bucket" "cache" {
  name                        = local.cache_bucket
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = true
  labels                      = local.labels

  depends_on = [google_project_service.apis]
}

# Every machine reads and writes the cache with its own service account, on this
# bucket only: still no project-wide role (iam.tf).
resource "google_storage_bucket_iam_member" "cache_users" {
  for_each = {
    workspace     = google_service_account.workspace.email
    pool          = google_service_account.pool.email
    elasticsearch = google_service_account.elasticsearch.email
  }

  bucket = google_storage_bucket.cache.name
  role   = "roles/storage.objectUser"
  member = "serviceAccount:${each.value}"
}

# Elastic asks for "Writer" access to a snapshot bucket, which also reads the
# bucket's own metadata (storage.buckets.get); objectUser does not include that.
resource "google_storage_bucket_iam_member" "elasticsearch_reads_cache_bucket" {
  bucket = google_storage_bucket.cache.name
  role   = "roles/storage.legacyBucketReader"
  member = "serviceAccount:${google_service_account.elasticsearch.email}"
}
