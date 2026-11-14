# How to reach the emulation once applied (all through IAP: nothing has a public
# address). `terraform output -raw ssh_workspace` prints a command to paste.

locals {
  gcloud_where = "--zone ${var.zone} --project ${var.project_id}"
}

output "workspace_name" {
  description = "The fake Domino + Workspace machine."
  value       = google_compute_instance.workspace.name
}

output "workspace_zone" {
  description = "Zone of every machine."
  value       = var.zone
}

output "workspace_address" {
  description = "The Workspace machine's internal address (deploy.yaml's domino_host)."
  value       = local.workspace_address
}

output "ssh_workspace" {
  description = "SSH to the Workspace as ubuntu, the user everything is set up for."
  value       = "gcloud compute ssh ubuntu@${google_compute_instance.workspace.name} ${local.gcloud_where} --tunnel-through-iap"
}

output "tunnel_fake_domino" {
  description = "Forward localhost:8899 to the fake Domino (Swagger at http://localhost:8899/docs)."
  value       = "gcloud compute start-iap-tunnel ${google_compute_instance.workspace.name} ${local.domino_port} --local-host-port=localhost:${local.domino_port} ${local.gcloud_where}"
}

output "tunnel_locust" {
  description = "Forward localhost:8089 to Locust's web UI on the Workspace (a run started without --headless)."
  value       = "gcloud compute start-iap-tunnel ${google_compute_instance.workspace.name} ${local.locust_port} --local-host-port=localhost:${local.locust_port} ${local.gcloud_where}"
}

output "elasticsearch_url" {
  description = "Elasticsearch, reachable from the Workspace and the pool machines only."
  value       = "http://${local.elasticsearch_address}:9200"
}

output "pool_machines" {
  description = "Each hardware tier's machines: name => internal address."
  value       = { for key, machine in google_compute_instance.pool : machine.name => machine.network_interface[0].network_ip }
}

output "readiness" {
  description = "Prints each machine's startup state (running, done or failed) from its guest attribute."
  value = join(" && ", [
    for machine in concat(
      [google_compute_instance.workspace.name, google_compute_instance.elasticsearch.name],
      [for key, pool in google_compute_instance.pool : pool.name],
    ) : "echo -n '${machine}: ' && gcloud compute instances get-guest-attributes ${machine} ${local.gcloud_where} --query-path=rag-emulation/startup --format='value(value)'"
  ])
}
