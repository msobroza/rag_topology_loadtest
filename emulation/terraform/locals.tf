# Derived values and the fixed numbers of the contract with the fake Domino.

locals {
  # Every resource that takes labels carries this one; test_terraform_files checks it.
  labels = merge(var.labels, { "rag-emulation" = var.name })

  # Fixed by Domino, not tunable: Apps listen on 8888, a run's sidecar (token
  # service and API proxy) on 8899, Locust's web UI on 8089.
  app_port    = 8888
  domino_port = 8899
  locust_port = 8089
  ssh_user    = "ubuntu"
  ssh_key     = "/home/ubuntu/.ssh/id_emulation"
  config_dir  = "/etc/rag-emulation"

  # Google's IAP TCP forwarding range: the only source allowed in from outside.
  iap_range = "35.235.240.0/20"

  max_run_seconds = var.max_run_hours * 3600

  # Memory per vCPU of the n2 / n2d classes the hardware_tiers validation allows.
  gib_per_core = { standard = 4, highmem = 8, highcpu = 1 }

  default_tier = one([for id, tier in var.hardware_tiers : id if tier.default])

  # One entry per pool machine, "<tier>-<n>", for for_each.
  pool_machines = merge([
    for tier_id, tier in var.hardware_tiers : {
      for n in range(1, tier.count + 1) : "${tier_id}-${n}" => {
        tier         = tier_id
        machine_type = tier.machine_type
        disk_gb      = tier.disk_gb
      }
    }
  ]...)

  tags = {
    workspace     = "${var.name}-workspace"
    pool          = "${var.name}-pool"
    elasticsearch = "${var.name}-es"
  }

  # The secrets the Workspace machine reads at boot (Secret Manager secret ids).
  ssh_key_secret = "${var.name}-ssh-key"
}
