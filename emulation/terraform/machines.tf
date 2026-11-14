# The machines: the fake Domino with its Workspace, Elasticsearch, and one pool
# of machines per hardware tier. None has a public address, and each deletes
# itself after var.max_run_hours (scheduling below). Terraform never creates an
# App: rag-deploy publishes them through the fake Domino, which places each on a
# pool machine of its hardware tier.
#
# Startup scripts go in metadata["startup-script"] (updated in place, re-run at the
# next boot) rather than metadata_startup_script (which re-creates the machine on
# any change). Each is scripts/common.sh (shared functions, no interpolation) followed by the role's.

locals {
  common_script = file("${path.module}/scripts/common.sh")

  # What the fake Domino reads from --placement (the contract with rag-domino-emulation).
  placement = {
    ssh      = { user = local.ssh_user, key = local.ssh_key }
    app_port = local.app_port
    tiers = [
      for tier_id, tier in var.hardware_tiers : {
        id         = tier_id
        cores      = tonumber(regex("-([0-9]+)$", tier.machine_type)[0])
        memory_gib = tonumber(regex("-([0-9]+)$", tier.machine_type)[0]) * local.gib_per_core[regex("^n2d?-([a-z]+)-", tier.machine_type)[0]]
        default    = tier.default
        machines = [
          for n in range(1, tier.count + 1) :
          google_compute_instance.pool["${tier_id}-${n}"].network_interface[0].network_ip
        ]
      }
    ]
  }

  workspace_address     = google_compute_address.workspace.address
  elasticsearch_address = google_compute_address.elasticsearch.address
}

resource "google_compute_instance" "elasticsearch" {
  name                      = "${var.name}-es"
  machine_type              = var.es_machine_type
  zone                      = var.zone
  tags                      = [local.tags.elasticsearch]
  labels                    = local.labels
  allow_stopping_for_update = true

  boot_disk {
    initialize_params {
      image  = var.image
      size   = var.es_disk_gb
      type   = "pd-balanced"
      labels = local.labels
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.emulation.id
    network_ip = local.elasticsearch_address
  }

  service_account {
    email  = google_service_account.elasticsearch.email
    scopes = ["cloud-platform"]
  }

  scheduling {
    provisioning_model          = "STANDARD"
    instance_termination_action = "DELETE"
    max_run_duration {
      seconds = local.max_run_seconds
    }
  }

  metadata = {
    enable-oslogin          = "FALSE"
    enable-guest-attributes = "TRUE"
    startup-script = join("\n", [
      local.common_script,
      templatefile("${path.module}/scripts/elasticsearch.sh.tftpl", {
        es_image = var.es_image
        es_heap  = var.es_heap
      }),
    ])
  }

  depends_on = [google_compute_router_nat.emulation]
}

resource "google_compute_instance" "pool" {
  for_each = local.pool_machines

  name                      = "${var.name}-${each.key}"
  machine_type              = each.value.machine_type
  zone                      = var.zone
  tags                      = [local.tags.pool]
  labels                    = merge(local.labels, { "hardware-tier" = each.value.tier })
  allow_stopping_for_update = true

  boot_disk {
    initialize_params {
      image  = var.image
      size   = each.value.disk_gb
      type   = "pd-balanced"
      labels = local.labels
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.emulation.id
  }

  service_account {
    email  = google_service_account.pool.email
    scopes = ["cloud-platform"]
  }

  scheduling {
    provisioning_model          = "STANDARD"
    instance_termination_action = "DELETE"
    max_run_duration {
      seconds = local.max_run_seconds
    }
  }

  metadata = {
    # Instance-level keys only work without OS Login; the fake logs in as ubuntu.
    enable-oslogin          = "FALSE"
    enable-guest-attributes = "TRUE"
    ssh-keys                = "${local.ssh_user}:${trimspace(tls_private_key.fake_domino.public_key_openssh)}"
    startup-script = join("\n", [
      local.common_script,
      templatefile("${path.module}/scripts/pool.sh.tftpl", {
        repo_url          = var.repo_url
        git_ref           = var.git_ref
        repo_path         = var.repo_path
        uv_version        = var.uv_version
        ovms_bundle_path  = var.ovms_bundle_path
        workspace_address = local.workspace_address
        domino_port       = local.domino_port
      }),
    ])
  }

  depends_on = [google_compute_router_nat.emulation]
}

resource "google_compute_instance" "workspace" {
  name                      = "${var.name}-workspace"
  machine_type              = var.workspace_machine_type
  zone                      = var.zone
  tags                      = [local.tags.workspace]
  labels                    = local.labels
  allow_stopping_for_update = true

  boot_disk {
    initialize_params {
      image  = var.image
      size   = var.workspace_disk_gb
      type   = "pd-balanced"
      labels = local.labels
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.emulation.id
    network_ip = local.workspace_address
  }

  service_account {
    email  = google_service_account.workspace.email
    scopes = ["cloud-platform"]
  }

  scheduling {
    provisioning_model          = "STANDARD"
    instance_termination_action = "DELETE"
    max_run_duration {
      seconds = local.max_run_seconds
    }
  }

  metadata = {
    # `gcloud compute ssh ubuntu@...` then logs in as the user the setup is for.
    enable-oslogin          = "FALSE"
    enable-guest-attributes = "TRUE"
    startup-script = join("\n", [
      local.common_script,
      templatefile("${path.module}/scripts/workspace.sh.tftpl", {
        repo_url           = var.repo_url
        git_ref            = var.git_ref
        repo_path          = var.repo_path
        uv_version         = var.uv_version
        project_id         = var.project_id
        ssh_key_secret     = local.ssh_key_secret
        ssh_key_path       = local.ssh_key
        project_env_secret = var.project_env_secret
        config_dir         = local.config_dir
        placement_json     = jsonencode(local.placement)
        pool_addresses     = join(" ", flatten([for tier in local.placement.tiers : tier.machines]))
        domino_port       = local.domino_port
        workspace_address  = local.workspace_address
        es_url             = "http://${local.elasticsearch_address}:9200"
        default_tier       = local.default_tier
        environment_id     = var.environment_id
        ovms_bundle_path   = var.ovms_bundle_path
        domino_project_id  = var.domino_project_id
      }),
    ])
  }

  depends_on = [
    google_compute_router_nat.emulation,
    google_secret_manager_secret_version.ssh_key,
    google_secret_manager_secret_iam_member.workspace_reads_ssh_key,
    google_secret_manager_secret_iam_member.workspace_reads_project_env,
  ]
}
