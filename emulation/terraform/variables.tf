# Every name and size of the emulation. Only project_id and git_ref are required.

variable "project_id" {
  description = "The Google Cloud project that hosts the emulation."
  type        = string

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{4,28}[a-z0-9]$", var.project_id))
    error_message = "project_id must be a Google Cloud project id (6-30 lowercase letters, digits or hyphens)."
  }
}

variable "region" {
  description = "Region of the subnet, the router and its Cloud NAT."
  type        = string
  default     = "europe-west1"
}

variable "zone" {
  description = "Zone of every machine; must be in var.region."
  type        = string
  default     = "europe-west1-b"
}

variable "name" {
  description = "Name of this emulation: the prefix of every resource and the value of the label that marks them."
  type        = string
  default     = "rag-emu"

  validation {
    # Service account ids are 6-30 characters and get a 5-character suffix ("-pool").
    condition     = can(regex("^[a-z][a-z0-9-]{2,18}[a-z0-9]$", var.name))
    error_message = "name must be 4-20 lowercase letters, digits or hyphens, starting with a letter and not ending with a hyphen."
  }
}

variable "labels" {
  description = "Extra labels on every resource that takes labels (the emulation label is always added)."
  type        = map(string)
  default     = {}
}

variable "subnet_cidr" {
  description = "Internal address range of the emulation's only subnet."
  type        = string
  default     = "10.10.0.0/24"

  validation {
    condition     = can(cidrhost(var.subnet_cidr, 0))
    error_message = "subnet_cidr must be a CIDR range such as 10.10.0.0/24."
  }
}

variable "image" {
  description = "Boot image of every machine (Ubuntu 22.04 LTS: the scripts use apt and systemd)."
  type        = string
  default     = "ubuntu-os-cloud/ubuntu-2204-lts"
}

variable "max_run_hours" {
  description = "Every machine deletes itself after this many hours of running: budget alerts do not cap Compute Engine spend."
  type        = number
  default     = 8

  validation {
    condition     = var.max_run_hours >= 1 && var.max_run_hours <= 72 && floor(var.max_run_hours) == var.max_run_hours
    error_message = "max_run_hours must be a whole number of hours from 1 to 72."
  }
}

variable "repo_url" {
  description = "The repository every machine clones (public: the machines hold no git credential)."
  type        = string
  default     = "https://github.com/msobroza/rag_topology_loadtest.git"

  validation {
    # Quoted in the startup scripts: no quote or space may appear.
    condition     = can(regex("^https://[A-Za-z0-9._~:/@-]+$", var.repo_url))
    error_message = "repo_url must be an https:// URL without quotes or spaces."
  }
}

variable "git_ref" {
  description = "The branch, tag or commit every machine checks out (one ref, so the Workspace and the Apps run the same code)."
  type        = string

  validation {
    condition     = can(regex("^[A-Za-z0-9._/-]+$", var.git_ref))
    error_message = "git_ref must be a branch, tag or commit: letters, digits, '.', '_', '/' or '-'."
  }
}

variable "repo_path" {
  description = "Absolute path of the clone on every machine: rag-deploy runs domino/app.sh by this path on each App's machine."
  type        = string
  default     = "/repos/rag_topology_loadtest"

  validation {
    condition     = can(regex("^/[A-Za-z0-9._/-]+[A-Za-z0-9._-]$", var.repo_path))
    error_message = "repo_path must be an absolute path without spaces or a trailing slash."
  }
}

variable "uv_version" {
  description = "The uv release installed on the Workspace and pool machines (Apps bootstrap with it)."
  type        = string
  default     = "0.8.0"

  validation {
    condition     = can(regex("^[0-9]+\\.[0-9]+\\.[0-9]+$", var.uv_version))
    error_message = "uv_version must be a release number such as 0.8.0."
  }
}

variable "workspace_machine_type" {
  description = "Machine type of the fake Domino + Workspace machine (Locust runs there too)."
  type        = string
  default     = "n2-standard-8"
}

variable "workspace_disk_gb" {
  description = "Boot disk of the Workspace machine, in GB (the project's virtualenv, the corpus, Locust results)."
  type        = number
  default     = 50

  validation {
    condition     = var.workspace_disk_gb >= 20
    error_message = "workspace_disk_gb must be at least 20."
  }
}

variable "es_machine_type" {
  description = "Machine type of the Elasticsearch machine."
  type        = string
  default     = "n2-standard-4"
}

variable "es_disk_gb" {
  description = "Boot disk of the Elasticsearch machine, in GB; the index lives on it."
  type        = number
  default     = 50

  validation {
    condition     = var.es_disk_gb >= 20
    error_message = "es_disk_gb must be at least 20."
  }
}

variable "es_image" {
  description = "Elasticsearch container image (the version docker-compose.yml runs locally)."
  type        = string
  default     = "docker.elastic.co/elasticsearch/elasticsearch:9.0.0"

  validation {
    condition     = can(regex("^[A-Za-z0-9._/:@-]+$", var.es_image))
    error_message = "es_image must be an image reference such as docker.elastic.co/elasticsearch/elasticsearch:9.0.0."
  }
}

variable "es_heap" {
  description = "Elasticsearch JVM heap (-Xms and -Xmx), about half the machine's memory."
  type        = string
  default     = "4g"

  validation {
    condition     = can(regex("^[0-9]+[mg]$", var.es_heap))
    error_message = "es_heap must be a JVM size such as 4g or 2048m."
  }
}

variable "hardware_tiers" {
  description = <<-EOT
    Hardware tiers: each id maps to a pool of identical machines. Exactly one tier is
    the default. machine_type is an n2/n2d standard, highmem or highcpu type (its cores
    and memory are what the fake Domino reports for the tier); an App downloads and
    converts its models on first start, so a pool machine needs 16 GiB and 50 GB.
  EOT
  type = map(object({
    machine_type = string
    count        = number
    default      = optional(bool, false)
    disk_gb      = optional(number, 50)
  }))
  default = {
    "cpu-8" = { machine_type = "n2-standard-8", count = 3, default = true }
  }

  validation {
    condition     = length([for tier in values(var.hardware_tiers) : tier if tier.default]) == 1
    error_message = "Exactly one hardware tier must have default = true."
  }

  validation {
    condition     = alltrue([for id in keys(var.hardware_tiers) : can(regex("^[a-z][a-z0-9-]{0,18}[a-z0-9]$", id))])
    error_message = "Each hardware tier id must be 2-20 lowercase letters, digits or hyphens (it is part of machine names)."
  }

  validation {
    condition     = alltrue([for tier in values(var.hardware_tiers) : can(regex("^n2d?-(standard|highmem|highcpu)-[0-9]+$", tier.machine_type))])
    error_message = "Each machine_type must be n2-/n2d- standard, highmem or highcpu (ordinary persistent disks; their memory is derived from the name)."
  }

  validation {
    condition     = alltrue([for tier in values(var.hardware_tiers) : tier.count >= 1 && tier.count <= 20 && floor(tier.count) == tier.count])
    error_message = "Each tier's count must be a whole number from 1 to 20."
  }

  validation {
    condition     = alltrue([for tier in values(var.hardware_tiers) : tier.disk_gb >= 50])
    error_message = "Each tier's disk_gb must be at least 50: an App downloads and converts its models on first start."
  }
}

variable "ovms_bundle_path" {
  description = "Where pool machines keep the OVMS binary tarball (deploy.yaml's ovms_bundle)."
  type        = string
  default     = "/opt/rag-emulation/ovms.tar.gz"

  validation {
    condition     = can(regex("^/[A-Za-z0-9._/-]+\\.tar\\.gz$", var.ovms_bundle_path))
    error_message = "ovms_bundle_path must be an absolute path ending in .tar.gz."
  }
}

variable "project_env_secret" {
  description = <<-EOT
    Optional: the id of an existing Secret Manager secret, in var.project_id, whose
    payload is KEY=VALUE lines (OPENAI_API_KEY=..., one per line): the Project
    environment variables the fake Domino hands every App and the Workspace's shell.
    "" = none (an empty file).
  EOT
  type        = string
  default     = ""

  validation {
    condition     = var.project_env_secret == "" || can(regex("^[A-Za-z0-9_-]{1,255}$", var.project_env_secret))
    error_message = "project_env_secret must be a secret id (letters, digits, '_' or '-'), not a resource path."
  }
}

variable "domino_project_id" {
  description = "DOMINO_PROJECT_ID in the Workspace's shell: the project rag-deploy publishes into."
  type        = string
  default     = "emulation"

  validation {
    condition     = can(regex("^[A-Za-z0-9._-]+$", var.domino_project_id))
    error_message = "domino_project_id must be letters, digits, '.', '_' or '-'."
  }
}

variable "environment_id" {
  description = "defaults.environment_id in the generated deploy.yaml (the fake Domino accepts any value)."
  type        = string
  default     = "emulation-environment"

  validation {
    # rag-deploy refuses an <angle-bracket> placeholder; the value is quoted in YAML.
    condition     = can(regex("^[A-Za-z0-9._-]+$", var.environment_id))
    error_message = "environment_id must be letters, digits, '.', '_' or '-'."
  }
}
