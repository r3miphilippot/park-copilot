variable "project_id" {
  description = "Atlas project ID (Project Settings in the Atlas UI)."
  type        = string
}

variable "cluster_name" {
  description = "Name of the free M0 cluster."
  type        = string
  default     = "Cluster0"
}

variable "backing_provider" {
  description = "Cloud provider behind the shared M0 tier."
  type        = string
  default     = "AWS"
}

variable "region" {
  description = "Atlas region of the cluster (EU_WEST_3 = Paris, EU_CENTRAL_1 = Frankfurt)."
  type        = string
  default     = "EU_WEST_3"
}

variable "app_database" {
  description = "Database the application reads and writes."
  type        = string
  default     = "park_copilot"
}

variable "db_username" {
  description = "Database user of the application (collector + API)."
  type        = string
  default     = "park-copilot"
}

variable "db_password" {
  description = "Password of the database user. Pass it with TF_VAR_db_password, never in a file."
  type        = string
  sensitive   = true
}

variable "allowed_cidr" {
  description = <<-EOT
    Network allowed to reach the cluster. 0.0.0.0/0 because GitHub Actions and Render have no
    fixed outbound IP on their free tiers; access stays protected by the user's password.
  EOT
  type        = string
  default     = "0.0.0.0/0"
}
