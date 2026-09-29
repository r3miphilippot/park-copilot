# Park Copilot data layer on MongoDB Atlas (free tier), described as code.
#
# The cluster, the application user and the network rule were first created by hand in the
# Atlas UI; imports.tf makes Terraform adopt them instead of creating new ones.

# Free shared cluster (M0): 512 MB of storage, enough for about a year of wait-time snapshots.
resource "mongodbatlas_advanced_cluster" "main" {
  project_id   = var.project_id
  name         = var.cluster_name
  cluster_type = "REPLICASET"

  replication_specs = [
    {
      region_configs = [
        {
          electable_specs = {
            instance_size = "M0"
          }
          provider_name         = "TENANT" # shared (free) tier
          backing_provider_name = var.backing_provider
          region_name           = var.region
          priority              = 7
        }
      ]
    }
  ]

  lifecycle {
    # The cluster holds the whole collected history: Terraform must never delete it, even if a
    # config change would require replacing it. Such a plan fails instead of destroying data.
    prevent_destroy = true
  }
}

# Least privilege: the app can read and write its own database, nothing else.
resource "mongodbatlas_database_user" "app" {
  project_id         = var.project_id
  username           = var.db_username
  password           = var.db_password
  auth_database_name = "admin"

  roles {
    role_name     = "readWrite"
    database_name = var.app_database
  }

  lifecycle {
    # Atlas never returns passwords, so Terraform cannot compare them: without this, every plan
    # would rotate the password and break the app until MONGODB_URI is updated everywhere.
    ignore_changes = [password]
  }
}

resource "mongodbatlas_project_ip_access_list" "anywhere" {
  project_id = var.project_id
  cidr_block = var.allowed_cidr
  comment    = "GitHub Actions + Render (no fixed outbound IP on free tiers)"
}
