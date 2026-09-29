# Adopt the resources created by hand in the Atlas UI (Terraform 1.5+ import blocks).
# On the first `terraform plan`, these show up as "to import", not "to create".
# They can stay here: once the resources are in the state, the blocks are no-ops.

import {
  to = mongodbatlas_advanced_cluster.main
  id = "${var.project_id}-${var.cluster_name}"
}

import {
  to = mongodbatlas_database_user.app
  id = "${var.project_id}-${var.db_username}-admin"
}

import {
  to = mongodbatlas_project_ip_access_list.anywhere
  id = "${var.project_id}-${var.allowed_cidr}"
}
