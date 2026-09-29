output "cluster_state" {
  description = "Current state of the cluster (IDLE when ready)."
  value       = mongodbatlas_advanced_cluster.main.state_name
}

output "mongodb_srv_host" {
  description = "SRV connection string, without credentials (to build MONGODB_URI)."
  value       = mongodbatlas_advanced_cluster.main.connection_strings.standard_srv
}
