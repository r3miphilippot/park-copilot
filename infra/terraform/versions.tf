terraform {
  # 1.6+ for `import` blocks whose id uses variables (imports.tf).
  required_version = ">= 1.6"

  required_providers {
    mongodbatlas = {
      source  = "mongodb/mongodbatlas"
      version = "~> 2.18"
    }
  }
}

# Credentials come from the environment, never from the code (Service Account, recommended):
#   MONGODB_ATLAS_CLIENT_ID / MONGODB_ATLAS_CLIENT_SECRET
# or a programmatic API key:
#   MONGODB_ATLAS_PUBLIC_API_KEY / MONGODB_ATLAS_PRIVATE_API_KEY
provider "mongodbatlas" {}
