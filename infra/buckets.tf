variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

# ── Data Bucket ──────────────────────────────────────────────────────────────
resource "google_storage_bucket" "data" {
  name                        = "${var.project_id}-data"
  location                    = var.region
  project                     = var.project_id
  force_destroy               = false
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action { type = "Delete" }
    condition { age = 365 }
  }

  labels = {
    env     = "production"
    managed = "terraform"
    purpose = "fairguard-data"
  }
}

# ── Artifacts Bucket ─────────────────────────────────────────────────────────
resource "google_storage_bucket" "artifacts" {
  name                        = "${var.project_id}-artifacts"
  location                    = var.region
  project                     = var.project_id
  force_destroy               = false
  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  labels = {
    env     = "production"
    managed = "terraform"
    purpose = "fairguard-artifacts"
  }
}

# ── Pipeline Root Bucket ─────────────────────────────────────────────────────
resource "google_storage_bucket" "pipeline_root" {
  name                        = "${var.project_id}-pipeline-root"
  location                    = var.region
  project                     = var.project_id
  force_destroy               = false
  uniform_bucket_level_access = true

  labels = {
    env     = "production"
    managed = "terraform"
    purpose = "vertex-pipeline-root"
  }
}

# ── IAM Bindings ─────────────────────────────────────────────────────────────
resource "google_storage_bucket_iam_member" "ingestion_data_writer" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:sa-ingestion@${var.project_id}.iam.gserviceaccount.com"
}

resource "google_storage_bucket_iam_member" "ingestion_artifacts_reader" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:sa-ingestion@${var.project_id}.iam.gserviceaccount.com"
}

resource "google_storage_bucket_iam_member" "training_data_reader" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:sa-training@${var.project_id}.iam.gserviceaccount.com"
}

resource "google_storage_bucket_iam_member" "training_artifacts_writer" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:sa-training@${var.project_id}.iam.gserviceaccount.com"
}

resource "google_storage_bucket_iam_member" "training_pipeline_root" {
  bucket = google_storage_bucket.pipeline_root.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:sa-training@${var.project_id}.iam.gserviceaccount.com"
}

resource "google_storage_bucket_iam_member" "deployment_artifacts_reader" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:sa-deployment@${var.project_id}.iam.gserviceaccount.com"
}

# ── Outputs ───────────────────────────────────────────────────────────────────
output "data_bucket_name" {
  value = google_storage_bucket.data.name
}

output "artifacts_bucket_name" {
  value = google_storage_bucket.artifacts.name
}

output "pipeline_root_uri" {
  value = "gs://${google_storage_bucket.pipeline_root.name}"
}
