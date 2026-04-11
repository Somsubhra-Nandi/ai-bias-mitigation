variable "project_id" {}
variable "region"     { default = "us-central1" }

# ── Service Accounts ─────────────────────────────────────────────────────────

resource "google_service_account" "ingestion" {
  account_id   = "sa-ingestion"
  display_name = "FairGuard — Ingestion & Validation SA"
  project      = var.project_id
  description  = "Runs Phase 1 (ingestion, hashing, profiling, schema agent, leakage gate)."
}

resource "google_service_account" "training" {
  account_id   = "sa-training"
  display_name = "FairGuard — Training & Evaluation SA"
  project      = var.project_id
  description  = "Runs Phase 2-3 inside Vertex AI CustomTrainingJob containers."
}

resource "google_service_account" "deployment" {
  account_id   = "sa-deployment"
  display_name = "FairGuard — Deployment SA"
  project      = var.project_id
  description  = "Manages Vertex AI Endpoints and canary traffic splits."
}

# ── Project-Level IAM for Ingestion SA ───────────────────────────────────────
resource "google_project_iam_member" "ingestion_vertex_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.ingestion.email}"
}

resource "google_project_iam_member" "ingestion_pubsub_publisher" {
  project = var.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${google_service_account.ingestion.email}"
}

# ── Project-Level IAM for Training SA ────────────────────────────────────────
resource "google_project_iam_member" "training_vertex_admin" {
  project = var.project_id
  role    = "roles/aiplatform.admin"
  member  = "serviceAccount:${google_service_account.training.email}"
}

resource "google_project_iam_member" "training_logging_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.training.email}"
}

resource "google_project_iam_member" "training_monitoring_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.training.email}"
}

# ── Project-Level IAM for Deployment SA ──────────────────────────────────────
resource "google_project_iam_member" "deployment_vertex_admin" {
  project = var.project_id
  role    = "roles/aiplatform.admin"
  member  = "serviceAccount:${google_service_account.deployment.email}"
}

resource "google_project_iam_member" "deployment_pubsub_publisher" {
  project = var.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${google_service_account.deployment.email}"
}

# ── Pub/Sub Topic for HITL Alerts ────────────────────────────────────────────
resource "google_pubsub_topic" "hitl_alerts" {
  name    = "fairguard-hitl-alerts"
  project = var.project_id

  labels = {
    env     = "production"
    managed = "terraform"
    purpose = "hitl-compliance-alerts"
  }
}

resource "google_pubsub_subscription" "hitl_subscription" {
  name    = "fairguard-hitl-subscription"
  topic   = google_pubsub_topic.hitl_alerts.name
  project = var.project_id

  ack_deadline_seconds       = 300
  message_retention_duration = "86400s"

  expiration_policy {
    ttl = "604800s"
  }
}

# ── Outputs ───────────────────────────────────────────────────────────────────
output "sa_ingestion_email"  { value = google_service_account.ingestion.email }
output "sa_training_email"   { value = google_service_account.training.email }
output "sa_deployment_email" { value = google_service_account.deployment.email }
output "hitl_topic_name"     { value = google_pubsub_topic.hitl_alerts.name }
