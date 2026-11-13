# The random part of every course bucket name: 3 random bytes as 6 hex characters (for example 4f2a9c).
# Bucket names are global across all AWS accounts, so the suffix keeps ours unique without the account ID.
# Every later layer gets the same 6 characters as var.suffix (scripts/create-env.sh keeps it as MLOPS_SUFFIX).
resource "random_id" "suffix" {
  byte_length = 3
}

locals {
  state_bucket_name = "ccfd-mlops-tfstate-${random_id.suffix.hex}"
}

# The state bucket. No force_destroy: Terraform refuses to delete it while it still holds state files,
# so a full exit empties it first (scripts/destroy-env.sh --all).
resource "aws_s3_bucket" "tfstate" {
  bucket = local.state_bucket_name
}

# Versioning keeps every older state file, so a bad apply can be rolled back by hand.
resource "aws_s3_bucket_versioning" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Encryption at rest with S3-managed keys (SSE-S3, no extra cost).
resource "aws_s3_bucket_server_side_encryption_configuration" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# State files can hold resource details: nothing in this bucket is ever public.
resource "aws_s3_bucket_public_access_block" "tfstate" {
  bucket                  = aws_s3_bucket.tfstate.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# infra/backend.hcl: the shared backend settings for layers 01 to 06 (each layer adds its own state key).
# use_lockfile = true: Terraform locks the state with a small .tflock file next to it in S3 (no DynamoDB table).
# The file is gitignored; infra/backend.hcl.example shows its shape.
resource "local_file" "backend_hcl" {
  filename        = "${path.module}/../backend.hcl"
  file_permission = "0644"
  content         = <<-EOT
    # infra/backend.hcl: written by infra/00-state-bucket (terraform apply). Never commit this file.
    # Layers 01 to 06: terraform init -backend-config=<path to this file> (../backend.hcl from a layer folder next to it)
    bucket       = "${aws_s3_bucket.tfstate.bucket}"
    region       = "us-east-1"
    use_lockfile = true
    encrypt      = true
  EOT

  depends_on = [
    aws_s3_bucket_versioning.tfstate,
    aws_s3_bucket_server_side_encryption_configuration.tfstate,
    aws_s3_bucket_public_access_block.tfstate,
  ]
}
