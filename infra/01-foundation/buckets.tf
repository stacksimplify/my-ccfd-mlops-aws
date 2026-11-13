# The 3 course buckets:
#   dvc     the DVC remote (data versions; s3://ccfd-mlops-dvc-<suffix>/dvc)
#   mlflow  the artifact store of the MLflow App (s3://ccfd-mlops-mlflow-<suffix>/mlflow)
#   jobs    inputs and outputs of SageMaker jobs, the gate holdout, the training config, capture and monitoring files
# No force_destroy: Terraform refuses to delete a bucket that still holds files, so nothing is lost by mistake.
# scripts/destroy-env.sh empties them first, then runs terraform destroy.
resource "aws_s3_bucket" "course" {
  for_each = local.bucket_names
  bucket   = each.value
}

resource "aws_s3_bucket_server_side_encryption_configuration" "course" {
  for_each = aws_s3_bucket.course
  bucket   = each.value.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "course" {
  for_each                = aws_s3_bucket.course
  bucket                  = each.value.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
