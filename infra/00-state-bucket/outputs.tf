# Read with: terraform output -raw <name>   (scripts/create-env.sh writes them into .env.mlops)

output "suffix" {
  description = "The 6 hex characters in every course bucket name (MLOPS_SUFFIX); pass it to layers 01 to 06 as var.suffix."
  value       = random_id.suffix.hex
}

output "state_bucket_name" {
  description = "The S3 bucket that holds the state of layers 01 to 06."
  value       = aws_s3_bucket.tfstate.bucket
}
