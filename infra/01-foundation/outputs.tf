# Read with: terraform output -raw <name>   (scripts/create-env.sh writes them into .env.mlops)

output "dvc_bucket_name" {
  description = "The DVC remote bucket (ccfd-mlops-dvc-<suffix>)."
  value       = aws_s3_bucket.course["dvc"].bucket
}

output "mlflow_bucket_name" {
  description = "The MLflow App artifact bucket (ccfd-mlops-mlflow-<suffix>)."
  value       = aws_s3_bucket.course["mlflow"].bucket
}

output "jobs_bucket_name" {
  description = "The bucket for job inputs and outputs (ccfd-mlops-jobs-<suffix>)."
  value       = aws_s3_bucket.course["jobs"].bucket
}

output "serving_repository_url" {
  description = "Push the serving image here: <url>:sha-<7>."
  value       = aws_ecr_repository.course["ccfd-serving"].repository_url
}

output "training_repository_url" {
  description = "Push the training image here: <url>:sha-<7>."
  value       = aws_ecr_repository.course["ccfd-training"].repository_url
}

output "execution_role_arn" {
  description = "The SageMaker execution role (RoleArn of every job, pipeline and model)."
  value       = aws_iam_role.execution.arn
}
