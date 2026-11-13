# Read with: terraform output -raw <name>   (scripts/create-env.sh writes it into .env.mlops as MLOPS_MLFLOW_APP_ARN)

output "mlflow_app_arn" {
  description = "The MLflow App ARN (arn:aws:sagemaker:us-east-1:<account>:mlflow-app/app-<id>): the tracking URI on AWS."
  value       = aws_sagemaker_mlflow_app.course.arn
}
