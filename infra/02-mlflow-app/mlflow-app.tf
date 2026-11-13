# The SageMaker AI MLflow App: managed MLflow tracking and model registry, with no server of ours to run
# (not the older, paid MLflow tracking server, which this course never uses).
# The App has no version, status or URL attribute in Terraform. Read them with:
#   aws sagemaker describe-mlflow-app --region us-east-1 --arn "$(terraform output -raw mlflow_app_arn)" \
#     --query '{Status:Status,Version:MlflowVersion,Mode:ModelRegistrationMode}'
# Creating the App takes about 5 minutes.
resource "aws_sagemaker_mlflow_app" "course" {
  name               = local.app_name
  artifact_store_uri = "s3://${data.aws_s3_bucket.mlflow.bucket}/mlflow"
  role_arn           = aws_iam_role.app.arn

  # MLflow registrations stay in MLflow; the SageMaker model registry (ccfd-fraud-detector) is written by our own
  # scripts only. Written out on purpose, so the setting is visible here.
  model_registration_mode = "AutoModelRegistrationDisabled"

  # IAM changes can take a while to reach every AWS service: the role policy goes first
  depends_on = [aws_iam_role_policy.app]
}
