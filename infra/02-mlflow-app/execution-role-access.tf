# The execution role of infra/01-foundation gets access to the MLflow App, so a training job can log its run.
# Every MLflow client that talks to an MLflow App needs sagemaker:CallMlflowAppApi on the App
# (the actions with the older sagemaker-mlflow prefix are for MLflow tracking servers, which this course does not use).
data "aws_iam_policy_document" "execution_mlflow_app" {
  statement {
    sid       = "MlflowAppDataPlane"
    actions   = ["sagemaker:CallMlflowAppApi"]
    resources = ["arn:aws:sagemaker:${local.region}:${local.account_id}:mlflow-app/*"]
  }
}

resource "aws_iam_role_policy" "execution_mlflow_app" {
  name   = "${local.execution_role_name}-mlflow-app-policy"
  role   = data.aws_iam_role.execution.name
  policy = data.aws_iam_policy_document.execution_mlflow_app.json
}
