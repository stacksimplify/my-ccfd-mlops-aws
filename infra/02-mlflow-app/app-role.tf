# The MLflow App's own service role: the App uses it to store run artifacts in the mlflow bucket and to write
# model registry and lineage entries. No ECR access: automatic model registration is off in this course.

data "aws_iam_policy_document" "trust_sagemaker" {
  statement {
    sid     = "SageMakerAssumesTheRole"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "app" {
  name               = local.app_role_name
  description        = "Service role of the ccfd-mlflow MLflow App (artifacts and registry calls)."
  assume_role_policy = data.aws_iam_policy_document.trust_sagemaker.json
}

data "aws_iam_policy_document" "app" {
  statement {
    sid = "Artifacts"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
      "s3:GetBucketLocation",
    ]
    resources = [local.mlflow_bucket_arn, "${local.mlflow_bucket_arn}/*"]
  }

  # ListArtifacts: without it the App's lineage calls end in AccessDenied
  statement {
    sid = "RegistrySyncAndLineage"
    actions = [
      "sagemaker:AddTags",
      "sagemaker:CreateModelPackageGroup",
      "sagemaker:DescribeModelPackageGroup",
      "sagemaker:CreateModelPackage",
      "sagemaker:UpdateModelPackage",
      "sagemaker:DescribeModelPackage",
      "sagemaker:CreateAction",
      "sagemaker:AddAssociation",
      "sagemaker:ListArtifacts",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "app" {
  name   = "${local.app_role_name}-policy"
  role   = aws_iam_role.app.id
  policy = data.aws_iam_policy_document.app.json
}
