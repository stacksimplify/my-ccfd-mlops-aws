# The SageMaker execution role: SageMaker uses it to run our training and processing jobs, pipeline steps and
# endpoints. It can pull our 2 images, write logs and metrics, use the 3 course buckets and register models.
# infra/02-mlflow-app adds one more statement (MlflowAppDataPlane) so jobs can log runs to the MLflow App.
# Nothing else: no EC2 network actions (our jobs run without a VPC).

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

resource "aws_iam_role" "execution" {
  name               = local.execution_role_name
  description        = "SageMaker execution role of the ccfd MLOps course: jobs, pipeline steps and endpoints."
  assume_role_policy = data.aws_iam_policy_document.trust_sagemaker.json
}

data "aws_iam_policy_document" "execution" {
  statement {
    sid       = "EcrLogin"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "EcrPull"
    actions = [
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchCheckLayerAvailability",
    ]
    resources = [for repo in aws_ecr_repository.course : repo.arn]
  }

  statement {
    sid = "Logs"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
    ]
    resources = ["arn:aws:logs:${local.region}:${local.account_id}:log-group:/aws/sagemaker/*"]
  }

  # DescribeAlarms: an endpoint rollout with automatic rollback reads the state of its CloudWatch alarms
  statement {
    sid = "Metrics"
    actions = [
      "cloudwatch:PutMetricData",
      "cloudwatch:DescribeAlarms",
    ]
    resources = ["*"]
  }

  statement {
    sid = "Buckets"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
      "s3:GetBucketLocation",
    ]
    resources = flatten([for b in aws_s3_bucket.course : [b.arn, "${b.arn}/*"]])
  }

  # The pipeline runs its steps as this role. ListModelPackages: the evaluate step looks up the newest
  # Approved model (the champion) when no champion is given.
  statement {
    sid = "PipelineSteps"
    actions = [
      "sagemaker:CreateTrainingJob",
      "sagemaker:DescribeTrainingJob",
      "sagemaker:StopTrainingJob",
      "sagemaker:CreateProcessingJob",
      "sagemaker:DescribeProcessingJob",
      "sagemaker:StopProcessingJob",
      "sagemaker:CreateModelPackage",
      "sagemaker:CreateModelPackageGroup",
      "sagemaker:DescribeModelPackage",
      "sagemaker:DescribeModelPackageGroup",
      "sagemaker:AddTags",
      "sagemaker:ListTags",
      "sagemaker:ListModelPackages",
    ]
    resources = ["*"]
  }

  # A pipeline step hands this role to the job it starts; the role may pass only itself
  statement {
    sid       = "PassOnlyItself"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.execution.arn]
  }
}

resource "aws_iam_role_policy" "execution" {
  name   = "${local.execution_role_name}-policy"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.execution.json
}
