# The role the course workflows assume with a short-lived GitHub token: no AWS key is stored in GitHub.
# Only 2 kinds of jobs of ONE repository may use it: jobs on the main branch and jobs in the environment production.
data "aws_iam_policy_document" "trust_github" {
  statement {
    sid     = "GitHubActionsOfOneRepository"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [local.github_oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = local.github_subjects
    }
  }
}

resource "aws_iam_role" "github_actions" {
  name               = local.role_name
  description        = "GitHub Actions of one repository (OIDC): CI, CT, CD and rollback of the ccfd MLOps course."
  assume_role_policy = data.aws_iam_policy_document.trust_github.json

  # Checked when the role is created or changed, never on terraform destroy (which needs none of these values)
  lifecycle {
    precondition {
      condition     = var.github_repository != ""
      error_message = "Set github_repository (<owner>/<repo> of your copy of the template) to apply this layer."
    }
    precondition {
      condition     = var.github_owner_id != ""
      error_message = "Set github_owner_id to apply this layer: gh api repos/<owner>/<repo> --jq '.owner.id'"
    }
    precondition {
      condition     = var.github_repository_id != ""
      error_message = "Set github_repository_id to apply this layer: gh api repos/<owner>/<repo> --jq '.id'"
    }
  }
}

# What the 4 workflows may do, and only on ccfd- resources:
#   ci.yml        push the 2 images
#   ct.yml        start the pipeline and follow it
#   cd.yml        staging endpoint, smoke test, approve the package, canary on ccfd-prod, move @champion in MLflow
#   rollback.yml  the previous package back on ccfd-prod, @champion back
data "aws_iam_policy_document" "github_actions" {
  statement {
    sid       = "EcrLogin"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "EcrPushTwoRepos"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:CompleteLayerUpload",
      "ecr:DescribeImages",
      "ecr:GetDownloadUrlForLayer",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
    ]
    resources = local.ecr_repository_arns
  }

  statement {
    sid = "PipelineStartAndDescribe"
    actions = [
      "sagemaker:StartPipelineExecution",
      "sagemaker:DescribePipeline",
      "sagemaker:DescribePipelineExecution",
      "sagemaker:ListPipelineExecutionSteps",
    ]
    resources = [
      "${local.sm_arn}:pipeline/ccfd-*",
      "${local.sm_arn}:pipeline/ccfd-*/execution/*",
    ]
  }

  # Read a package, and approve or reject it (after the staging test and the review in GitHub)
  statement {
    sid = "ModelPackages"
    actions = [
      "sagemaker:DescribeModelPackage",
      "sagemaker:UpdateModelPackage",
    ]
    resources = ["${local.sm_arn}:model-package/ccfd-fraud-detector/*"]
  }

  # Find the newest Approved package (rollback, sync check). This call has no resource-level permission.
  statement {
    sid       = "ModelPackageList"
    actions   = ["sagemaker:ListModelPackages"]
    resources = ["*"]
  }

  # Staging and production endpoints: create, update, delete, wait for InService, tag, and the smoke test calls
  statement {
    sid = "EndpointsOnCcfd"
    actions = [
      "sagemaker:CreateEndpoint",
      "sagemaker:UpdateEndpoint",
      "sagemaker:DeleteEndpoint",
      "sagemaker:DescribeEndpoint",
      "sagemaker:CreateEndpointConfig",
      "sagemaker:DeleteEndpointConfig",
      "sagemaker:DescribeEndpointConfig",
      "sagemaker:CreateModel",
      "sagemaker:DeleteModel",
      "sagemaker:DescribeModel",
      "sagemaker:AddTags",
      "sagemaker:InvokeEndpoint",
    ]
    resources = [
      "${local.sm_arn}:endpoint/ccfd-*",
      "${local.sm_arn}:endpoint-config/ccfd-*",
      "${local.sm_arn}:model/ccfd-*",
    ]
  }

  # cd.yml writes the 2 ccfd-prod rollout alarms before a canary (put_metric_alarm is idempotent), and SageMaker's
  # auto-rollback reads them
  statement {
    sid       = "RolloutAlarms"
    actions   = ["cloudwatch:PutMetricAlarm", "cloudwatch:DescribeAlarms", "cloudwatch:TagResource"]
    resources = ["arn:aws:cloudwatch:${local.region}:${local.account_id}:alarm:ccfd-prod-rollout-*"]
  }

  # Deleting ccfd-staging lists its autoscaling targets, configs and models first (read only; these 3 list calls
  # have no resource-level permission)
  statement {
    sid = "CleanupLists"
    actions = [
      "application-autoscaling:DescribeScalableTargets",
      "sagemaker:ListEndpointConfigs",
      "sagemaker:ListModels",
    ]
    resources = ["*"]
  }

  # A new model runs as the SageMaker execution role (infra/01-foundation); this role may hand over only that one
  statement {
    sid       = "PassExecutionRole"
    actions   = ["iam:PassRole"]
    resources = [local.execution_role_arn]

    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }

  # MLflow App access (the @champion alias in ccfd-fraud-model)
  statement {
    sid       = "MlflowAppDataPlane"
    actions   = ["sagemaker:CallMlflowAppApi"]
    resources = ["${local.sm_arn}:mlflow-app/*"]
  }

  # Read the data versions (dvc bucket) and the job files (jobs bucket); no write
  statement {
    sid = "ReadDvcAndJobsBuckets"
    actions = [
      "s3:GetObject",
      "s3:ListBucket",
      "s3:GetBucketLocation",
    ]
    resources = flatten([for b in local.read_bucket_arns : [b, "${b}/*"]])
  }
}

resource "aws_iam_role_policy" "github_actions" {
  name   = "${local.role_name}-policy"
  role   = aws_iam_role.github_actions.id
  policy = data.aws_iam_policy_document.github_actions.json
}
