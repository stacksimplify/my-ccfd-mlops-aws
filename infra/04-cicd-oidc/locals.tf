# Who and where: the account ID is read at plan time and never typed into any file.
data "aws_caller_identity" "current" {}

locals {
  region     = "us-east-1"
  account_id = data.aws_caller_identity.current.account_id
  sm_arn     = "arn:aws:sagemaker:${local.region}:${local.account_id}"

  role_name           = "ccfd-github-actions"
  execution_role_arn  = "arn:aws:iam::${local.account_id}:role/ccfd-sagemaker-execution-role"
  ecr_repository_arns = [for r in ["ccfd-serving", "ccfd-training"] : "arn:aws:ecr:${local.region}:${local.account_id}:repository/${r}"]
  read_bucket_arns    = [for p in ["dvc", "jobs"] : "arn:aws:s3:::ccfd-mlops-${p}-${var.suffix}"]

  # Repositories made after 2026-07-15 get GitHub OIDC subjects that carry the owner ID and the repository ID
  # (GitHub changelog of 2026-04-23). The form (the sub claim of a workflow token):
  #   repo:<owner>@<owner-id>/<repo>@<repo-id>:<context>
  # The IDs never change, so a deleted and re-made repository with the same name cannot use this role.
  # Read them with: gh api repos/<owner>/<repo> --jq '.owner.id, .id'
  # try(): with the empty values of a destroy the names are empty too (the role's precondition stops an apply).
  github_owner = try(split("/", var.github_repository)[0], "")
  github_repo  = try(split("/", var.github_repository)[1], "")
  subject_base = "repo:${local.github_owner}@${var.github_owner_id}/${local.github_repo}@${var.github_repository_id}"
  github_subjects = [
    "${local.subject_base}:ref:refs/heads/main",    # workflows on the main branch (ci.yml push, ct.yml)
    "${local.subject_base}:environment:production", # jobs with environment: production (cd.yml, rollback.yml)
  ]
}
