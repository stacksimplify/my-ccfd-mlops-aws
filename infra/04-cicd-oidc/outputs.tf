# Read with: terraform output -raw <name>

output "github_actions_role_arn" {
  description = "The role GitHub signs in as: the repository secret AWS_ROLE_ARN of your copy."
  value       = aws_iam_role.github_actions.arn
}

output "github_oidc_provider_arn" {
  description = "The GitHub OIDC provider the role trusts (made here only when create_github_oidc_provider = true)."
  value       = local.github_oidc_provider_arn
}

output "github_oidc_subjects" {
  description = "The 2 token subjects the role accepts (the main branch, the environment production)."
  value       = local.github_subjects
}
