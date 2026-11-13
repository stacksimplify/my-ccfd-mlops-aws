# The GitHub OIDC identity provider: one per AWS account.
#   create_github_oidc_provider = false (default): the provider already exists in the account; a data source reads
#     it, so this layer never changes or deletes it.
#   create_github_oidc_provider = true: this layer makes it (an account that has none yet); terraform destroy of
#     this layer (the full exit) deletes it with the role.
# Check first: aws iam list-open-id-connect-providers
data "aws_iam_openid_connect_provider" "github" {
  count = var.create_github_oidc_provider ? 0 : 1
  url   = "https://token.actions.githubusercontent.com"
}

# No thumbprint_list: IAM fills it in at apply time and trusts the certificate authority of GitHub itself.
resource "aws_iam_openid_connect_provider" "github" {
  count          = var.create_github_oidc_provider ? 1 : 0
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
}

locals {
  github_oidc_provider_arn = (var.create_github_oidc_provider
    ? aws_iam_openid_connect_provider.github[0].arn
  : data.aws_iam_openid_connect_provider.github[0].arn)
}
