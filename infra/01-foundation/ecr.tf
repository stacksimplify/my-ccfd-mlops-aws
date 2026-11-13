# The 2 image repositories: ccfd-serving (Dockerfile.serving) and ccfd-training (Dockerfile.training).
# force_delete = true: terraform destroy also removes the images inside (a repository with images cannot be
# deleted otherwise). Images are always tagged sha-<7 characters of the git commit>, never latest.
resource "aws_ecr_repository" "course" {
  for_each     = local.ecr_repositories
  name         = each.value
  force_delete = true
}
