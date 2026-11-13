# ccfd-mlops-aws: the project template of "MLOps on AWS"

Your own copy of the credit card fraud detection app, built in the course from the laptop to AWS. In MLOps 14 you
click **Use this template**, and the workflows in `.github/workflows/` run in your copy.

| Folder | What it holds |
|---|---|
| `ccfd-project/` | The fraud model app: the base project's code (unchanged) plus the MLOps code (serving, jobs, pipeline, release scripts) |
| `infra/` | Terraform layers 00 to 06 |
| `scripts/` | Optional helper scripts (each section shows the commands first; the scripts do the same in one go) |
| `.github/workflows/` | `ci.yml` (tests, images), `ct.yml` (start retraining), `cd.yml` (staging, approval, canary), `rollback.yml` |

The workflows sign in to AWS with OIDC (no access keys). MLOps 14 shows how to set the repository variables they need.
