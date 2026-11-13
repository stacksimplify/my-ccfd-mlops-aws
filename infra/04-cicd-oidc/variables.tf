# The tag values come from .env.mlops (MLOPS_OWNER, MLOPS_PURPOSE, MLOPS_CREATED); scripts/create-env.sh passes them in.
# There is no default on purpose: every resource must say who made it, why and when.

variable "owner" {
  description = "Tag owner: who made these resources (for example your name)."
  type        = string

  validation {
    condition     = can(regex("^[A-Za-z0-9 _.:/=+@-]{1,128}$", var.owner))
    error_message = "owner must be 1 to 128 characters: letters, digits, spaces and _ . : / = + @ -"
  }
}

variable "purpose" {
  description = "Tag purpose: why these resources exist (for example the course section)."
  type        = string

  validation {
    condition     = can(regex("^[A-Za-z0-9 _.:/=+@-]{1,128}$", var.purpose))
    error_message = "purpose must be 1 to 128 characters: letters, digits, spaces and _ . : / = + @ -"
  }
}

variable "created" {
  description = "Tag created: the date the environment was first made, YYYY-MM-DD (set once, never changed)."
  type        = string

  validation {
    condition     = can(regex("^[0-9]{4}-[0-9]{2}-[0-9]{2}$", var.created))
    error_message = "created must be a date in the form YYYY-MM-DD."
  }
}

variable "suffix" {
  description = "The 6 hex characters of your course buckets: MLOPS_SUFFIX in .env.mlops (or terraform output -raw suffix in infra/00-state-bucket)"
  type        = string

  validation {
    condition     = can(regex("^[0-9a-f]{6}$", var.suffix))
    error_message = "suffix must be the 6 lowercase hex characters that infra/00-state-bucket made (output suffix)."
  }
}

variable "create_github_oidc_provider" {
  description = "true = make the GitHub OIDC provider (a new account without one); false = use the one the account already has."
  type        = bool
  default     = false
}

# The 3 GitHub values below are needed to create or change the role (a precondition on the role checks them).
# They default to empty only so that terraform destroy runs without them: a destroy never needs them.

variable "github_repository" {
  description = "Your copy of the course template on GitHub, as <owner>/<repo>. Needed for apply; may stay empty for destroy."
  type        = string
  default     = ""

  validation {
    condition     = var.github_repository == "" || can(regex("^[A-Za-z0-9](-?[A-Za-z0-9])*/[A-Za-z0-9._-]+$", var.github_repository))
    error_message = "github_repository must look like <owner>/<repo>."
  }
}

variable "github_owner_id" {
  description = "The numeric ID of the repository owner: gh api repos/<owner>/<repo> --jq '.owner.id'. Needed for apply."
  type        = string
  default     = ""

  validation {
    condition     = var.github_owner_id == "" || can(regex("^[0-9]{1,20}$", var.github_owner_id))
    error_message = "github_owner_id must be the numeric owner ID (digits only)."
  }
}

variable "github_repository_id" {
  description = "The numeric ID of the repository: gh api repos/<owner>/<repo> --jq '.id'. Needed for apply."
  type        = string
  default     = ""

  validation {
    condition     = var.github_repository_id == "" || can(regex("^[0-9]{1,20}$", var.github_repository_id))
    error_message = "github_repository_id must be the numeric repository ID (digits only)."
  }
}
