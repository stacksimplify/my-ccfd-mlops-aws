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
