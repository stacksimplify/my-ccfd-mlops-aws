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

variable "pipeline_name" {
  description = "The retraining pipeline that the alarm starts (made by m13_01_start_pipeline.py)."
  type        = string
  default     = "ccfd-retraining-pipeline"
}

# input_data_s3_uri, data_version and gate_holdout_s3_uri are needed to create or change the rule's target
# (a precondition in loop.tf checks them). They default to empty only so that terraform destroy runs without them.

variable "input_data_s3_uri" {
  description = "Pipeline parameter InputDataS3Uri: the S3 URI of the retrain data file. Needed for apply."
  type        = string
  default     = ""

  validation {
    condition     = var.input_data_s3_uri == "" || can(regex("^s3://", var.input_data_s3_uri))
    error_message = "input_data_s3_uri must be an s3:// URI."
  }
}

variable "data_version" {
  description = "Pipeline parameter DataVersion: the label of that data (for example the DVC md5 or a git tag). Needed for apply."
  type        = string
  default     = ""
}

variable "training_config_s3_uri" {
  description = "Pipeline parameter TrainingConfigS3Uri. Empty = the pipeline's own default (the same file in the jobs bucket)."
  type        = string
  default     = ""
}

variable "gate_holdout_s3_uri" {
  description = "Pipeline parameter GateHoldoutS3Uri: the labeled month the candidate and the champion are scored on. Needed for apply."
  type        = string
  default     = ""

  validation {
    condition     = var.gate_holdout_s3_uri == "" || can(regex("^s3://", var.gate_holdout_s3_uri))
    error_message = "gate_holdout_s3_uri must be an s3:// URI."
  }
}
