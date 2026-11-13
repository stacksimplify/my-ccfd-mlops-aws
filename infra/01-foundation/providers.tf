# One Region for the whole course, and the 4 course tags on every resource this layer makes.
provider "aws" {
  region = "us-east-1"

  default_tags {
    tags = {
      course  = "mlops-on-aws"
      owner   = var.owner
      purpose = var.purpose
      created = var.created
    }
  }
}
