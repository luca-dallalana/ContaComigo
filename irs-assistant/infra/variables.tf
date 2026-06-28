variable "aws_region" {
  description = "AWS region to deploy into"
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type (must be a GPU instance for Ollama)"
  default     = "g5.xlarge"
}

variable "key_name" {
  description = "Name of an existing AWS key pair for SSH access"
}

variable "my_cidr" {
  description = "Your public IP in CIDR notation (e.g. 1.2.3.4/32). Restricts SSH access."
}

variable "generation_model" {
  description = "Ollama model tag to use for generation (e.g. qwen2.5:32b, phi4:14b)"
  default     = "qwen2.5:32b"
}
