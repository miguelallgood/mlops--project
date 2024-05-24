variable "mlflow-image" {
  description = "mlflow-image"
  type        = string
}

variable "port" {
  description = "Port on which the container will run"
  type        = number
  default     = 5000
}

variable "docker_registry" {
  description = "Optional Docker registry for image storage (defaults to Docker Hub)"
  type        = string
  default     = ""
}