terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0.1"
    }
  }
}

provider "docker" {  
}

locals {
  tags = var.docker_registry != "" ? ["${var.docker_registry}/${var.mlflow-image}:latest"] : ["${var.mlflow-image}:latest"]
}

output "dockerfile_path" {
  value = "${path.module}/Dockerfile"
}

resource "docker_image" "mlflow-image" {
  name = local.tags[0]
  build {
    context    = "${path.module}/"
    dockerfile = "Dockerfile"
  }
}

resource "docker_container" "mlflow-container" {
  image = docker_image.mlflow-image.image_id
  name  = "mlflow-container"
  ports {
    internal = 5000
    external = 5000
  }
}
