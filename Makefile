#!make
include system.env

# Docker related commands
build-fertilizer-app:
	docker build --tag ${IMAGE_NAME}:${IMAGE_TAG} --tag ${IMAGE_NAME}:latest -f ./fertilizer/.Dockerfile .

recreate-fertilizer-app:
	docker build --no-cache --tag ${IMAGE_NAME}:${IMAGE_TAG} --tag ${IMAGE_NAME}:latest -f ./fertilizer/.Dockerfile .


# Kubernetes related commands
start-data-app:
	kubectl exec -it ${DATA_IMAGE_NAME} -- /bin/bash -c "--action=sync_db"


# Docker compose define targets
all-compose: up

up:
	docker-compose -f $(COMPOSE_FILE) --env-file system.env up -d

down:
	docker-compose -f $(COMPOSE_FILE) --env-file system.env down

clean:
	docker-compose -f $(COMPOSE_FILE) --env-file system.envdown -v

re-create:
	docker-compose -f $(COMPOSE_FILE) --env-file system.env up -d --force-recreate

config:
	docker-compose -f $(COMPOSE_FILE) --env-file system.env config
# Help target
help:
	@echo "Available targets:"
	@echo "  all      Run the Docker Compose service"
	@echo "  up      Run the Docker Compose service in detached mode"
	@echo "  down    Stop and remove the Docker Compose service"
	@echo "  clean   Remove all Docker Compose containers and volumes"
	@echo "  help    Show this help message"