.PHONY: help build up down restart logs ps shell-api shell-dashboard test clean

# Docker Compose files
COMPOSE_FILE = docker/docker-compose.yml
COMPOSE_DEV_FILE = docker/docker-compose.dev.yml

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build all Docker images
	docker-compose -f $(COMPOSE_FILE) build

build-no-cache: ## Build all Docker images without cache
	docker-compose -f $(COMPOSE_FILE) build --no-cache

up: ## Start all services
	docker-compose -f $(COMPOSE_FILE) up -d

up-dev: ## Start all services in development mode
	docker-compose -f $(COMPOSE_FILE) -f $(COMPOSE_DEV_FILE) up -d

down: ## Stop all services
	docker-compose -f $(COMPOSE_FILE) down

down-v: ## Stop all services and remove volumes
	docker-compose -f $(COMPOSE_FILE) down -v

restart: ## Restart all services
	docker-compose -f $(COMPOSE_FILE) restart

logs: ## Show logs from all services
	docker-compose -f $(COMPOSE_FILE) logs -f

logs-api: ## Show API service logs
	docker-compose -f $(COMPOSE_FILE) logs -f api

logs-dashboard: ## Show dashboard service logs
	docker-compose -f $(COMPOSE_FILE) logs -f dashboard

ps: ## Show service status
	docker-compose -f $(COMPOSE_FILE) ps

shell-api: ## Open shell in API container
	docker-compose -f $(COMPOSE_FILE) exec api /bin/bash

shell-dashboard: ## Open shell in dashboard container
	docker-compose -f $(COMPOSE_FILE) exec dashboard /bin/bash

shell-db: ## Open PostgreSQL shell
	docker-compose -f $(COMPOSE_FILE) exec postgres psql -U bondtrader -d bondtrader

test: ## Run tests in API container
	docker-compose -f $(COMPOSE_FILE) run --rm api pytest tests/ -v

train-ml: ## Run ML training service
	docker-compose -f $(COMPOSE_FILE) --profile training up ml-service

train-ml-run: ## Run one-off ML training
	docker-compose -f $(COMPOSE_FILE) run --rm ml-service python scripts/train_all_models.py

clean: ## Clean up Docker resources
	docker-compose -f $(COMPOSE_FILE) down -v
	docker system prune -f

backup-db: ## Backup PostgreSQL database
	docker-compose -f $(COMPOSE_FILE) exec postgres pg_dump -U bondtrader bondtrader > backup_$(shell date +%Y%m%d_%H%M%S).sql

health: ## Check health of all services
	@echo "API Health:"
	@curl -s http://localhost:8000/health | python -m json.tool || echo "API not responding"
	@echo "\nDashboard Health:"
	@curl -s http://localhost:8501/_stcore/health || echo "Dashboard not responding"
	@echo "\nMLflow Health:"
	@curl -s http://localhost:5000/health || echo "MLflow not responding"
