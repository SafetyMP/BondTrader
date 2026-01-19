# Docker Setup Guide

Complete guide for running BondTrader with Docker and Docker Compose.

## Overview

BondTrader is containerized into multiple microservices:

1. **API Service** - FastAPI REST API
2. **Dashboard Service** - Streamlit web interface
3. **MLflow Service** - ML experiment tracking
4. **PostgreSQL** - Database for MLflow
5. **Redis** - Caching layer
6. **ML Training Service** - On-demand model training

## Quick Start

### 1. Prerequisites

- Docker 20.10+ installed
- Docker Compose 2.0+ installed
- 4GB+ RAM available
- 10GB+ disk space

### 2. Setup Environment

```bash
# Copy example environment file
cp docker/.env.example docker/.env

# Edit with your configuration
nano docker/.env  # or use your preferred editor
```

### 3. Start Services

```bash
# Using Makefile (recommended)
make up

# Or using docker-compose directly
docker-compose -f docker/docker-compose.yml up -d
```

### 4. Verify Services

```bash
# Check service status
make ps

# Or manually
docker-compose -f docker/docker-compose.yml ps

# Check health
make health
```

## Service Details

### API Service

- **URL**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

RESTful API for bond operations. See API documentation for endpoints.

### Dashboard Service

- **URL**: http://localhost:8501

Interactive Streamlit dashboard for bond analysis and visualization.

### MLflow Service

- **URL**: http://localhost:5000

ML experiment tracking and model registry. Track model training runs and compare metrics.

### Database Services

- **PostgreSQL**: Internal (port 5432) - Used by MLflow
- **Redis**: Internal (port 6379) - Caching

## Development Mode

For development with live code reloading:

```bash
make up-dev
```

This mounts your source code as volumes so changes are reflected immediately.

## Managing Services

### View Logs

```bash
# All services
make logs

# Specific service
make logs-api
make logs-dashboard
```

### Restart Services

```bash
# All services
make restart

# Specific service
docker-compose -f docker/docker-compose.yml restart api
```

### Stop Services

```bash
make down
```

### Clean Everything

```bash
# Stop and remove containers
make clean

# WARNING: This removes all data volumes
make down-v
```

## ML Training

### Run Training Service

```bash
# Start training service
make train-ml

# Run one-off training job
make train-ml-run
```

### Access Trained Models

Trained models are stored in the `ml_models` volume and accessible to all services.

## Data Persistence

All data is persisted in Docker volumes:

- `postgres_data` - Database
- `redis_data` - Cache
- `mlflow_data` - ML artifacts
- `api_data` - Application data
- `api_models` - Trained models

Volumes persist even if containers are stopped.

### Backup

```bash
# Backup database
make backup-db

# Backup volumes (manual)
docker run --rm -v bondtrader_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz /data
```

## Scaling

Scale services horizontally:

```bash
# Scale API instances
docker-compose -f docker/docker-compose.yml up -d --scale api=3
```

## Environment Variables

Key environment variables (set in `docker/.env`):

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_PASSWORD` | Database password | `bondtrader_password` |
| `API_PORT` | API service port | `8000` |
| `DASHBOARD_PORT` | Dashboard port | `8501` |
| `MLFLOW_PORT` | MLflow port | `5000` |
| `DEFAULT_RFR` | Risk-free rate | `0.03` |
| `ML_MODEL_TYPE` | ML model type | `random_forest` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Troubleshooting

### Services Not Starting

1. Check logs: `make logs`
2. Verify ports are available: `netstat -an | grep 8000`
3. Check Docker resources: `docker system df`

### Database Connection Issues

1. Verify PostgreSQL is healthy: `docker-compose ps`
2. Check database logs: `docker-compose logs postgres`
3. Test connection: `make shell-db`

### API Not Responding

1. Check API logs: `make logs-api`
2. Verify API health: `curl http://localhost:8000/health`
3. Check service dependencies: `docker-compose ps`

### Dashboard Not Loading

1. Check dashboard logs: `make logs-dashboard`
2. Verify dashboard health: `curl http://localhost:8501/_stcore/health`
3. Check API connectivity from dashboard

## Production Deployment

For production:

1. **Change default passwords** in `.env`
2. **Enable TLS/SSL** for services
3. **Configure reverse proxy** (nginx/traefik)
4. **Set up monitoring** (Prometheus/Grafana)
5. **Configure backups** (automated)
6. **Use secrets management** for sensitive data
7. **Set resource limits** in docker-compose.yml
8. **Enable logging aggregation**

## Advanced Configuration

### Custom Network

Modify `docker-compose.yml` to use custom network:

```yaml
networks:
  bondtrader-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Resource Limits

Add resource limits to services:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

## See Also

- [Docker README](../docker/README.md) - Detailed Docker documentation
- [API Reference](../api/API_REFERENCE.md) - API documentation
- [User Guide](USER_GUIDE.md) - Application usage guide
