# Docker Configuration for BondTrader

This directory contains Docker configuration files for componentizing the BondTrader application.

## Architecture

The application is divided into the following Docker services:

1. **PostgreSQL** - Database for MLflow tracking and persistent storage
2. **Redis** - Caching layer
3. **MLflow** - ML experiment tracking server
4. **API** - FastAPI REST API service
5. **Dashboard** - Streamlit web dashboard
6. **ML Service** - ML training service (optional, on-demand)

## Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+

### Production Setup

1. **Create environment file**:
```bash
cp docker/.env.example docker/.env
# Edit docker/.env with your configuration
```

2. **Start all services**:
```bash
docker-compose -f docker/docker-compose.yml up -d
```

3. **Access services**:
- Dashboard: http://localhost:8501
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- MLflow: http://localhost:5000

4. **Check service status**:
```bash
docker-compose -f docker/docker-compose.yml ps
```

5. **View logs**:
```bash
# All services
docker-compose -f docker/docker-compose.yml logs -f

# Specific service
docker-compose -f docker/docker-compose.yml logs -f api
```

### Development Setup

For development with live code reloading:

```bash
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d
```

This mounts the source code as volumes for live updates.

## Service Details

### API Service (FastAPI)

- **Port**: 8000
- **Health**: http://localhost:8000/health
- **Docs**: http://localhost:8000/docs

Endpoints:
- `GET /bonds` - List bonds
- `POST /bonds` - Create bond
- `GET /bonds/{id}/valuation` - Get valuation
- `GET /arbitrage/opportunities` - Find arbitrage opportunities
- `GET /ml/predict/{id}` - ML prediction
- `GET /risk/{id}` - Risk metrics

### Dashboard Service (Streamlit)

- **Port**: 8501
- **URL**: http://localhost:8501

Interactive web dashboard for bond analysis.

### MLflow Service

- **Port**: 5000
- **URL**: http://localhost:5000

ML experiment tracking and model registry.

### PostgreSQL Database

- **Port**: 5432 (internal only by default)
- **Database**: bondtrader
- **User**: bondtrader
- **Password**: Set in `.env` file

### Redis Cache

- **Port**: 6379 (internal only by default)

## ML Training Service

To run ML training:

```bash
# Start with training profile
docker-compose -f docker/docker-compose.yml --profile training up ml-service

# Or run one-off training
docker-compose -f docker/docker-compose.yml run --rm ml-service python scripts/train_all_models.py
```

## Environment Variables

Create `docker/.env` file with:

```env
# Database
POSTGRES_PASSWORD=your_secure_password
POSTGRES_PORT=5432

# Redis
REDIS_PORT=6379

# MLflow
MLFLOW_PORT=5000

# API
API_PORT=8000

# Dashboard
DASHBOARD_PORT=8501

# Application
DEFAULT_RFR=0.03
ML_MODEL_TYPE=random_forest
LOG_LEVEL=INFO
```

## Data Persistence

All data is persisted in Docker volumes:
- `postgres_data` - Database files
- `redis_data` - Redis persistence
- `mlflow_data` - MLflow artifacts
- `api_data` - Application data
- `api_models` - Trained models
- `dashboard_data` - Dashboard data

To backup:
```bash
docker run --rm -v bondtrader_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz /data
```

To restore:
```bash
docker run --rm -v bondtrader_postgres_data:/data -v $(pwd):/backup alpine sh -c "cd /data && tar xzf /backup/postgres_backup.tar.gz"
```

## Scaling

To scale services:

```bash
# Scale API instances
docker-compose -f docker/docker-compose.yml up -d --scale api=3
```

## Troubleshooting

### Check service logs
```bash
docker-compose -f docker/docker-compose.yml logs [service-name]
```

### Restart a service
```bash
docker-compose -f docker/docker-compose.yml restart [service-name]
```

### Rebuild services
```bash
docker-compose -f docker/docker-compose.yml build --no-cache
```

### Clean up everything
```bash
# Stop and remove containers
docker-compose -f docker/docker-compose.yml down

# Remove volumes (WARNING: Deletes all data)
docker-compose -f docker/docker-compose.yml down -v
```

## Health Checks

All services include health checks. Check status:

```bash
docker-compose -f docker/docker-compose.yml ps
```

Healthy services show "healthy" status.

## Networking

All services communicate via the `bondtrader-network` bridge network. Services can reach each other by service name (e.g., `http://api:8000`).

## Security Notes

- Change default passwords in production
- Configure CORS appropriately in API service
- Use secrets management for sensitive data
- Enable TLS/SSL for production deployments
- Restrict network access as needed
