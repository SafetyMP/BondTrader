# Tooling Quick Start Guide

Quick implementation guides for the highest-impact tools.

## 🚀 Quick Wins (Implement Today)

### 1. Dependabot (15 minutes)

Already configured! Just enable in GitHub:
1. Go to repository Settings → Security → Code security and analysis
2. Enable "Dependabot alerts" and "Dependabot security updates"
3. Dependabot will automatically create PRs for dependency updates

**File**: `.github/dependabot.yml` (already created)

---

### 2. OpenAPI Documentation Enhancement (2 hours)

Enhance existing FastAPI documentation:

```python
# scripts/api_server.py - Add to existing FastAPI app
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="BondTrader API",
        version="1.0.0",
        description="Financial bond trading and arbitrage detection API",
        routes=app.routes,
    )
    # Add custom tags, examples, etc.
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

**Export OpenAPI spec**:
```bash
curl http://localhost:8000/openapi.json > openapi.yaml
```

---

### 3. Alembic Database Migrations (4 hours)

**Step 1: Install and Initialize**
```bash
pip install alembic
alembic init alembic
```

**Step 2: Configure**
```bash
# Copy alembic.ini.example to alembic.ini
cp alembic.ini.example alembic.ini

# Edit alembic.ini - set sqlalchemy.url
# sqlalchemy.url = sqlite:///bondtrader.db
# Or for PostgreSQL:
# sqlalchemy.url = postgresql://user:pass@localhost/bondtrader
```

**Step 3: Configure env.py**
```python
# alembic/env.py
from bondtrader.data.data_persistence_enhanced import Base
target_metadata = Base.metadata
```

**Step 4: Create Initial Migration**
```bash
alembic revision --autogenerate -m "Initial schema"
alembic upgrade head
```

**Step 5: Add to CI/CD**
```yaml
# .github/workflows/ci.yml
- name: Run database migrations
  run: alembic upgrade head
```

---

### 4. Postman Collection (1 hour)

**Option A: Auto-generate from OpenAPI**
```bash
# Install Postman CLI
npm install -g postman-cli

# Convert OpenAPI to Postman
openapi2postman -s openapi.yaml -o BondTrader.postman_collection.json
```

**Option B: Export from FastAPI**
1. Start API server
2. Visit http://localhost:8000/docs
3. Click "Export" → "OpenAPI JSON"
4. Import into Postman

**File**: `postman/BondTrader.postman_collection.json`

---

## 📊 Medium Effort (This Week)

### 5. Log Aggregation with Grafana Loki (4-6 hours)

**Step 1: Add Loki to docker-compose**
```yaml
# docker/docker-compose.yml - add to existing
services:
  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    command: -config.file=/etc/loki/local-config.yaml
    networks:
      - bondtrader-network

  promtail:
    image: grafana/promtail:latest
    volumes:
      - ./logs:/var/log/bondtrader:ro
      - ./promtail-config.yml:/etc/promtail/config.yml
    command: -config.file=/etc/promtail/config.yml
    networks:
      - bondtrader-network
    depends_on:
      - loki
```

**Step 2: Configure Promtail**
```yaml
# promtail-config.yml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: bondtrader
    static_configs:
      - targets:
          - localhost
        labels:
          job: bondtrader
          __path__: /var/log/bondtrader/*.log
```

**Step 3: Add Loki datasource to Grafana**
- Go to Grafana → Configuration → Data Sources
- Add Loki datasource: `http://loki:3100`

---

### 6. Performance Testing with k6 (2-3 hours)

**Step 1: Install k6**
```bash
# macOS
brew install k6

# Linux
sudo gpg -k
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6
```

**Step 2: Create Test Script**
```javascript
// tests/performance/api_load_test.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

const errorRate = new Rate('errors');

export const options = {
  stages: [
    { duration: '2m', target: 50 },   // Ramp up to 50 users
    { duration: '5m', target: 50 },    // Stay at 50 users
    { duration: '2m', target: 100 },   // Ramp up to 100 users
    { duration: '5m', target: 100 },   // Stay at 100 users
    { duration: '2m', target: 0 },     // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],  // 95% of requests < 500ms
    errors: ['rate<0.01'],              // Error rate < 1%
  },
};

export default function () {
  const BASE_URL = __ENV.API_URL || 'http://localhost:8000';
  
  // Test bond valuation endpoint
  const response = http.get(`${BASE_URL}/api/v1/bonds`, {
    headers: { 'Content-Type': 'application/json' },
  });
  
  const success = check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
  
  errorRate.add(!success);
  sleep(1);
}
```

**Step 3: Run Tests**
```bash
k6 run tests/performance/api_load_test.js
```

**Step 4: Add to CI/CD**
```yaml
# .github/workflows/performance.yml
name: Performance Tests
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: grafana/k6-action@v0.3.0
        with:
          filename: tests/performance/api_load_test.js
```

---

## 🔧 Advanced Tools (This Month)

### 7. API Gateway with Traefik (6-8 hours)

**Step 1: Add Traefik to docker-compose**
```yaml
# docker/docker-compose.yml
services:
  traefik:
    image: traefik:v2.10
    command:
      - "--api.insecure=true"
      - "--providers.docker=true"
      - "--entrypoints.web.address=:80"
    ports:
      - "80:80"
      - "8080:8080"  # Traefik dashboard
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
    networks:
      - bondtrader-network

  api:
    # ... existing config ...
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.api.rule=Host(`api.bondtrader.local`)"
      - "traefik.http.routers.api.entrypoints=web"
      - "traefik.http.services.api.loadbalancer.server.port=8000"
```

**Step 2: Add Rate Limiting**
```yaml
# traefik.yml
http:
  middlewares:
    rate-limit:
      rateLimit:
        average: 100
        period: 1m
        burst: 50
```

---

### 8. Infrastructure as Code with Terraform (8-12 hours)

**Step 1: Create Terraform Configuration**
```hcl
# terraform/main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "docker" {}

resource "docker_network" "bondtrader" {
  name = "bondtrader-network"
}

resource "docker_image" "api" {
  name = "bondtrader-api:latest"
  build {
    context = "../"
    dockerfile = "docker/Dockerfile.api"
  }
}

resource "docker_container" "api" {
  name  = "bondtrader-api"
  image = docker_image.api.image_id
  ports {
    internal = 8000
    external = 8000
  }
  networks_advanced {
    name = docker_network.bondtrader.name
  }
}
```

**Step 2: Initialize and Apply**
```bash
cd terraform
terraform init
terraform plan
terraform apply
```

---

## 📝 Summary Checklist

### Immediate (Today)
- [x] Dependabot configured
- [ ] OpenAPI documentation enhanced
- [ ] Postman collection created

### This Week
- [ ] Alembic migrations set up
- [ ] Log aggregation (Loki) configured
- [ ] Performance tests (k6) created

### This Month
- [ ] API Gateway (Traefik) deployed
- [ ] Infrastructure as Code (Terraform) set up
- [ ] Code quality tool (SonarQube) integrated

---

## 🎯 Priority Order

1. **Dependabot** - 15 min, high impact
2. **OpenAPI Docs** - 2 hours, high impact
3. **Alembic** - 4 hours, critical impact
4. **Postman** - 1 hour, medium impact
5. **k6 Tests** - 3 hours, high impact
6. **Loki Logs** - 6 hours, high impact
7. **Traefik** - 8 hours, medium impact
8. **Terraform** - 12 hours, high impact (if using cloud)

---

**Start with the quick wins and work your way up!** 🚀
