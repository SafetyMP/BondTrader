#!/bin/bash
# Setup script for monitoring infrastructure

set -e

echo "🔧 Setting up BondTrader Monitoring Infrastructure"
echo "=================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Navigate to monitoring directory
MONITORING_DIR="docker/monitoring"
if [ ! -d "$MONITORING_DIR" ]; then
    echo "❌ Monitoring directory not found: $MONITORING_DIR"
    exit 1
fi

cd "$MONITORING_DIR"

# Create Grafana directories if they don't exist
mkdir -p grafana/provisioning/datasources
mkdir -p grafana/provisioning/dashboards
mkdir -p grafana/dashboards

# Create Prometheus datasource configuration
cat > grafana/provisioning/datasources/prometheus.yml <<EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

echo "✅ Created Grafana datasource configuration"

# Create dashboard provisioning
cat > grafana/provisioning/dashboards/dashboard.yml <<EOF
apiVersion: 1

providers:
  - name: 'BondTrader'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

echo "✅ Created Grafana dashboard provisioning"

# Set Grafana password
GRAFANA_PASSWORD=${GRAFANA_PASSWORD:-admin}
export GRAFANA_PASSWORD

echo ""
echo "📊 Starting monitoring stack..."
echo "   Grafana password: $GRAFANA_PASSWORD"
echo ""

# Start the stack
docker-compose -f docker-compose.monitoring.yml up -d

echo ""
echo "✅ Monitoring stack started!"
echo ""
echo "📊 Access the services:"
echo "   Prometheus: http://localhost:9090"
echo "   Grafana:    http://localhost:3000 (admin/$GRAFANA_PASSWORD)"
echo "   Node Exporter: http://localhost:9100/metrics"
echo ""
echo "To stop the stack:"
echo "   cd $MONITORING_DIR && docker-compose -f docker-compose.monitoring.yml down"
echo ""
echo "To view logs:"
echo "   cd $MONITORING_DIR && docker-compose -f docker-compose.monitoring.yml logs -f"
