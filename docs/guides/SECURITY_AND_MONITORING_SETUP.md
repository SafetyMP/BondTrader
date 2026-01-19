# Security and Monitoring Setup Guide

This guide covers the setup and configuration of the new security and monitoring features implemented based on CTO recommendations.

## 🔐 Authentication Setup

### Quick Start

1. **Set up users via environment variables:**
```bash
export USERS="admin:secure_password123,trader:trader_pass,analyst:analyst_pass"
export ENABLE_DEFAULT_ADMIN=false
```

2. **Or use a users file:**
```json
{
  "admin": {
    "password_hash": "hashed_password",
    "salt": "salt_value",
    "roles": ["admin", "user"]
  },
  "trader": {
    "password_hash": "hashed_password",
    "salt": "salt_value",
    "roles": ["user"]
  }
}
```

3. **Start the dashboard:**
```bash
streamlit run scripts/dashboard.py
```

The dashboard will now require authentication before access.

### Creating User Hashes

To create password hashes for the users file:

```python
from bondtrader.utils.auth import hash_password

hashed, salt = hash_password("your_password")
print(f"Hash: {hashed}")
print(f"Salt: {salt}")
```

### Role-Based Access

Protect specific pages with roles:

```python
from bondtrader.utils.auth import require_role

@require_role("admin")
def admin_page():
    st.write("Admin only content")
```

---

## 🚦 Rate Limiting

### Configuration

Set rate limits via environment variables:

```bash
# API rate limiting
export API_RATE_LIMIT=100          # Max 100 requests
export API_RATE_LIMIT_WINDOW=60     # Per 60 seconds

# Dashboard rate limiting
export DASHBOARD_RATE_LIMIT=200     # Max 200 requests
export DASHBOARD_RATE_LIMIT_WINDOW=60
```

### Usage in Code

```python
from bondtrader.utils.rate_limiter import rate_limit

@rate_limit(max_requests=10, window_seconds=60)
def my_api_endpoint():
    return {"data": "..."}
```

---

## 🔑 Secrets Management

### Environment Variables (Default)

```bash
export FRED_API_KEY=your_fred_key
export FINRA_API_KEY=your_finra_key
```

### Encrypted File Backend

1. **Set up encryption:**
```bash
export SECRETS_BACKEND=file
export SECRETS_FILE=.secrets.encrypted
export SECRETS_MASTER_PASSWORD=your_master_password
```

2. **Store secrets:**
```python
from bondtrader.utils.secrets import get_secrets_manager

secrets = get_secrets_manager()
secrets.set_secret("FRED_API_KEY", "your_key")
secrets.set_secret("DB_PASSWORD", "secure_password")
```

3. **Retrieve secrets:**
```python
from bondtrader.utils.secrets import get_api_key

fred_key = get_api_key("fred")
```

### AWS Secrets Manager

```bash
export SECRETS_BACKEND=aws
export AWS_SECRET_NAME=bondtrader/secrets
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

### HashiCorp Vault

```bash
export SECRETS_BACKEND=vault
export VAULT_ADDR=http://localhost:8200
export VAULT_TOKEN=your_token
export VAULT_SECRET_PATH=secret/bondtrader
```

---

## 📊 Monitoring Setup

### Enable Metrics

```bash
export ENABLE_METRICS=true
export METRICS_PORT=8001
```

### Prometheus + Grafana Stack

1. **Start monitoring stack:**
```bash
cd docker/monitoring
docker-compose -f docker-compose.monitoring.yml up -d
```

2. **Access services:**
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

3. **Configure Grafana:**
   - Add Prometheus data source: `http://prometheus:9090`
   - Import dashboards from `docker/monitoring/grafana/dashboards/`

### Available Metrics

- `bondtrader_api_requests_total` - Total API requests
- `bondtrader_api_request_duration_seconds` - API latency
- `bondtrader_valuations_total` - Bond valuations
- `bondtrader_ml_predictions_total` - ML predictions
- `bondtrader_risk_calculations_total` - Risk calculations
- `bondtrader_cache_hits_total` - Cache hits
- `bondtrader_cache_misses_total` - Cache misses

### Using Metrics in Code

```python
from bondtrader.utils.monitoring import track_api_request, track_valuation

@track_api_request("GET", "/api/bonds")
def get_bonds():
    ...

@track_valuation("CORPORATE")
def calculate_value(bond):
    ...
```

---

## 🔍 Dependency Vulnerability Scanning

### CI/CD Integration

Vulnerability scanning runs automatically in CI/CD. To run locally:

```bash
# Install tools
pip install safety pip-audit

# Run Safety (known vulnerabilities)
safety check

# Run pip-audit (dependency scanning)
pip-audit
```

### Fixing Vulnerabilities

1. Check the vulnerability report
2. Update affected packages in `requirements.txt`
3. Test the updates
4. Commit the fixes

---

## 🧪 Testing

### Test Authentication

```python
from bondtrader.utils.auth import UserManager

manager = UserManager()
assert manager.authenticate("admin", "password")
assert manager.has_role("admin", "admin")
```

### Test Rate Limiting

```python
from bondtrader.utils.rate_limiter import RateLimiter

limiter = RateLimiter(max_requests=5, time_window_seconds=60)
allowed, _ = limiter.is_allowed("user1")
assert allowed
```

### Test Secrets Management

```python
from bondtrader.utils.secrets import SecretsManager

secrets = SecretsManager(backend="file")
secrets.set_secret("TEST_KEY", "test_value")
assert secrets.get_secret("TEST_KEY") == "test_value"
```

---

## 🚨 Security Best Practices

1. **Never commit secrets to git**
   - Use `.gitignore` for `.secrets.encrypted`
   - Use environment variables in CI/CD
   - Use secrets management in production

2. **Use strong passwords**
   - Minimum 12 characters
   - Mix of letters, numbers, symbols
   - Different passwords for different environments

3. **Rotate credentials regularly**
   - Update API keys quarterly
   - Rotate master passwords annually
   - Review user access monthly

4. **Monitor access**
   - Review authentication logs
   - Set up alerts for failed logins
   - Track rate limit violations

5. **Keep dependencies updated**
   - Run `safety check` regularly
   - Update packages monthly
   - Review security advisories

---

## 📝 Configuration Checklist

- [ ] Authentication configured (users or users file)
- [ ] Rate limits set appropriately
- [ ] Secrets management backend chosen
- [ ] Secrets migrated to secure storage
- [ ] Monitoring enabled and configured
- [ ] Prometheus/Grafana deployed
- [ ] Vulnerability scanning in CI/CD
- [ ] Security policies documented
- [ ] Access controls tested
- [ ] Monitoring dashboards configured

---

## 🆘 Troubleshooting

### Authentication Issues

**Problem:** Can't log in
- Check `USERS` environment variable format
- Verify password hashing if using users file
- Check logs for authentication errors

**Problem:** Default admin not working
- Set `ENABLE_DEFAULT_ADMIN=true`
- Default password: `admin123` (change in production!)

### Rate Limiting Issues

**Problem:** Getting rate limit errors
- Increase limits in environment variables
- Check if per-user or global limiting
- Review rate limit logs

### Secrets Management Issues

**Problem:** Can't decrypt secrets file
- Verify `SECRETS_MASTER_PASSWORD` matches
- Check file permissions
- Ensure `cryptography` package installed

### Monitoring Issues

**Problem:** Metrics not appearing
- Check `ENABLE_METRICS=true`
- Verify `prometheus-client` installed
- Check metrics endpoint: `http://localhost:8001/metrics`

---

## 📚 Additional Resources

- [Authentication Documentation](../api/AUTH_API.md)
- [Monitoring Best Practices](../development/MONITORING_GUIDE.md)
- [Security Policy](../../SECURITY.md)
- [Implementation Summary](../../IMPLEMENTATION_SUMMARY.md)

---

**Last Updated:** Implementation Date
**Maintained By:** Development Team
