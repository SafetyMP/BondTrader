# BondTrader Postman Collection

This directory contains Postman collections and environments for testing the BondTrader API.

## Files

- **BondTrader.postman_collection.json** - Complete API collection with all endpoints
- **environments/Development.postman_environment.json** - Development environment variables
- **environments/Production.postman_environment.json** - Production environment variables

## Importing into Postman

### Option 1: Import Collection
1. Open Postman
2. Click "Import" button
3. Select `BondTrader.postman_collection.json`
4. Click "Import"

### Option 2: Import from URL
1. Open Postman
2. Click "Import"
3. Select "Link" tab
4. Enter collection URL (if hosted)
5. Click "Continue" → "Import"

## Importing Environments

1. Open Postman
2. Click "Import" button
3. Select environment files from `environments/` directory
4. Click "Import"

## Using the Collection

### 1. Select Environment
- Choose "Development" for local testing (http://localhost:8000)
- Choose "Production" for production API (https://api.bondtrader.local)

### 2. Update Variables
- `base_url`: API base URL
- `bond_id`: Default bond ID for testing
- `api_key`: API key if authentication is enabled

### 3. Run Requests
- Start with "Health Check" to verify API is running
- Create a bond using "Create Bond"
- Use the returned bond_id for other requests

## Collection Structure

### System
- Health Check
- API Root

### Bonds
- Create Bond
- List Bonds
- Get Bond

### Valuation
- Get Bond Valuation

### Arbitrage
- Find Arbitrage Opportunities

### Machine Learning
- ML-Enhanced Prediction

### Risk Management
- Get Risk Metrics

## Auto-Generating Collection from OpenAPI

You can regenerate the collection from the OpenAPI spec:

```bash
# Export OpenAPI spec
python scripts/export_openapi.py --format json

# Convert to Postman (requires openapi2postman)
npm install -g openapi2postman
openapi2postman -s openapi.json -o postman/BondTrader.postman_collection.json
```

## Testing Workflow

1. **Health Check** → Verify API is running
2. **Create Bond** → Add a test bond
3. **Get Bond** → Verify bond was created
4. **Get Bond Valuation** → Calculate metrics
5. **Find Arbitrage Opportunities** → Discover trading opportunities
6. **ML-Enhanced Prediction** → Get ML predictions (requires trained model)
7. **Get Risk Metrics** → Calculate risk

## Environment Variables

### Development
- `base_url`: http://localhost:8000
- `bond_id`: BOND-001

### Production
- `base_url`: https://api.bondtrader.local
- `bond_id`: BOND-001
- `api_key`: (set your API key)

## Notes

- All endpoints use the `base_url` variable
- Bond IDs are stored in the `bond_id` variable
- Update variables as needed for your environment
- Some endpoints require bonds to exist first (create bonds before testing)

## Troubleshooting

**Issue**: Requests fail with connection error
- **Solution**: Verify API server is running and `base_url` is correct

**Issue**: 404 errors on bond endpoints
- **Solution**: Create a bond first using "Create Bond" endpoint

**Issue**: 503 errors on ML endpoints
- **Solution**: Train ML models first using `scripts/train_all_models.py`

---

**Last Updated**: Implementation Date  
**Maintained By**: Development Team
