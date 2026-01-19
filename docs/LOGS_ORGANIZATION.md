# Log Files Organization

This document describes the organization and management of log files in the BondTrader system.

## Log Directory Structure

All log files are stored in the `logs/` directory at the project root:

```
logs/
├── bond_trading.log          # Main application log
├── audit.log                 # Audit trail log (in logs/audit/)
├── evaluation_run.log        # Evaluation execution logs
└── *.log                     # Other application logs
```

## Log File Configuration

### Default Configuration

Log files are configured in `bondtrader/config.py`:

```python
logs_dir: str = "logs"           # Log directory
log_file: str = "bond_trading.log"  # Main log file
log_level: str = "INFO"          # Logging level
```

### Environment Variables

You can override defaults using environment variables:

```bash
export LOGS_DIR="logs"
export LOG_FILE="bond_trading.log"
export LOG_LEVEL="INFO"
```

## Log File Types

### 1. Application Logs
- **Location**: `logs/bond_trading.log`
- **Purpose**: General application logging
- **Rotation**: Configured via logging handlers
- **Retention**: Managed by logging configuration

### 2. Audit Logs
- **Location**: `logs/audit/audit.log`
- **Purpose**: Audit trail for security and compliance
- **Format**: Structured JSON format
- **Retention**: Long-term retention recommended

### 3. Evaluation Logs
- **Location**: `logs/evaluation_run*.log`
- **Purpose**: ML model evaluation execution logs
- **Format**: Standard logging format
- **Retention**: Can be cleaned after review

### 4. Performance Logs
- **Location**: `logs/performance.log` (if enabled)
- **Purpose**: Performance metrics and profiling
- **Format**: Structured format
- **Retention**: Short-term (for analysis)

## Log Management

### Log Rotation

Log rotation is handled by the logging configuration:

```python
# Example: loguru configuration
loguru_logger.add(
    "logs/bond_trading.log",
    rotation="10 MB",      # Rotate at 10 MB
    retention="30 days",    # Keep for 30 days
    level="INFO"
)
```

### Log Cleanup

Old log files should be cleaned up periodically:

```bash
# Remove logs older than 30 days
find logs/ -name "*.log" -mtime +30 -delete
```

### Log Compression

For long-term storage, compress old logs:

```bash
# Compress logs older than 7 days
find logs/ -name "*.log" -mtime +7 -exec gzip {} \;
```

## Best Practices

1. **Never commit log files to git** - Logs are in `.gitignore`
2. **Use structured logging** - Use `bondtrader.utils.logging` for structured logs
3. **Set appropriate log levels** - Use DEBUG for development, INFO for production
4. **Rotate logs regularly** - Prevent disk space issues
5. **Monitor log sizes** - Set up alerts for large log files
6. **Archive important logs** - Keep audit logs for compliance

## Log File Locations in Code

### Main Application Logging
- **Module**: `bondtrader/utils/utils.py`
- **Configuration**: `bondtrader/config.py`
- **Default**: `logs/bond_trading.log`

### Audit Logging
- **Module**: `bondtrader/core/audit.py`
- **Location**: `logs/audit/audit.log`
- **Format**: Structured JSON

### Structured Logging
- **Module**: `bondtrader/utils/logging.py`
- **Features**: Correlation IDs, context, external library support

## Troubleshooting

### Log Files Not Created

1. Check directory permissions:
   ```bash
   ls -la logs/
   ```

2. Verify configuration:
   ```python
   from bondtrader.config import get_config
   config = get_config()
   print(config.logs_dir, config.log_file)
   ```

3. Check logging setup:
   ```python
   from bondtrader.utils.utils import logger
   logger.info("Test log message")
   ```

### Log Files in Wrong Location

If log files appear in the root directory instead of `logs/`:

1. Check configuration is using `get_config()`
2. Verify `logs_dir` is set correctly
3. Ensure `logs/` directory exists

### Large Log Files

1. Enable log rotation
2. Reduce log level (INFO → WARNING)
3. Clean up old logs
4. Compress archived logs

## Migration Notes

**Previous Behavior**: Log files were sometimes created in the project root directory.

**Current Behavior**: All log files are created in the `logs/` directory.

**Action Required**: If you have log files in the root directory, move them to `logs/`:

```bash
mkdir -p logs
mv *.log logs/ 2>/dev/null
```

## Related Documentation

- [Configuration Guide](guides/QUICK_START_GUIDE.md#configuration) - Configuration setup
- [Logging Module](../bondtrader/utils/logging.py) - Logging implementation
- [Security Policy](../SECURITY.md) - Security and audit logging
