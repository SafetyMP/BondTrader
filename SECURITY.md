# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Which versions are eligible for receiving such patches depends on the CVSS v3.0 Rating:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

- **Email**: Send details to the maintainers (see repository maintainers section)
- **Private Security Advisory**: Create a private security advisory on GitHub (if you have access)

### What to Include

When reporting a vulnerability, please include:

1. **Type of issue** (e.g., buffer overflow, SQL injection, cross-site scripting)
2. **Full paths of source file(s) related to the vulnerability**
3. **Location of the affected code** (tag/branch/commit or direct URL)
4. **Step-by-step instructions to reproduce the issue**
5. **Proof-of-concept or exploit code** (if possible)
6. **Impact of the issue**, including how an attacker might exploit the issue

### Our Commitment

- We will respond to your report within 48 hours
- We will provide a more detailed response within 7 days
- We will keep you informed of the progress towards a fix
- We will notify you when the vulnerability has been fixed

### Disclosure Policy

- When we receive a security bug report, we will assign it to a primary handler
- The handler will confirm the problem and determine the affected versions
- We will audit code to find any potential similar problems
- We will prepare fixes for all releases still under maintenance
- We will disclose the vulnerability after a patch has been released

We appreciate your efforts to responsibly disclose your findings, and will make every effort to acknowledge your contributions.

## Security Best Practices

### For Users

If you're using BondTrader in a production environment:

1. **Keep dependencies updated**: Regularly update `requirements.txt` dependencies
2. **Review configuration**: Ensure proper security settings in `config.py`
3. **Audit API keys**: Use secure storage for API keys (never commit to git)
4. **Network security**: If exposing the dashboard, use proper authentication
5. **Data validation**: Validate all input data before processing
6. **Regular backups**: Maintain backups of critical data

### For Developers

When contributing code:

1. **Input validation**: Always validate and sanitize user inputs
2. **SQL injection**: Use parameterized queries (SQLAlchemy handles this)
3. **Secrets management**: Never commit API keys or credentials
4. **Dependency updates**: Keep dependencies updated for security patches
5. **Error handling**: Don't expose sensitive information in error messages
6. **Code review**: Security-sensitive code should be reviewed carefully

## Known Security Limitations

This software is designed for **educational and demonstration purposes**. Known limitations include:

- **No authentication/authorization** for the dashboard (if deployed publicly)
- **Simulated data** - not intended for real trading decisions
- **No encryption** for data at rest (uses local SQLite)
- **Limited input validation** in some areas (improvements welcome)

For production use, implement:
- Proper authentication and authorization
- Encrypted data storage
- Comprehensive input validation
- Audit logging
- Security monitoring

## Security Updates

Security updates will be announced in:
- GitHub Security Advisories
- The CHANGELOG.md file
- Release notes on GitHub

Subscribe to repository notifications to receive security alerts.

---

**Thank you for helping keep BondTrader and its users safe!**
