# Security Policy

## Supported Versions

We currently support the following versions of this project:

| Version | Supported          |
| ------- | ------------------ |
| 1.1.0   | :white_check_mark: |
| 1.0.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability within this project, please send an email to sksnilshub@gmail.com. All security vulnerabilities will be promptly addressed.

Please include the following information:
- Type of vulnerability
- Full paths of the source file(s) related to the vulnerability
- Location of the affected source code
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue

## Security Best Practices

When using this project, please follow these security guidelines:

### API Keys
- Never commit API keys to version control
- Use environment variables or `.env` files for sensitive credentials
- Rotate API keys regularly
- Use the minimum required permissions for API keys

### Neo4j Database
- Use strong passwords for Neo4j authentication
- Enable SSL/TLS for Neo4j connections in production
- Restrict network access to the Neo4j instance
- Regularly backup the database

### LLM API
- Monitor API usage for anomalies
- Set up usage limits to prevent unexpected charges
- Review LLM responses before acting on them

## Dependencies

This project relies on third-party dependencies. Please ensure you:
- Keep dependencies up to date
- Review dependency changes before updating
- Use dependency scanning tools to identify vulnerabilities

## Compliance

This project processes medical information. When deploying:
- Ensure compliance with HIPAA, GDPR, and other applicable regulations
- Implement proper data retention policies
- Enable audit logging
- Use encryption at rest and in transit
