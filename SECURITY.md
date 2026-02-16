# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in Drone AI, please report it responsibly.

### How to Report

1. **Do NOT** create a public GitHub issue for security vulnerabilities
2. Email security concerns directly to: **ian@allowayllc.com**
3. Include the following in your report:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Response Time**: You will receive an acknowledgment within 48 hours
- **Updates**: We will provide status updates every 5 business days
- **Resolution**: Critical vulnerabilities will be addressed within 7 days
- **Credit**: Security researchers will be credited in release notes (unless anonymity is requested)

## Security Considerations for Drone Operations

### Physical Safety

This software controls physical vehicles. Security vulnerabilities could result in:

- Uncontrolled flight behavior
- Collision with obstacles or people
- Unauthorized access to drone controls
- GPS spoofing attacks
- Signal jamming vulnerabilities

### Secure Deployment Checklist

- [ ] Use encrypted communication channels (TLS/SSL)
- [ ] Implement authentication for MAVLink connections
- [ ] Enable geofencing to prevent unauthorized flight zones
- [ ] Use secure boot on companion computers
- [ ] Regularly update firmware and dependencies
- [ ] Monitor for anomalous behavior patterns
- [ ] Implement fail-safe return-to-home procedures

### Known Security Considerations

1. **MAVLink Protocol**: Default MAVLink v1 is unencrypted. Use MAVLink v2 with signing for production deployments.

2. **Telemetry Data**: Flight logs may contain sensitive location data. Secure storage and transmission appropriately.

3. **ML Model Integrity**: Verify model checksums before deployment to prevent adversarial model injection.

4. **Sensor Spoofing**: GPS and other sensors can be spoofed. Implement sensor fusion and anomaly detection.

## Responsible Disclosure

We follow responsible disclosure practices:

1. Reporter notifies us of vulnerability
2. We acknowledge and begin investigation
3. We develop and test a fix
4. We release the fix and notify users
5. After 90 days (or upon fix release), details may be published

## Security Updates

Security updates will be announced via:

- GitHub Security Advisories
- Release notes
- Email to registered users (for critical vulnerabilities)

## Contact

- Security Email: ian@allowayllc.com
- General Contact: [@ianallowayxyz](https://x.com/ianallowayxyz)
