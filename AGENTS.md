# AGENTS.md - AI Assistant Instructions

This document provides guidelines and instructions for AI assistants working on the catalog-enrichment project.

## Project Overview

**Project Name:** catalog-enrichment  
**Repository:** https://gitlab-master.nvidia.com/anmartinez/catalog-enrichment.git  
**Purpose:** A system for enriching product catalog data with additional metadata, descriptions, and categorization.

### Current Status
- ⚠️ **Early Development Phase** - Project is in initial setup stage
- Core architecture and technology stack are being established
- Documentation and development practices are being defined

### Key Goals
- Enhance product catalog data with AI-powered enrichment
- Implement scalable data processing pipelines
- Ensure data quality and consistency
- Provide APIs for catalog data access and manipulation

## Build and Test Commands

### Prerequisites
Since this is an early-stage project, specific dependencies will be documented as they are added.

### Common Commands
```bash
# Clone the repository
git clone https://gitlab-master.nvidia.com/anmartinez/catalog-enrichment.git
cd catalog-enrichment

# Install dependencies (to be updated as project evolves)
# npm install / pip install -r requirements.txt / etc.

# Run tests (to be defined)
# npm test / pytest / etc.

# Build the project (to be defined)
# npm run build / make / etc.

# Start development server (to be defined)
# npm run dev / python app.py / etc.
```

**Note:** Build and test commands will be updated as the project's technology stack is finalized.

## Code Style Guidelines

### General Principles
- **Clarity over cleverness** - Write code that is easy to understand and maintain
- **Consistent formatting** - Use automated formatting tools when available
- **Meaningful names** - Use descriptive variable, function, and class names
- **Documentation** - Include docstrings and comments for complex logic

### File Organization
- Keep files focused on a single responsibility
- Use clear, descriptive file names
- Organize code into logical directories/modules
- Separate configuration from business logic

## Testing Instructions

### Testing Strategy
- **Unit Tests** - Test individual functions and components in isolation
- **Integration Tests** - Test interactions between system components
- **Data Validation Tests** - Ensure data integrity and format compliance
- **Performance Tests** - Validate system performance under load

### Test Organization
- Place tests in a dedicated `tests/` directory or alongside source code
- Mirror the source code structure in test organization
- Use descriptive test names that explain what is being tested

### Data Testing
Given the catalog enrichment focus, pay special attention to:
- Data validation and sanitization
- Schema compliance
- Data transformation accuracy
- Edge cases and malformed input handling

### Coverage Goals
- Aim for >80% code coverage on critical paths
- Prioritize testing of data processing and enrichment logic
- Include error handling and edge case scenarios

## Security Considerations

### Data Security
- **PII Protection** - Identify and protect personally identifiable information
- **Data Classification** - Understand and respect data sensitivity levels
- **Access Controls** - Implement appropriate authentication and authorization
- **Data Encryption** - Encrypt sensitive data in transit and at rest

### API Security
- Validate all input data
- Implement rate limiting to prevent abuse
- Use HTTPS for all external communications
- Log security events for monitoring

### Development Security
- Keep dependencies up to date
- Use environment variables for sensitive configuration
- Never commit credentials or secrets to version control
- Conduct security reviews for critical changes

### NVIDIA-Specific Considerations
- Follow NVIDIA security policies and guidelines
- Ensure compliance with enterprise security requirements
- Use approved tools and services where applicable

## AI Assistant Guidelines

### When Working on This Project

1. **Understand the Context**
   - This is a data-centric project focused on catalog enrichment
   - Consider data quality, performance, and scalability in all decisions
   - Be mindful of the enterprise environment (NVIDIA internal)

2. **Code Quality**
   - Always run tests before suggesting changes
   - Ensure new code follows established patterns
   - Include appropriate error handling and logging

3. **Documentation**
   - Update relevant documentation when making changes
   - Include examples in API documentation
   - Keep this AGENTS.md file current as the project evolves

4. **Communication**
   - Ask for clarification when requirements are ambiguous
   - Suggest improvements to architecture and processes
   - Flag potential security or performance concerns

5. **Incremental Development**
   - Start with simple, working solutions
   - Iterate and improve based on feedback
   - Consider backwards compatibility when making changes

---

**Last Updated:** $(date)  
**Version:** 1.0  

*This document should be updated as the project evolves and new practices are established.*
