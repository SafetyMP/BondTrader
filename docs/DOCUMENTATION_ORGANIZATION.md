# Documentation Organization Guide

This document describes the organization of all documentation in the BondTrader project.

## Documentation Structure

```
docs/
├── README.md                    # Main documentation index
├── ORGANIZATION.md              # Codebase organization
├── LOGS_ORGANIZATION.md         # Log file organization
├── DOCUMENTATION_ORGANIZATION.md # This file
├── ARCHIVE.md                   # Archived documentation index
│
├── api/                         # API documentation
│   └── API_REFERENCE.md
│
├── guides/                      # User guides and tutorials
│   ├── QUICK_START_GUIDE.md
│   ├── USER_GUIDE.md
│   ├── TRAINING_GUIDE.md
│   ├── EVALUATION_DATASET.md
│   └── ...
│
├── development/                 # Developer documentation
│   ├── ARCHITECTURE.md
│   ├── ARCHITECTURE_V2.md
│   ├── CODEBASE_REORGANIZATION.md
│   └── ...
│
├── executive/                   # Executive summaries
│   ├── CTO_REVIEW_AND_OPTIMIZATION.md
│   ├── DEMO_ASSESSMENT.md       # CONSOLIDATED
│   ├── EXECUTIVE_DEMO_GUIDE.md
│   └── ...
│
├── analysis/                    # Technical analysis
│   ├── ML_PIPELINE_ANALYSIS.md  # CONSOLIDATED
│   ├── ML_IMPROVEMENTS_IMPLEMENTED.md
│   └── ...
│
└── demo/                        # Demo reports
    └── README.md
```

## Consolidated Documentation

### ML Pipeline Analysis
- **New File**: `docs/analysis/ML_PIPELINE_ANALYSIS.md`
- **Replaces**: 
  - `ML_PIPELINE_REVIEW.md` (configuration focus)
  - `ML_PIPELINE_CRITIQUE.md` (industry standards focus)
- **Status**: Old files can be archived but kept for reference

### Demo Assessment
- **New File**: `docs/executive/DEMO_ASSESSMENT.md`
- **Replaces**:
  - `DEMO_EXECUTION_ASSESSMENT.md`
  - `FINAL_DEMO_ASSESSMENT_AND_RECOMMENDATIONS.md`
- **Status**: Old files can be archived but kept for reference

## Root Directory Documentation

The following files in the root directory are project-level documentation:

- `README.md` - Main project documentation
- `CHANGELOG.md` - Version history
- `CONTRIBUTING.md` - Contribution guidelines
- `ROADMAP.md` - Project roadmap
- `SECURITY.md` - Security policy
- `LICENSE` - License file
- `CODE_OF_CONDUCT.md` - Code of conduct

### Files That Should Be in Docs

The following root-level files should be considered for moving to docs:

- `GIT_HISTORY_CLEANUP_GUIDE.md` → `docs/development/`
- `GITHUB_PREP_CHECKLIST.md` → `docs/development/` (already exists in docs/development/)
- `SECURITY_AUDIT_REPORT.md` → `docs/development/` or `docs/security/`
- `SECURITY_INCIDENT_RESPONSE.md` → `docs/development/` or `docs/security/`

## Documentation Standards

### File Naming
- Use descriptive names: `QUICK_START_GUIDE.md`
- Use UPPERCASE for main documents: `README.md`, `CHANGELOG.md`
- Use lowercase with underscores for technical docs: `ml_pipeline_analysis.md`
- Avoid duplicates: consolidate similar content

### Organization Principles
1. **By Audience**: Separate user, developer, executive docs
2. **By Purpose**: Guides, API, analysis, etc.
3. **Avoid Duplication**: Consolidate similar content
4. **Clear Navigation**: Use README files in each directory
5. **Regular Updates**: Keep docs in sync with code

### Documentation Types

1. **User Guides** (`docs/guides/`)
   - How-to documentation
   - Tutorials
   - Usage examples
   - Configuration guides

2. **Developer Docs** (`docs/development/`)
   - Architecture documentation
   - Development guides
   - Troubleshooting
   - Code organization

3. **API Documentation** (`docs/api/`)
   - API reference
   - Endpoint documentation
   - Authentication
   - Examples

4. **Executive Docs** (`docs/executive/`)
   - High-level summaries
   - Business value
   - Demo guides
   - CTO reviews

5. **Analysis** (`docs/analysis/`)
   - Technical analysis
   - Performance reports
   - Optimization studies
   - ML improvements

## Log Files Organization

All log files are organized in the `logs/` directory. See [LOGS_ORGANIZATION.md](LOGS_ORGANIZATION.md) for details.

**Key Points**:
- All logs in `logs/` directory (not root)
- Logs are git-ignored
- Rotation and retention configured
- Audit logs in `logs/audit/`

## Maintenance

### Regular Tasks
1. Review for duplicate content
2. Update outdated information
3. Consolidate similar documents
4. Update cross-references
5. Archive old documentation

### When Adding New Documentation
1. Check for existing similar content
2. Place in appropriate directory
3. Update relevant README files
4. Add cross-references
5. Follow naming conventions

## Related Documentation

- [Codebase Organization](ORGANIZATION.md) - Code organization
- [Logs Organization](LOGS_ORGANIZATION.md) - Log file management
- [Archive](ARCHIVE.md) - Archived documentation
- [Project History](PROJECT_HISTORY.md) - Consolidated project summaries
