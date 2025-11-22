# GitHub Actions Workflows

This directory contains the automation pipelines for deploying our AWS stack.

- `deploy.yml`: Automatically deploys CloudFormation and syncs frontend to S3.

Trigger:
- Runs on every push to `main`.

Secrets required:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
