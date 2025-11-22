# Infrastructure (Amanda)

This folder contains the CloudFormation/SAM templates that define our AWS infrastructure.

Current resources deployed:
- S3 static website bucket
- API Gateway
- Lambda (/hello)
- IAM roles

Next steps:
- Add DynamoDB table (game results)
- Add SNS topic
- Add additional Lambda functions (backend)
- Add API Gateway routes for game logic
- Integrate SageMaker endpoint once ready

This folder is deployed automatically by GitHub Actions.
