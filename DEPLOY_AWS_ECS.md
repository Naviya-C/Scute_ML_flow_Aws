# AWS ECS Deployment Guide (Fargate)

This guide walks through deploying the **Turtle ID Backend** container to **AWS ECS** using **AWS Fargate** and an **Application Load Balancer (ALB)**.

---

## Architecture Overview

- **Compute**: AWS ECS (Fargate - serverless container compute).
- **Registry**: AWS Elastic Container Registry (ECR).
- **Networking**: Application Load Balancer (ALB) terminates HTTPS and forwards to port `8000`.
- **Database**: PostgreSQL (Supabase / AWS RDS / Aurora) with `pgvector`.
- **Secrets Management**: AWS Secrets Manager or AWS Systems Manager (SSM) Parameter Store.
- **Logging**: Amazon CloudWatch Logs via `awslogs` driver.

---

## Step 1: Create an ECR Repository and Push Image

### 1.1 Set your AWS variables
```bash
export AWS_REGION="ap-south-1"               # Adjust to your AWS region
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export REPO_NAME="turtle-id-backend"
export IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:latest"
```

### 1.2 Create the repository in ECR
```bash
aws ecr create-repository \
  --repository-name ${REPO_NAME} \
  --region ${AWS_REGION} \
  --image-scanning-configuration scanOnPush=true
```

### 1.3 Authenticate Docker to ECR
```bash
aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
```

### 1.4 Build, Tag, and Push
From the `turtle_id_backend_aws/` root:
```bash
docker build -t ${REPO_NAME}:latest .
docker tag ${REPO_NAME}:latest ${IMAGE_URI}
docker push ${IMAGE_URI}
```

---

## Step 2: Store Secrets in AWS Secrets Manager

Store your sensitive configuration variables securely in AWS Secrets Manager:

```bash
aws secretsmanager create-secret \
  --name "turtle-id/backend-secrets" \
  --region ${AWS_REGION} \
  --secret-string '{
    "DATABASE_URL": "postgresql://postgres.<ref>:<PASSWORD>@aws-0-ap-south-1.pooler.supabase.com:6543/postgres?sslmode=require",
    "MIGRATION_DATABASE_URL": "postgresql://postgres.<ref>:<PASSWORD>@aws-0-ap-south-1.pooler.supabase.com:5432/postgres?sslmode=require",
    "API_KEYS": "your-secure-frontend-api-key-here",
    "ADMIN_KEYS": "your-secure-admin-api-key-here",
    "ALLOWED_IMAGE_HOSTS": "https://pub-xxxx.r2.dev",
    "MODAL_TOKEN_ID": "ak-xxxx",
    "MODAL_TOKEN_SECRET": "as-xxxx"
  }'
```

---

## Step 3: Run Database Migrations (One-Off Task)

Before starting the service, run database migrations. You can execute this either locally using `alembic upgrade head` or as a one-time ECS run-task command overriding the CMD:

```bash
# Locally:
export MIGRATION_DATABASE_URL="your-supabase-session-pooler-url"
alembic upgrade head
```

Or using Docker:
```bash
docker run --rm \
  -e MIGRATION_DATABASE_URL="your-supabase-session-pooler-url" \
  ${IMAGE_URI} \
  alembic upgrade head
```

---

## Step 4: ECS Task Execution Role & Task Role

Create an IAM execution role (`ecsTaskExecutionRole`) that has:
1. `AmazonECSTaskExecutionRolePolicy`
2. Custom inline policy to read secrets:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": "arn:aws:secretsmanager:*:*:secret:turtle-id/backend-secrets*"
    }
  ]
}
```

---

## Step 5: Register the ECS Task Definition

Create `task-definition.json`:

```json
{
  "family": "turtle-id-backend",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "<ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/turtle-id-backend:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        { "name": "ENV", "value": "production" },
        { "name": "PORT", "value": "8000" },
        { "name": "ENABLE_DOCS", "value": "false" }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:<REGION>:<ACCOUNT_ID>:secret:turtle-id/backend-secrets:DATABASE_URL::"
        },
        {
          "name": "API_KEYS",
          "valueFrom": "arn:aws:secretsmanager:<REGION>:<ACCOUNT_ID>:secret:turtle-id/backend-secrets:API_KEYS::"
        },
        {
          "name": "ADMIN_KEYS",
          "valueFrom": "arn:aws:secretsmanager:<REGION>:<ACCOUNT_ID>:secret:turtle-id/backend-secrets:ADMIN_KEYS::"
        },
        {
          "name": "ALLOWED_IMAGE_HOSTS",
          "valueFrom": "arn:aws:secretsmanager:<REGION>:<ACCOUNT_ID>:secret:turtle-id/backend-secrets:ALLOWED_IMAGE_HOSTS::"
        },
        {
          "name": "MODAL_TOKEN_ID",
          "valueFrom": "arn:aws:secretsmanager:<REGION>:<ACCOUNT_ID>:secret:turtle-id/backend-secrets:MODAL_TOKEN_ID::"
        },
        {
          "name": "MODAL_TOKEN_SECRET",
          "valueFrom": "arn:aws:secretsmanager:<REGION>:<ACCOUNT_ID>:secret:turtle-id/backend-secrets:MODAL_TOKEN_SECRET::"
        }
      ],
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "python -c \"import urllib.request, os; port=os.environ.get('PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health')\" || exit 1"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 15
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/turtle-id-backend",
          "awslogs-region": "<REGION>",
          "awslogs-stream-prefix": "ecs",
          "awslogs-create-group": "true"
        }
      }
    }
  ]
}
```

Register it:
```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

---

## Step 6: Configure Application Load Balancer (ALB)

1. **Target Group**:
   - Target Type: `IP` (required for Fargate).
   - Protocol / Port: `HTTP` on port `8000`.
   - Health Check Path: `/health`.
   - Healthy threshold: `2`, Unhealthy threshold: `3`.
   - Success codes: `200`.

2. **ALB Listener**:
   - Port 443 (HTTPS) with an ACM SSL certificate.
   - Forward rule to the Target Group created above.

---

## Step 7: Create the ECS Service

```bash
aws ecs create-service \
  --cluster turtle-cluster \
  --service-name turtle-id-backend-service \
  --task-definition turtle-id-backend \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxx,subnet-yyyy],securityGroups=[sg-zzzz],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:<REGION>:<ACCOUNT_ID>:targetgroup/turtle-tg/xxxx,containerName=api,containerPort=8000"
```

---

## Step 8: Verify Deployment

1. Check Target Group Health in AWS EC2 Console -> Target Groups: All targets should transition to **healthy**.
2. Send a test request to your ALB DNS name or custom domain:
   ```bash
   curl -I https://api.yourdomain.com/health
   # Returns: HTTP/2 200
   ```
3. Test ready route (verifies database connection):
   ```bash
   curl -I https://api.yourdomain.com/ready
   # Returns: HTTP/2 200
   ```
