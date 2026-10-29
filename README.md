# Sea Turtle ID Backend (AWS ECS Edition)

Production-ready FastAPI backend for the Sea Turtle Identification System, configured and optimized for deployment on **AWS ECS (Elastic Container Service / Fargate)**.

```
Frontend / Client ──► AWS ALB ──► FastAPI on ECS (Fargate) ──spawn──► Modal GPU Worker
                           │                  │                                │
                           ▼                  ▼                                ▼
                 ALB Health Check         PostgreSQL                   Object Storage
                     (/health)         (Supabase/RDS)               (Supabase Storage images)
```

---

## Directory Structure

```
turtle_id_backend_aws/
├── app/
│   ├── config.py              # Environment settings & production safety validators
│   ├── db.py                  # SQLAlchemy engine & session maker
│   ├── dispatch.py            # Async dispatcher to Modal GPU workers
│   ├── main.py                # FastAPI entry point, middleware & health routes
│   ├── models.py              # SQLAlchemy DB models (source of truth)
│   ├── routers/
│   │   ├── admin.py           # Admin management endpoints (/v1/admin)
│   │   └── public.py          # Public identification & status routes (/v1)
│   ├── schemas.py             # Pydantic request/response validation schemas
│   └── security.py            # API key authentication & rate limiting
├── migrations/
│   ├── env.py                 # Alembic migration runner with pgvector support
│   └── versions/              # Migration versions history
├── tests/                     # Pytest suite
├── alembic.ini                # Alembic database migration config
├── Dockerfile                 # Optimized, non-root Dockerfile for AWS ECS
├── .dockerignore              # Clean build context exclusion list
├── .env.example               # Environment variable reference
├── API.md                     # Endpoint documentation for clients
├── DEPLOY_AWS_ECS.md          # Step-by-step AWS ECS deployment guide
└── requirements.txt           # Python backend dependencies
```

---

## Key Features for AWS ECS

- **Non-Root Execution**: Runs as unprivileged `appuser` (UID 10001) in compliance with AWS ECS security best practices.
- **Port Adaptability**: Default port `8000` (customizable via `PORT` environment variable to match your ECS Task Definition / Target Group).
- **Graceful Shutdown**: Properly intercepts `SIGTERM` signals from AWS ECS during container deployments and scale-in events.
- **Liveness & Readiness**:
  - `/health` — Fast liveness probe (used by ALB / ECS container health checks).
  - `/ready` — Readiness probe (verifies database connectivity).
- **Built-in Docker Healthcheck**: Uses Python's native standard library (zero external tools needed) to verify container health.
- **Unbuffered Logging**: Outputs directly to `stdout`/`stderr` for AWS CloudWatch Logs (`awslogs` driver).

---

## Quickstart: Local Development

### 1. Virtual Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your connection strings and API keys:

```bash
cp .env.example .env
```

### 3. Run Database Migrations

```bash
# Uses MIGRATION_DATABASE_URL (or DATABASE_URL)
alembic upgrade head
```

### 4. Run the Dev Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Building and Running with Docker

### Build the Image:

```bash
docker build -t turtle-id-backend:latest .
```

### Run Locally:

```bash
docker run -p 8000:8000 --env-file .env turtle-id-backend:latest
```

Verify health status:
```bash
curl http://localhost:8000/health
# Response: {"ok": true}
```

---

## Deployment to AWS ECS

For a step-by-step guide on creating the ECR repository, pushing images, configuring IAM roles, task definitions, Application Load Balancers, and running database migrations in AWS, refer to:

👉 **[DEPLOY_AWS_ECS.md](DEPLOY_AWS_ECS.md)**
