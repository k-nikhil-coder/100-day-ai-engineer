# Day 23: Production Containerization with Docker Compose & Health Probes

A production-grade container orchestration setup for a FastAPI machine learning inference microservice featuring native Docker healthchecks, asynchronous audit logging, and host-mounted volume persistence.


---

## 📌 Architecture Overview

Deploying ML microservices in production requires more than standalone container images. Orchestrators (such as Kubernetes, AWS ECS, or Docker Swarm) must be able to continuously verify service viability, and audit logs used for drift detection must persist across container restarts and redeployments.

```text
 Client (cURL / Frontend)
           │
           ▼ HTTP (Port 8000)
┌────────────────────────────────────────────────────────┐
│  Docker Container: housing_api_prod                    │
│                                                        │
│  ┌────────────────────────┐    ┌────────────────────┐  │
│  │   FastAPI Endpoints    │    │ Docker HEALTHCHECK │  │
│  │   • /predict (POST)    │    │ (curl /health)     │  │
│  │   • /health  (GET) ◄───┼────┼─── Every 15s       │  │
│  └───────────┬────────────┘    └────────────────────┘  │
│              │ (Non-blocking BackgroundTask)           │
│              ▼                                         │
│       Async File Logger                                │
│              │                                         │
│              ▼ Writes to /app/logs                     │
└──────────────┼─────────────────────────────────────────┘
               │ (Docker Bind Mount Volume)
               ▼
   Host System: ./logs/inference_audit.jsonl
   (Persistent MLOps Drift & Audit Stream)
