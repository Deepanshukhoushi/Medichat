# MediChat - Free & Scalable Deployment Guide

This document provides comprehensive strategies to deploy MediChat for **free** (or near-free) while keeping it **scalable** and **accessible** to users.

---

## Table of Contents

1. [Architecture Overview & Dependencies](#1-architecture-overview--dependencies)
2. [Option A: Fully Free Stack (Recommended)](#2-option-a-fully-free-stack-recommended)
3. [Option B: Hybrid Free/Low-Cost Stack](#3-option-b-hybrid-freelow-cost-stack)
4. [Option C: One-Click Deploy on Render (Easiest)](#4-option-c-one-click-deploy-on-render-easiest)
5. [External Service Setup (All Options)](#5-external-service-setup-all-options)
6. [Environment Configuration](#6-environment-configuration)
7. [Scaling Strategy](#7-scaling-strategy)
8. [Cost Breakdown](#8-cost-breakdown)
9. [Recommended Approach](#9-recommended-approach)

---

## 1. Architecture Overview & Dependencies

### Runtime Dependencies

| Component | Service | Free Tier Available? |
|-----------|---------|---------------------|
| **Flask Backend** | App server | Yes |
| **Frontend (Angular)** | Static hosting / Flask-served | Yes |
| **Vector Database** | Pinecone | **No** (min $70/mo for serverless) |
| **LLM + Embeddings** | Cohere | Yes (API free tier: 100 req/min) |
| **User Auth + DB** | Supabase | Yes (500 MB DB, 50k users) |
| **Session Caching** | Redis | Yes (Upstash 10 MB free) |
| **Logging / Analytics** | Self-hosted | Yes |

### Key Bottleneck

**Pinecone** is the only paid dependency. The free tier was deprecated. Alternatives exist (see below).

---

## 2. Option A: Fully Free Stack (Recommended)

Replace Pinecone with a **free vector database** and deploy everything on free tiers.

### Updated Architecture

```
Frontend (Angular PWA)
    ↓ HTTPS
Flask Backend → Chroma (in-process) or Qdrant Cloud Free
    ↓
Supabase Free (Auth + DB + Sessions)
    ↓
Cohere API Free (LLM + Embeddings)
```

### Step 1: Replace Pinecone with Chroma (free, in-process)

Chroma runs in-process with Flask (no external service needed) - perfect for low-to-medium traffic.

#### Install Chroma

```bash
pip install chromadb
```

#### Create `app/rag/vectorstore.py`

```python
"""Drop-in replacement for Pinecone vector store using Chroma (free, in-process)."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from app.core.config.settings import get_settings

settings = get_settings()

# Cohere embedding function for Chroma
cohere_ef = embedding_functions.CohereEmbeddingFunction(
    api_key=settings.cohere_api_key,
    model_name=settings.embedding_model,  # "embed-english-v3.0"
)


class ChromaVectorStore:
    """Free, in-process vector store using ChromaDB."""

    def __init__(self, collection_name: str = "medichat") -> None:
        self.client = chromadb.PersistentClient(path="./chroma_data")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=cohere_ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add_texts(self, texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> list[str]:
        ids = [str(uuid.uuid4()) for _ in texts]
        self.collection.add(
            documents=texts,
            metadatas=metadatas or [{}] * len(texts),
            ids=ids,
        )
        return ids

    def similarity_search(self, query: str, k: int = 8) -> list[tuple[str, float, dict]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        if not results["documents"] or not results["documents"][0]:
            return []
        docs = results["documents"][0]
        metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
        distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)
        return list(zip(docs, distances, metas))

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection.name)

    @property
    def count(self) -> int:
        return self.collection.count()
```

Then update `app/rag/` to use `ChromaVectorStore` instead of Pinecone.

### Alternative: Qdrant Cloud Free (10 GB vector storage)

If you want a dedicated vector DB service (better for scaling):

```bash
pip install qdrant-client
```

Sign up at [Qdrant Cloud](https://cloud.qdrant.io) → create a free cluster (1 GB storage, no time limit).

```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

client = QdrantClient(
    url="https://your-cluster-url.qdrant.io",
    api_key="your-qdrant-api-key",
)
```

### Step 2: Deploy Backend on Render Free

[Render](https://render.com) offers a **free tier** for web services:
- 512 MB RAM, 0.1 CPU
- 750 hours/month (always-on)
- Automatic HTTPS + custom domain

#### Create `render.yaml` (infrastructure-as-code)

```yaml
services:
  - type: web
    name: medichat-backend
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn -w 2 -b 0.0.0.0:8000 app:app
    envVars:
      - key: PINECONE_API_KEY
        sync: false
      - key: COHERE_API_KEY
        sync: false
      - key: FLASK_SECRET_KEY
        generateValue: true
      - key: SUPABASE_URL
        sync: false
      - key: SUPABASE_KEY
        sync: false
      - key: PERSISTENCE_ENABLED
        value: "true"
      - key: FRONTEND_ORIGINS
        value: "https://your-frontend.onrender.com"
```

### Step 3: Deploy Frontend on Vercel/Netlify (Free)

Build the Angular app and deploy to **Vercel** (free tier: 100 GB bandwidth, unlimited sites):

```bash
cd frontend
npm install
npm run build -- --configuration production
# Output: frontend/dist/medichat-frontend/browser
```

#### Create `frontend/vercel.json`

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist/medichat-frontend/browser",
  "rewrites": [
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
```

Then connect your GitHub repo to Vercel - it auto-deploys on push.

### Step 4: Replace Redis with Upstash (Free)

[Upstash](https://upstash.com) offers **10 MB Redis** for free (enough for session caching).

```bash
pip install upstash-redis
```

```python
# In app/core/cache/session_store.py
from upstash_redis import Redis

redis = Redis(
    url="https://your-upstash-url.upstash.io",
    token="your-upstash-token",
)
```

### Step 5: Enable PWA for Offline Support

The Angular app already has a service worker configured (`ngsw-config.json`). Deploying on Vercel/Netlify with HTTPS enables PWA features:
- Add-to-homescreen
- Offline fallback
- Push notifications (future)

---

## 3. Option B: Hybrid Free/Low-Cost Stack

If you want maximum simplicity with minimal code changes:

| Service | Plan | Monthly Cost |
|---------|------|-------------|
| **Backend** | Render Free | $0 |
| **Frontend** | Vercel Free | $0 |
| **Pinecone** | Pinecone Serverless Starter | ~$70/mo *or use gcp free credits* |
| **Supabase** | Free | $0 |
| **Cohere** | Free API | $0 |
| **Domain** | `.tk` / `.ml` free Freenom domains or `your-app.vercel.app` | $0 |

> **Tip**: If you have a GCP/AWS free tier account, you may get $300-400 in credits that cover Pinecone for months.

---

## 4. Option C: One-Click Deploy on Render (Easiest)

For the absolute easiest deployment with minimal changes:

### Deploy Flask + Serve Angular from Flask

1. Build Angular: `cd frontend && npm run build -- --configuration production`
2. The Flask app already serves the Angular build from `frontend/static/` and `frontend/templates/`
3. Deploy everything as a single service on Render

**One service to manage, one URL to remember.**

```yaml
# render.yaml (monolith approach)
services:
  - type: web
    name: medichat
    env: python
    buildCommand: |
      cd frontend && npm install && npm run build
      cd .. && pip install -r requirements.txt
    startCommand: gunicorn -w 2 -b 0.0.0.0:8000 app:app
    healthCheckPath: /health
    envVars:
      - key: PINECONE_API_KEY
        sync: false
      - key: COHERE_API_KEY
        sync: false
      - key: FLASK_SECRET_KEY
        generateValue: true
      - key: SUPABASE_URL
        sync: false
      - key: SUPABASE_KEY
        sync: false
```

**Downside**: Harder to scale frontend/backend independently.

---

## 5. External Service Setup (All Options)

### A. Cohere API (Free)

1. Go to [cohere.com](https://dashboard.cohere.com/api-keys)
2. Sign up → Free tier: **100 requests/minute** (enough for a small app)
3. Copy your API key → set as `COHERE_API_KEY`

### B. Supabase (Free)

1. Go to [supabase.com](https://supabase.com)
2. Create a project
3. Get `SUPABASE_URL` and `SUPABASE_KEY` (anon/public key)
4. Run migration scripts from `scripts/` folder:
   - `scripts/profiles_migration.sql`
   - `scripts/memory_migration.sql`
   - `scripts/flashcards_migration.sql`
   - `scripts/quizzes_migration.sql`
   - `scripts/audit_migration.sql`

### C. Custom Domain (Optional, Free)

- Use **Vercel** subdomain: `medichat.vercel.app`
- Or use **Freenom** for a free `.tk` / `.ml` domain
- Or **Cloudflare Pages** + custom domain (no-cost domain through Cloudflare)

### D. CI/CD (Free)

- **GitHub Actions** (2000 min/month free)
- Auto-deploy to Render/Vercel on push to `main`

Example `.github/workflows/deploy.yml`:

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 22
      - run: cd frontend && npm install && npm run build
      # Deploy to Vercel via vercel-action

  deploy-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      # Deploy to Render via Render API or render-deploy-action
```

---

## 6. Environment Configuration

Create `.env.production`:

```env
# Required
PINECONE_API_KEY=your_key       # Or remove if using Chroma
COHERE_API_KEY=your_key
FLASK_SECRET_KEY=generate_a_random_64_char_string

# Optional (enable persistence)
PERSISTENCE_ENABLED=true
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_anon_key

# Optional (performance)
REDIS_URL=your_upstash_url

# CORS
FRONTEND_ORIGINS=https://medichat.vercel.app

# Production
SECURE_COOKIES=true
CSP_ENABLED=true
```

---

## 7. Scaling Strategy

### When You Outgrow Free Tier

| Bottleneck | Upgrade Path | Est. Cost |
|------------|-------------|-----------|
| Cohere rate limit (100 req/min) | Cohere Pro ($0.25/1M tokens) | Pay-as-you-go |
| Chroma (in-process, single server) | Qdrant Cloud ($0 → ~$25/mo) | ~$25/mo |
| Render Free (0.1 CPU, 512 MB) | Render Starter ($7/mo) | $7/mo |
| Supabase Free (500 MB DB) | Supabase Pro ($25/mo) | $25/mo |
| Vercel bandwidth | Vercel Pro ($20/mo) | $20/mo |

### Horizontal Scaling (Future)

```
                  ┌─ Render instance 1 ─┐
Cloudflare CDN ───── Render instance 2 ──── Qdrant Cluster
  (free)            └─ Render instance 3 ─┘     ($25/mo)
       │                                              │
  Vercel (Angular)                              Supabase
  (free)                                        ($25/mo)
```

---

## 8. Cost Breakdown

### Fully Free Stack (Option A: Chroma + Render + Vercel + Supabase)

| Service | Free Tier | Monthly Cost |
|---------|-----------|-------------|
| Render (backend) | 512 MB RAM, 750 hrs | **$0** |
| Vercel (frontend) | 100 GB bandwidth | **$0** |
| Supabase (DB+Auth) | 500 MB DB, 50k users | **$0** |
| Cohere API | 100 req/min | **$0** |
| Chroma/Qdrant Free | In-process or 1 GB | **$0** |
| Upstash Redis (optional) | 10 MB | **$0** |
| GitHub (repos + CI/CD) | Free | **$0** |
| Cloudflare (DNS + CDN) | Free | **$0** |
| **Total** | | **$0/month** |

### Low-Cost Stack (Option B: Keep Pinecone)

| Service | Cost |
|---------|------|
| Pinecone Serverless Starter | ~$70/mo |
| Everything else (free tier) | $0 |
| **Total** | **~$70/month** |

---

## 9. Recommended Approach

### 🏆 **Best Free Strategy: Option A with Chroma**

1. **Replace Pinecone → Chroma** (10 lines of code change)
2. **Deploy Flask on Render Free** (connect GitHub repo)
3. **Deploy Angular on Vercel Free** (connect GitHub repo)
4. **Supabase Free** for auth + persistence
5. **Cohere Free API** for LLM
6. **Upstash Free Redis** for session caching (optional)
7. **Cloudflare Free** for DNS + DDoS protection

**Result**: 100% free, scalable to ~100-200 concurrent users, easy to upgrade components individually as you grow.

### Quick Start Commands

```bash
# 1. Install Chroma
pip install chromadb

# 2. Build Angular for production
cd frontend && npm install && npm run build

# 3. Push to GitHub
git add .
git commit -m "Ready for production deployment"
git push origin main

# 4. Deploy on Render (connect repo, set env vars)
# 5. Deploy on Vercel (connect repo, set FRONTEND_ORIGINS)
```

---

## Migration Checklist

- [ ] Replace Pinecone with Chroma (or Qdrant Free) in `app/rag/`
- [ ] Set up Supabase project + run migrations
- [ ] Set up Cohere API key
- [ ] Create Render account + deploy backend
- [ ] Create Vercel account + deploy frontend
- [ ] Configure environment variables in Render/Vercel
- [ ] Update `FRONTEND_ORIGINS` to point to Vercel URL
- [ ] Test health endpoint: `https://your-app.onrender.com/health`
- [ ] Test chat flow end-to-end
- [ ] Set up GitHub Actions for auto-deploy (optional)
- [ ] Configure custom domain (optional)
- [ ] Enable HTTPS (automatic on Render/Vercel)
- [ ] Test PWA installability (Angular service worker)
- [ ] Monitor free tier usage limits