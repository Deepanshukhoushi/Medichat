# Refactoring Report

## What Changed

- Split the monolithic Flask logic into controllers, services, repositories, core config, logging, security, and RAG modules.
- Restored the Flask template and static asset pipeline under `frontend/`.
- Centralized environment configuration and standardized path handling.
- Added a TTL-backed session history store with optional Redis support.
- Replaced loose auth parsing with validated Pydantic request schemas.

## Why

- To reduce coupling and make the codebase easier to extend without editing a single large file.
- To preserve behavior while making dependencies explicit.
- To prevent memory growth from unbounded in-memory session history.
- To make request validation and error responses consistent.

