---
description: "Use when changing Docker Compose services, environment examples, Dockerfiles, or Python dependency manifests and lockfiles. Covers service DNS, Dagster compatibility, lock regeneration, and validation."
applyTo: "docker-compose.yaml, env.example, **/Dockerfile, **/requirements*.in, **/requirements*.txt"
---

# Compose And Dependency Changes

- For general project setup and architecture, follow [AGENTS.md](../../AGENTS.md) and [README.md](../../README.md); keep this instruction focused on stack and dependency changes.
- When adding or renaming a Compose service, check that internal URLs in `env.example` and service environment settings use the Compose service key as the hostname. The current example has `MLFLOW_HOST=mflow`, but the service key is `mlflow`; preserve or correct this deliberately when touching the setting.
- Before changing Dagster dependencies, compare `dagster/requirements.in` and `training_pipeline/requirements.in` and their generated lockfiles. The webserver/daemon and user-code images currently pin different Dagster release families; verify cross-version compatibility and update the relevant pins and lockfiles together rather than assuming either side can change independently.
- Treat `requirements*.txt` files as pip-compile-generated locks. Update the corresponding `requirements*.in` first, then regenerate its lock with pip-compile; do not hand-edit generated transitive pins. Check the lock's recorded Python version against the target image before regenerating: the current lock headers say Python 3.12, while `training_pipeline/Dockerfile` uses Python 3.11.
- Keep credentials out of checked-in Compose files and `env.example`; use placeholder values there and local `.env` values for development.
- After Compose or environment changes, run `docker compose config`. After dependency or Dockerfile changes, build the affected service with `docker compose build <service>` when Docker is available; avoid rebuilding unrelated services.