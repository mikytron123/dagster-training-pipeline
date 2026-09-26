# Repository Guidance

- Read [README.md](README.md) for project setup and the Dagster UI workflow; do not duplicate its setup steps here.
- The stack is orchestrated by [docker-compose.yaml](docker-compose.yaml). Copy `env.example` to `.env` before starting it with `docker compose up`.
- Dagster definitions are assembled in `training_pipeline/definitions.py`; asset dependencies and the training job live in `training_pipeline/assets.py`. Keep S3 access and Parquet persistence in `training_pipeline/resources/`, and model objectives/scoring in `training_pipeline/models/`.
- Use `just format` for Black formatting. `just lint` runs isort, pyupgrade (`--py312-plus`), autoflake, and flake8 across the repository; it modifies files, so review its changes before keeping them.
- Check for relevant tests before changing behavior. The repository does not define a test task in `justfile`; run focused tests with the project's available test tooling rather than assuming a project-wide test command.
- When changing Dagster dependencies, check both `dagster/requirements.txt` and `training_pipeline/requirements.txt`: they currently pin different Dagster versions, and the webserver/user-code versions must remain compatible.
- When changing service configuration, compare `.env` values with Compose service names. In particular, `env.example` currently sets `MLFLOW_HOST=mflow`, while the Compose service is named `mlflow`.