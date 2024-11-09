#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE dagster;
    CREATE USER dagster_user WITH PASSWORD 'dagster_pass';
    GRANT ALL PRIVILEGES ON DATABASE dagster TO dagster_user;
    GRANT ALL ON SCHEMA public TO dagster_user;
    ALTER DATABASE dagster OWNER TO dagster_user;

    CREATE DATABASE mlflow;
    CREATE USER mlflow_user WITH PASSWORD 'mlflow_pass';
    GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow_user;
    GRANT ALL ON SCHEMA public TO mlflow_user;
    ALTER DATABASE mlflow OWNER TO mlflow_user;

    CREATE DATABASE optuna;
    CREATE USER optuna_user WITH PASSWORD 'optuna_pass';
    GRANT ALL PRIVILEGES ON DATABASE optuna TO optuna_user;
    GRANT ALL ON SCHEMA public TO optuna_user;
    ALTER DATABASE optuna OWNER TO optuna_user;
EOSQL
    