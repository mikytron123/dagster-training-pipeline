FROM python:3.11-slim

RUN mkdir -p /opt/dagster/dagster_home /opt/dagster/app

ENV DAGSTER_HOME=/opt/dagster/dagster_home/
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV GIT_PYTHON_REFRESH=quiet

RUN mkdir -p $DAGSTER_HOME

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
     --mount=type=bind,source=requirements.txt,target=requirements.txt \
     python -m pip install -r requirements.txt

COPY dagster.yaml workspace.yaml $DAGSTER_HOME

WORKDIR $DAGSTER_HOME