FROM python:3.11-slim

ENV MPLCONFIGDIR=/tmp/matplotlib
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GIT_PYTHON_REFRESH=quiet

WORKDIR /opt/dagster/app

RUN --mount=type=cache,target=/root/.cache/pip \
     --mount=type=bind,source=requirements.txt,target=requirements.txt \
     python -m pip install -r requirements.txt

COPY . .


# Run dagster gRPC server on port 4266

EXPOSE 4266

# CMD allows this to be overridden from run launchers or executors that want
# to run other commands against your repository
CMD ["dagster", "api", "grpc", "-h", "0.0.0.0", "-p", "4266", "--python-file", "definitions.py","--working-directory","/opt/dagster/app"]