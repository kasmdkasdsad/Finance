# syntax=docker/dockerfile:1
# One image for the API and the dashboard. No secret is ever baked in: .env files, deploy/ and data/ are excluded
# by .dockerignore, and every credential arrives at run time from the environment (deploy/.env on the server).
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN useradd --create-home --uid 10001 quantpulse

COPY pyproject.toml constraints.txt README.md alembic.ini ./
COPY src ./src
COPY frontend ./frontend
COPY .streamlit ./.streamlit

RUN pip install ".[frontend]" -c constraints.txt \
    && mkdir -p /app/data /app/logs \
    && chown -R quantpulse:quantpulse /app

USER quantpulse

ENV QP_DATABASE_URL=sqlite+aiosqlite:////app/data/quantpulse.db

# the commit this image was built from (the server's ./qp update passes it; shown in the status and the logs)
ARG QP_GIT_COMMIT=""
ENV QP_GIT_COMMIT=${QP_GIT_COMMIT}
LABEL org.opencontainers.image.revision=${QP_GIT_COMMIT}

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "--factory", "quantpulse.api.app:app_factory", "--host", "0.0.0.0", "--port", "8000"]
