# Park Copilot API: runs on any Docker host (Render free tier in production).
#   docker build -t park-copilot .
#   docker run -p 7860:7860 --env-file .env park-copilot
FROM python:3.12-slim

# uv: fast installs, exactly the versions pinned in uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /bin/uv

# Non-root user (uid 1000 also matches what Hugging Face Spaces expects)
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /home/user/app

# Dependencies first: this layer is cached as long as uv.lock does not change
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY --chown=user app ./app
COPY --chown=user knowledge ./knowledge

# Download the embedding model at build time: no download on each cold start
RUN python -c "from app.config import get_settings; from app.rag.index import FastEmbedEmbedder; \
s = get_settings(); FastEmbedEmbedder(s.embedding_model, s.fastembed_cache_dir)"

EXPOSE 7860
# The host may impose the port through $PORT (Render does); 7860 otherwise.
# --proxy-headers: the real client IP comes from the host's proxy (used by rate limiting).
# `exec` makes uvicorn PID 1, so it receives the stop signal and shuts down cleanly.
CMD ["sh", "-c", "exec uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers --forwarded-allow-ips '*'"]
