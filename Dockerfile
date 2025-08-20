# Multi-stage Dockerfile
FROM python:3.11.13-slim-bookworm as builder

WORKDIR /app
COPY prod.requirements.txt .
RUN pip install --user --no-cache-dir -r prod.requirements.txt

FROM python:3.11.13-slim-bookworm
WORKDIR /app

# Copy only the installed packages
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY . .
RUN mkdir -p uploads logs

CMD gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --workers 4