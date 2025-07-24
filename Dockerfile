ARG PYTHON_VERSION=3.11.13

FROM python:${PYTHON_VERSION}-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads sessions logs

EXPOSE 5000