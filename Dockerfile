FROM python:3.11.13-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads sessions logs

EXPOSE 5000