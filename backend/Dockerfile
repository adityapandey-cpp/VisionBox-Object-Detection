FROM python:3.10-slim

# OpenCV headless still needs a couple of system libs
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PORT=8000
EXPOSE 8000
CMD ["gunicorn", "-w", "2", "-t", "180", "-b", "0.0.0.0:8000", "app:app"]
