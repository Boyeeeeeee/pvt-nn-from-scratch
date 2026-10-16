FROM python:3.11-slim
WORKDIR /app

# CPU-only torch keeps the image small
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY src ./src
COPY models ./models
COPY app ./app

ENV PYTHONPATH=/app/src:/app
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
