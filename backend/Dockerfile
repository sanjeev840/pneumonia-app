FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 TF_CPP_MIN_LOG_LEVEL=2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY model/ ./model/

EXPOSE 5000
CMD ["gunicorn", "-w", "1", "--threads", "2", "--timeout", "120", "-b", "0.0.0.0:5000", "app:app"]
