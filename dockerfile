# Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install pandas scikit-learn fairlearn pyyaml joblib
ENTRYPOINT ["python", "agent.py", "--config", "config.yaml"]
