# Dockerfile (UPDATED for direct litellm installation)
FROM python:3.10-slim
WORKDIR /app
COPY . /app
# Explicitly install litellm alongside giskard
RUN pip install pandas scikit-learn fairlearn pyyaml joblib requests giskard litellm
ENTRYPOINT ["python", "agent.py", "--config", "config.yaml"]