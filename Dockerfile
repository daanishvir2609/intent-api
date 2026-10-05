FROM public.ecr.aws/docker/library/python:3.14-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 1) Dependencies first: this layer is cached until requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Code + the trained model (app/main.py imports predict_with_fallback from src/)
COPY src/ src/
COPY app/ app/
COPY model/ model/

# 3) Lets this SAME image run on AWS Lambda; it is ignored everywhere else
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 /lambda-adapter /opt/extensions/lambda-adapter
ENV AWS_LWA_PORT=8080 \
    AWS_LWA_READINESS_CHECK_PATH=/health

# 4) Never run as root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
