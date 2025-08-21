# Lambda-compatible Python 3.11 image for ARM64
FROM public.ecr.aws/lambda/python:3.11-arm64

WORKDIR /lambda

COPY requirements.txt .

# Install dependencies and create layer
RUN yum install -y zip && \
    pip install --upgrade pip && \
    mkdir -p python && \
    pip install -r requirements.txt -t python/ --no-cache-dir && \
    find python/ -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find python/ -name "*.pyc" -delete && \
    zip -r layer.zip python/ && \
    ls -la layer.zip
