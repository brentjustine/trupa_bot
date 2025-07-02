FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc curl unzip build-essential libffi-dev python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first
RUN pip install torch==2.2.2+cpu torchvision==0.17.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary app files
COPY . .

CMD ["python", "main.py"]
