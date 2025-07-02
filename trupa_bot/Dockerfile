FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libffi-dev \
    python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (important!)
RUN pip install --no-cache-dir torch==2.2.2+cpu torchvision==0.17.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Copy and install all other dependencies (torch is now pre-installed)
COPY requirements.txt .
RUN pip install --no-cache-dir --no-deps -r requirements.txt

# Copy your application code
COPY . .

CMD ["python", "main.py"]
