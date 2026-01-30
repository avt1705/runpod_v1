# Use a modern PyTorch image (PyTorch 2.4.0+ is required for enable_gqa)
FROM runpod/pytorch:2.4.0-py3.10-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# Start the handler
CMD ["python", "-u", "handler.py"]
