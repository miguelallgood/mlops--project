# Use a base image with Python
FROM python:3.11.9

# Set working directory
WORKDIR /app

# Copy the individual files
COPY ./scripts/experiment.py /app/experiment.py
COPY ./scripts/preprocessing.py /app/preprocessing.py
COPY ./scripts/entrypoint.sh /app/entrypoint.sh
COPY ./scripts/requirements.txt /app/requirements.txt

# Copy the data directory
COPY ./scripts/data /app/data/

# Install dependencies
RUN pip install -r requirements.txt  

# Expose MLflow port
EXPOSE 5000

# Set permissions for the entrypoint script
RUN chmod +x /app/entrypoint.sh

# Set MLflow server as the entry point
ENTRYPOINT ["/app/entrypoint.sh"]
