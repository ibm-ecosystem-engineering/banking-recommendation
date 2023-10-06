# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt .
COPY . .

# Install essential packages including ensurepip

# Create and activate a virtual environment
RUN python3 -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Install any needed packages specified in requirements.txt
RUN python -m venv venv
RUN venv/bin/pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD . venv/bin/activate && exec python app.py




