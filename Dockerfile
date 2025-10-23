# 1. Start with an official Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements file first to cache packages
COPY requirements.txt .

# 4. Install all the Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your backend code into the container
COPY . .

# 6. Expose the port your app will run on
EXPOSE 8000

# 7. The command to run your app
# We use gunicorn here, a production-grade server (uvicorn is for dev)
# We listen on 0.0.0.0 (all interfaces) and the port GCP provides
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]
