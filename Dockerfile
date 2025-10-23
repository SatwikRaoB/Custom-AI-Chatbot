FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements file from the backend/ subdirectory on the host 
# into the current WORKDIR (/app) in the container
COPY backend/requirements.txt .

# 4. Install all the Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the Python application files (main.py, etc.) 
# from the backend/ subdirectory on the host to /app in the container
COPY backend/. .

# 6. Expose the port your app will run on
EXPOSE 8000

# 7. The command to run your app
# The 'main:app' command now correctly finds /app/main.py
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]
