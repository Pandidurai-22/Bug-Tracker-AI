# Gunicorn configuration file
# This will be automatically used by gunicorn if present in the same directory

import os
import multiprocessing

# Server socket - use PORT from environment (Render sets this)
port = int(os.environ.get("PORT", 10000))
bind = f"0.0.0.0:{port}"

# Worker processes - CRITICAL: Must use uvicorn worker for FastAPI (ASGI)
workers = 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 300
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "bug-tracker-ai"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

