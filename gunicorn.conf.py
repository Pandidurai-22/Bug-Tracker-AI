# Gunicorn configuration file
# This will be automatically used by gunicorn if present in the same directory

import os
import multiprocessing

# Server socket - use PORT from environment (Render sets this)
port = int(os.environ.get("PORT", 10000))
bind = f"0.0.0.0:{port}"

# Debug: Print port being used
import sys
print(f"[Gunicorn Config] Using PORT from environment: {os.environ.get('PORT', 'NOT SET')}", file=sys.stderr)
print(f"[Gunicorn Config] Binding to: {bind}", file=sys.stderr)

# Worker processes - CRITICAL: Must use uvicorn worker for FastAPI (ASGI)
workers = 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 600  # Increased to 10 minutes for model loading
keepalive = 5
graceful_timeout = 120  # Time to wait for workers to finish

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

