#!/bin/bash
# Startup script for Render deployment
# This ensures uvicorn worker is used even if Render uses default gunicorn command

exec gunicorn app:app \
  --bind 0.0.0.0:${PORT:-10000} \
  --workers 1 \
  --worker-class uvicorn.workers.UvicornWorker \
  --timeout 300 \
  --access-logfile - \
  --error-logfile - \
  --log-level info

