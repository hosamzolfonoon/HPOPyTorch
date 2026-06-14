#!/bin/bash
set -x  # Enable script debugging

echo "Copying default configuration to Nginx sites-available..."
cp -rf default /etc/nginx/sites-available/default

echo "Reloading Nginx..."
nginx -s reload

echo "Starting Streamlit with PM2..."
pm2 start "streamlit run /deeplearnhpost/deeplearnhpost.py \
  --server.port 8501 \
  --server.address 0.0.0.0" \
  --name DeepLearnHPOSt \
  --output streamlit-out.log \
  --error streamlit-error.log
