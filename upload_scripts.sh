#!/bin/bash
# Clear all caches
find . -type f -name '*.py[co]' -delete
find . -type d -name '__pycache__' -exec rm -rf {} +

# Create clean archive
tar czvf /tmp/clean_scripts.tar.gz \
    --exclude="*.pyc" \
    --exclude="__pycache__" \
    --exclude=".DS_Store" \
    scripts/

# Atomic upload
aws s3 cp /tmp/clean_scripts.tar.gz \
    s3://salesforce-leads-data-puneet/code/scripts.tar.gz \
    --cache-control="no-store, no-cache, must-revalidate" \
    --metadata="Content-Type=application/gzip" \
    --acl bucket-owner-full-control

# Verify
echo "Uploaded script content:"
aws s3 cp s3://salesforce-leads-data-puneet/code/scripts.tar.gz - | tar tz | head -n 10