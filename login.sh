#!/bin/bash

# Load environment variables from .env
set -a
source .env
set +a

# Check required variables
if [[ -z "$OPENSHIFT_TOKEN" || -z "$OPENSHIFT_API_URL" ]]; then
  echo "Missing OPENSHIFT_TOKEN or OPENSHIFT_API_URL in .env"
  exit 1
fi

# Perform OpenShift login (skip TLS verify)
echo "Logging in to OpenShift..."
oc login --insecure-skip-tls-verify=true --token="$OPENSHIFT_TOKEN" --server="$OPENSHIFT_API_URL"

# Show current user
oc whoami
