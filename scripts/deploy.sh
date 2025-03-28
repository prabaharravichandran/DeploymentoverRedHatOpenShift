#!/bin/bash

set -e  # Exit if any command fails

# Step 1: Apply all resources
echo "Applying Deployment, Service, and Route..."
oc apply -f ufpsapp.yaml
oc apply -f ufpsapp-service.yaml
oc apply -f ufpsapp-route.yaml

# Step 2: Wait for the pod to be ready
echo "Waiting for pod to be ready..."
oc wait --for=condition=Ready pod -l app=ufpsapp --timeout=120s

# Step 3: Get the route URL
ROUTE_URL=$(oc get route ufpsapp-route -o jsonpath='http://{.spec.host}')
echo "App is available at: $ROUTE_URL"

# Step 4: Optional test with curl
echo "Testing with curl..."
HTTP_STATUS=$(curl -o /dev/null -s -w "%{http_code}\n" "$ROUTE_URL")

if [ "$HTTP_STATUS" = "200" ]; then
  echo "App responded successfully (HTTP 200)"
else
  echo "Warning: App returned HTTP $HTTP_STATUS"
fi
