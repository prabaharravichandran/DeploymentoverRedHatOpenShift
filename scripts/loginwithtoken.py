import os
from dotenv import load_dotenv
from kubernetes import client, config

# === SET YOUR CLUSTER DETAILS HERE ===
load_dotenv()

api_url = os.getenv("OPENSHIFT_API_URL")
token = os.getenv("OPENSHIFT_TOKEN")

# === CONFIGURE CLIENT ===
configuration = client.Configuration()
configuration.host = api_url
configuration.verify_ssl = False  # Set to True if you have proper CA certs
configuration.api_key = {"authorization": f"Bearer {token}"}
client.Configuration.set_default(configuration)

# === USE THE CLIENT ===
v1 = client.CoreV1Api()

# Example: List Pods in a namespace
namespace = "prr000-devspaces"
pods = v1.list_namespaced_pod(namespace=namespace)
for pod in pods.items:
    print(f"{pod.metadata.name} - {pod.status.phase}")
