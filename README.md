# Deploying inference engine over Red Hat OpenShift platform

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Static Badge](https://img.shields.io/badge/Proudly-Canadian-FF0000.svg)

```bash
docker build -t prabaharravichandran/ufpsapp:1.0 .
docker login
docker push prabaharravichandran/ufpsapp:1.0
```


```bash
oc apply -f ufpsapp.yaml
oc apply -f ufpsapp-service.yaml
oc apply -f ufpsapp-route.yaml
oc get route ufpsapp-route -o jsonpath="http://{.spec.host}{'\n'}"
```