#!/bin/bash

if [[ -z "${O3AS_LISTEN_IP}" ]]; then
    export O3AS_LISTEN_IP='0.0.0.0'
fi

if [[ -z "${O3AS_PORT}" ]]; then
    export O3AS_PORT=5005
fi

if [[ -z "${O3AS_TIMEOUT}" ]]; then
    export O3AS_TIMEOUT=120
fi

if [[ -z "${O3AS_WORKERS}" ]]; then
    export O3AS_WORKERS=1
fi

if [ "${ENABLE_HTTPS}" == "True" ]; then
  if test -e /certs/cert.pem && test -f /certs/key.pem ; then
    exec gunicorn --bind $O3AS_LISTEN_IP:$O3AS_PORT -w "$O3AS_WORKERS" \
    --certfile /certs/cert.pem --keyfile /certs/key.pem --timeout "$O3AS_TIMEOUT"  o3as:app
  else
    echo "[ERROR] File /certs/cert.pem or /certs/key.pem NOT FOUND!"
    exit 1
  fi
else
  exec gunicorn --bind $O3AS_LISTEN_IP:$O3AS_PORT -w "$O3AS_WORKERS" --timeout "$O3AS_TIMEOUT"  o3as:app
fi
