# Docker Hub Publishing Guide

## Prerequisites

- [Docker Hub](https://hub.docker.com) account
- Docker CLI installed
- Image built locally: `docker build -t hlas-server .`

## Publish

```bash
# Log in to Docker Hub
docker login

# Tag the image
docker tag hlas-server davidbudac/hlas-server:latest

# Push to Docker Hub
docker push davidbudac/hlas-server:latest
```

## Pull (from any machine)

```bash
docker pull davidbudac/hlas-server:latest
```
