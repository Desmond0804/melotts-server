# MeloTTS Server

## Install
```
git clone https://github.com/Desmond0804/melotts-server.git
cd melotts-server
pip install -r requirements.txt
python -m unidic download
```

## Run Server
```
python app.py
```

## Build Docker Image
```
git clone -b MeloTTS-MS https://github.com/Desmond0804/melotts-server.git
cd melotts-server
docker build \
    --build-arg http_proxy=$http_proxy \
    --build-arg https_proxy=$https_proxy \
    -t melotts-server .
```

## Run Docker Container
```
docker run -d \
    --net=host \
    -e http_proxy=$http_proxy \
    -e https_proxy=$https_proxy \
    -e no_proxy=$no_proxy \
    --device=/dev/dri \
    --restart always \
    --name=melotts \
    melotts-server
```

## Run Docker Container (using Docker Compose)
```
docker compose up -d
```
