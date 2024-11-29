yes | docker container prune

docker build -t diffusion_wavelets .

docker run -it --name diffusion -v /data/ns241/satellite_tiles:/usr/data/images -v ./weights:/usr/data/weights --shm-size=1g --gpus 1 diffusion_wavelets:latest bash
