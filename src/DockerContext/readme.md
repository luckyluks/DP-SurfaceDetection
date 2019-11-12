grabbed from https://github.com/JulianoLagana/deep-machine-learning

## run jupyter notebook
docker run command to start up jupyter notebook on cloud (standard image/ costum image)
```
original: $ docker run -it -e HOST_USER_ID=$(id -u) -e HOST_GROUP_ID=$(id -g) -v "$PWD":/workspace -p 9090:8888 --ipc=host --gpus all ssy340dml/dml-image:gpu
new gpu: $ docker run -it -e HOST_USER_ID=$(id -u) -e HOST_GROUP_ID=$(id -g) -v "$PWD":/workspace -p 9090:8888 --ipc=host --gpus all dpproject/image:gpu
new cpu: $ docker run -it -e HOST_USER_ID=$(id -u) -e HOST_GROUP_ID=$(id -g) -v "$PWD":/workspace -p 9090:8888 --ipc=host --gpus all dpproject/image:cpu
```

## build costum image
rebuild docker image with updated context from this folder
1. make sure you are in the root directory of the repo
2. use docker build to create image (be careful wiht gpu/cpu version)
    ```
    docker build --build-arg CONDA_ENV_SUFFIX=gpu -t dpproject/image:gpu src/DockerContext
    docker build --build-arg CONDA_ENV_SUFFIX=cpu -t dpproject/image:cpu src/DockerContext
    ```


### remove tangling images
removes all unused images/containers in one call:
```
docker system prune
```
Or do it manually, by removing single images by their ID, which you can get from "docker images -a" command:
```
docker images -a
docker images <ID_HERE>
```
or combined:
```
docker rmi $(docker images --filter "dangling=true" -q --no-trunc)
```