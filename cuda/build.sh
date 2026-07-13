# https://forums.developer.nvidia.com/t/unable-to-work-with-tensorflow-in-docker-on-dgx-spark/350419
export DOCKER_DEFAULT_PLATFORM=linux/amd64
#export DOCKER_DEFAULT_PLATFORM=linux/arm64
cd environments/windows
docker build -t ml-tensorflow-win .
cd ../../
docker run --rm --gpus all --name ml-tensorflow-win ml-tensorflow-win

