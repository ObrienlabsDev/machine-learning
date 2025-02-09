
cd environments/windows
docker build -t ml-tensorflow-win .
cd ../../
docker run --rm --gpus all --name ml-tensorflow-win ml-tensorflow-win

