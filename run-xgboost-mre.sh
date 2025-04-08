docker run \
  --rm \
  -it \
  --gpus all \
  --pull=always \
  --network=host \
  --volume $PWD:/repo \
  --workdir /repo \
  rapidsai/ci-conda:cuda11.8.0-ubuntu22.04-py3.10 \
  bash xgboost-mre.sh
