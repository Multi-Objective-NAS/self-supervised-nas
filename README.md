# Environments
- conda environment configs: `environment.yaml`

# How to train
- `CUDA_VISIBLE_DEVICES={device_index} python3 pretrain.py experiment=TripletMarginLoss`
- `outputs/{date}/{time}/logs` -> tensorboard directory
- Above direcoty can be checked with `tensorboard --bind_all --logdir={.../logs} --port 8080`
- `outputs/{date}/{time}/weights` -> output model weights
- **Use with caution** `pgrep -f "python pretrain.py"` will show all the training process. `pkill -f "python pretrain.py"` will kill them all.
