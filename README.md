# Environments
- conda environment configs: `environment.yaml`
- How to install:
```bash
conda env create -f environment.yaml
conda activate self-supervised-nas-3.6
pip install -e libs/nasbench --no-deps
pip install -e libs/NAS-Bench-201
```

# How to train
- `CUDA_VISIBLE_DEVICES={device_index} python3 pretrain.py experiment=TripletMarginLoss`
- `outputs/{date}/{time}/logs` -> tensorboard directory
- Above directory can be checked with `tensorboard --bind_all --logdir=/path/to/log --port 8080` or `tensorboard --bind_all --logdir_spec=exp1:/path/to/log1,exp2:/path/to/log2 --port 8080`
- `outputs/{date}/{time}/weights` -> output model weights
- **Use with caution** `pgrep -f "python pretrain.py" -a` will show all the training process. `pkill -f "python pretrain.py"` will kill them all.

# tmux cheatsheet
- `tmux ls`: Current tmux sessions
- `tmux new -s {session_name}`: Create session with `session_name`
- `tmux attach -t {session_name}`: Attach to existing session
- (Inside tmux session) `Ctrl+a d`: Detach existing session
- (Inside tmux session) `Ctrl+a PgUp/PgDn`: Scroll inside session (`Ctrl+c` to abort)
- (Inside tmux session) `Ctrl+d`: Close session
