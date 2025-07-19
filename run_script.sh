torchrun --nproc_per_node=2 multilayer_neural_networks_ddp.py
# for all available GPUs run below:
# torchrun --nproc_per_node=$(nvidia-smi -L | wc -l) multilayer_neural_networks_ddp.py
