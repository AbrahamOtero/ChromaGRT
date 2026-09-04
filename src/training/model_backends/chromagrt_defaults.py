"""Fixed architecture and runtime defaults for ChromaGRT."""


HIDDEN_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
FFN_DIM = 512
MAX_DISTANCE = 8
WEIGHT_DECAY = 1e-2
EARLY_STOPPING_PATIENCE = 30


DEFAULT_CHROMAGRT_CONFIG = {
    "batch_size": 16,
    "num_workers": 0,
    "condition_normalization": "standard",
    "use_tanaka": True,
    "excluded_condition_blocks": (),
    "excluded_molecular_descriptors": (),
    "dropout": 0.05,
    
    "lr": 5e-5,
    "max_epochs": 250,
    "seed": 42,
    "accelerator": "auto",
}
