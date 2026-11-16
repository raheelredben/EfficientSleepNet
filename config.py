"""Configuration for EfficientSleepNet single-channel sleep-stage classification."""
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
DATA_DIR = str(ROOT / "data")
MODEL_DIR = str(ROOT / "trained_models")
RESULTS_DIR = str(ROOT / "results")

# Input data and preprocessing.
DATASET = "sleep_edf"                 # Only Sleep-EDF EDF+hypnogram loader implemented.
SAMPLE_RATE = 100
EPOCH_DURATION = 30
INPUT_LENGTH = SAMPLE_RATE * EPOCH_DURATION
IN_CHANNELS = 1
EEG_CHANNEL_BY_DATASET = {"sleep_edf": "Fpz-Cz", "isruc": "C4-A1", "shhs": "C4-A1"}
NUM_CLASSES = 5                       # W, N1, N2, N3, REM
CLASS_NAMES = ["W", "N1", "N2", "N3", "R"]
TRIM_WAKE_MINUTES = None
# Set the desired Sleep-EDF participant cohort explicitly.
SLEEP_EDF_SUBJECT_IDS = tuple(f"SC4{i:02d}" for i in range(20))
REQUIRE_COMPLETE_COHORT = True         # Fail rather than silently compare a smaller cohort.

# Two parallel multi-resolution CNN branches with optional compact channel connectivity.
# Compact mode uses grouped convolutions and pointwise channel mixing.
# Dense mode uses standard (ungrouped) convolutions.
COMPACT_CONVOLUTIONS = True
ARCHITECTURE_SPEC = "compact_2p5m" if COMPACT_CONVOLUTIONS else "dense_convolutions"
# Keep experiment artifacts for the compact and dense alternatives separate.
MODEL_DIR = str(ROOT / "trained_models" / ARCHITECTURE_SPEC)
RESULTS_DIR = str(ROOT / "results" / ARCHITECTURE_SPEC)
MRCNN_CHANNELS = (32, 64, 128)
MRCNN_SHORT_KERNELS = (50, 50, 50)
MRCNN_LONG_KERNELS = (400, 400, 400)
MRCNN_POOL_SIZE = 4
# Compact settings: short Conv3 groups=2, long Conv2/Conv3 groups=4.
# Long Conv3 gains a 1x1 pointwise mixing matrix to restore cross-group interaction.
# Temporal output length is unchanged by grouping.
MRCNN_SHORT_GROUPS = (1, 1, 2) if COMPACT_CONVOLUTIONS else (1, 1, 1)
MRCNN_LONG_GROUPS = (1, 4, 4) if COMPACT_CONVOLUTIONS else (1, 1, 1)
MRCNN_LONG_CONV3_POINTWISE = COMPACT_CONVOLUTIONS
MRCNN_DROPOUT = 0.30

CARA_HEADS = 5
CARA_PROJECTION_DIM = 255
CARA_REDUCTION = 4
CARA_GROUPS = 8
CARA_DEPTHWISE_KERNEL = 3
CARA_DROPOUT = 0.0

RNN_HIDDEN_SIZE = 128
RNN_NUM_LAYERS = 2
RNN_CELL_TYPE = "lstm"
RNN_DROPOUT = 0.0
ATTENTION_POOL_DIM = 128

# Stage-dependent loss weight coefficients.
CLASS_AWARE_RHO = 1.2
CLASS_AWARE_ALPHA = 1.0
CLASS_AWARE_BETA = 1.5
CLASS_AWARE_GAMMA = 2.0

# Subject-independent 5-fold CV, 3:1:1 train/validation/test.
N_FOLDS = 5
RANDOM_STATE = 42
BATCH_SIZE = 32
OPTIMIZER = "rmsprop"
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# Training schedule and optional controls.
EPOCHS = 100
EARLY_STOP_PATIENCE = None           # None disables early stopping.
GRAD_CLIP_NORM = None                 # None disables gradient clipping.
USE_LR_SCHEDULER = False
SAVE_METRICS_EVERY_EPOCH = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CPU_NUM_THREADS = 2                   # Environment convenience; not a scientific parameter.
if DEVICE == "cpu":
    torch.set_num_threads(CPU_NUM_THREADS)
