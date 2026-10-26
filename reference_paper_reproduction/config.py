from pathlib import Path


ROOT_DIR = Path(__file__).parent
REPO_ROOT = ROOT_DIR.parent

LFW_ROOT  = REPO_ROOT / "datasets" / "lfw" / "lfw"       # folder with one sub-dir per person
PAIRS_CSV = REPO_ROOT / "datasets" / "lfw" / "pairs.csv"

# Project output directories (created automatically)
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOG_DIR        = ROOT_DIR / "logs"

for _d in (CHECKPOINT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


TARGET_PAIRS  = 15_000
SIMILAR_RATIO = 0.5

TRAIN_SPLIT = 0.60
TEST_SPLIT  = 0.40
RANDOM_SEED = 42


LABEL_SIMILAR    = 0
LABEL_DISSIMILAR = 1


IMAGE_HEIGHT   = 128
IMAGE_WIDTH    = 128
IMAGE_CHANNELS = 3
INPUT_SHAPE    = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


# Model — VGG-16 encoder
VGG16_FREEZE_UP_TO = 24
FC_UNITS           = 512
FC_NUM_LAYERS      = 3
EMBEDDING_DIM      = 512


# Loss
CONTRASTIVE_MARGIN = 1.0   # m = 1

# Decision threshold
DECISION_THRESHOLD = 0.5

# Training
BATCH_SIZE    = 32
LEARNING_RATE = 0.000005
EPOCHS        = 100

# Weights & Biases
WANDB_PROJECT  = "siamese-face-recognition"
WANDB_RUN_NAME = "siamese-vgg-lfw"
