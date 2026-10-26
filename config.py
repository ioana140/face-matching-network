

from pathlib import Path

# Paths
LFW_ROOT  = Path(r"/datasets/lfw/lfw")       # folder with one sub-dir per person
PAIRS_CSV = Path(r"/datasets/lfw/pairs.csv")

# CASIA-WebFace
CASIA_ROOT = Path(r"/datasets/casia/casia_webface")

# CelebA
CELEBA_ROOT           = Path(r"/datasets/celeba/img_align_celeba")
CELEBA_IDENTITY_FILE  = Path(r"/datasets/celeba/identity_CelebA.txt")
CELEBA_PARTITION_FILE = Path(r"/datasets/celeba/list_eval_partition.txt")

# MTCNN-aligned datasets
CELEBA_ALIGNED_ROOT   = Path(r"/datasets/celeba_aligned")
LFW_ALIGNED_ROOT      = Path(r"/datasets/lfw_aligned")
CELEBA_TARGET_PAIRS   = 200_000

# Folder output
ROOT_DIR       = Path(__file__).parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOG_DIR        = ROOT_DIR / "logs"

for _d in (CHECKPOINT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


TARGET_PAIRS  = 15_000
SIMILAR_RATIO = 0.5

TRAIN_SPLIT = 0.60       # 60 % train / 40 % test
TEST_SPLIT  = 0.40

# CASIA-WebFace
CASIA_TARGET_PAIRS = 100_000
CASIA_TRAIN_SPLIT  = 0.80
CASIA_VAL_SPLIT    = 0.20
RANDOM_SEED = 42

LABEL_SIMILAR    = 0
LABEL_DISSIMILAR = 1


# Input
IMAGE_HEIGHT   = 128
IMAGE_WIDTH    = 128
IMAGE_CHANNELS = 3
INPUT_SHAPE    = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

VGGFACE_IMAGE_HEIGHT = 224
VGGFACE_IMAGE_WIDTH  = 224


# Model — VGG-16 encoder
VGG16_FREEZE_UP_TO = 17

# Pretrained VGG-Face
VGGFACE_WEIGHTS_PATH = ROOT_DIR / "weights" / "vggface.pth"
FC_UNITS           = 512
FC_NUM_LAYERS      = 3
EMBEDDING_DIM      = 512


# Loss
CONTRASTIVE_MARGIN = 1.0
DECISION_THRESHOLD = 0.5


# Training
BATCH_SIZE    = 64
LEARNING_RATE = 5e-4
BACKBONE_LR   = 5e-5
WEIGHT_DECAY  = 1e-4
DROPOUT       = 0.3
EPOCHS        = 100


WANDB_PROJECT  = "siamese-face-recognition"
WANDB_RUN_NAME = "siamese-vgg-lfw"
