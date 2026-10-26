# Face Matching Network

Face verification: a ResNet-50 ArcFace model and a VGG16/ResNet50
siamese-contrastive baseline fine-tuned on CelebA/CASIA-WebFace and
evaluated on LFW, FaceNet/AdaFace/InsightFace/MagFace baselines for
comparison, and Grad-CAM based explainability analysis.

## Repository layout

```
.
├── train_arcface_resnet.py          # main ArcFace training script
├── train_siamese_methodology.py     # VGG16/ResNet50 siamese-contrastive methodology
├── baseline_facenet.py              # FaceNet baseline (comparison)
├── baseline_adaface.py              # AdaFace baseline (comparison)
├── baseline_insightface.py          # InsightFace baseline (comparison)
├── baseline_magface.py              # MagFace baseline (comparison)
├── magface_iresnet_arch.py          # iResNet-100 architecture used by the MagFace baseline
├── align_dataset.py / align_arcface_112.py   # face alignment (MTCNN) used by training/baselines
├── gradcam_alignment.py             # shared model/data loading hub (used by the baselines
│                                     # and the Grad-CAM scripts below)
├── gradcam_face_region_metric.py    # quantitative Grad-CAM face-region-energy metric
├── gradcam_example_figures.py       # Grad-CAM example-panel figure
├── config.py                        # dataset paths & hyperparameters (shared config)
├── data/                            # dataset loading / pairing pipeline (code only)
├── logs/
│   ├── v4/      # VGG16-siamese, SGD + aug + L4-5 finetune + LR schedule (unaligned)
│   ├── v5/      # same as v4, MTCNN-aligned + full augmentation
│   ├── r4/      # ResNet50-siamese, SGD + aug + L4-5 finetune + LR schedule (unaligned)
│   ├── r5/      # same as r4, MTCNN-aligned + full augmentation
│   ├── af1/     # ResNet50-ArcFace, 224x224, layer3/4 finetune (new method)
│   ├── af2/     # ResNet50-ArcFace, unaligned, light augmentation
│   ├── af3/     # ResNet50-ArcFace, aligned, full augmentation
│   ├── af4/     # ResNet50-ArcFace, 224x224, without TTA, aligned datasets
│   └── scratch/ # reference_paper_reproduction/ results (LFW only)
├── gradcam_results/face_region_metric/
├── results_summary/                 # baseline comparison tables/plots
└── reference_paper_reproduction/    # from-scratch siamese-VGG reproduction of
                                      # Heidari & Fouladi-Ghaleh, "Using Siamese
                                      # Networks with Transfer Learning for Face
                                      # Recognition on Small-Samples Datasets"
                                      # (results in logs/scratch/)
```

## Datasets

- CelebA: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- CASIA-WebFace: https://paperswithcode.com/dataset/casia-webface (research-use license)
- LFW: https://vis-www.cs.umass.edu/lfw/

Point `config.py` at wherever you download/align them locally.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
