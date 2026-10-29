# Reproduction Guide

## Release Status

This repository contains the original research Python scripts and new submission utilities. The research code was developed for a Linux CUDA VM with an NVIDIA A100-SXM4 40 GB GPU. The current release does not include trained weights, datasets, training logs, or a complete lockfile of that VM.

The recovered source is preserved in `research/`, including its original absolute paths and historical defaults. This supports inspection of what was implemented. End-to-end training and competition-score reproduction require reconstructing the historical environment and obtaining the original checkpoints.

## Environment

For ZIP packaging and validation, use Python 3.10 or newer and:

```bash
python -m pip install -r requirements.txt
```

The historical VM reported Python 3.9.2, PyTorch 2.8.0+cu128, NumPy 2.0.2, Albumentations 2.0.8, Pillow 11.3.0, OpenCV 4.13.0, and tqdm 4.67.3. These are reported versions, not a complete independently verified environment lock.

For research work, start with a compatible Python/PyTorch/CUDA environment and use the explicit packages in `requirements-research.txt` as a reconstruction starting point. CUDA wheel installation depends on the target machine. The source imports SAM's `segment_anything` and downloads DINOv2 through `torch.hub`; the original source revisions and cached backbone checksums are not recorded in this repository. Consult the [official SAM](https://github.com/facebookresearch/segment-anything) and [DINOv2](https://github.com/facebookresearch/dinov2) projects for their setup and licenses. The optional CRF experiment requires `pydensecrf` separately.

## Data Access and Layout

Request SBVPI, MASD, SynCROI, MOBIUS, and SynMOBIUS through the [SSBC organizers](https://sites.google.com/hyderabad.bits-pilani.ac.in/ssbc2026/home) and complete the relevant licensing agreements. Keep T1 and T2 training versions separate. Do not use the T1 labels to train the T2 method.

The historical layout is:

```text
/home/jupyter/sclera/
├── data/
│   ├── T1 Training/
│   │   ├── SynCROI_1/{Images,Masks}/
│   │   ├── SBVPI/{Images,Masks}/
│   │   └── MASD_1/{Images,Masks}/
│   └── T2 Training/
│       ├── SynCROI/{Images,Masks}/
│       ├── SBVPI/{Images,Masks}/
│       ├── MASD/{Images,Masks}/
│       └── MOBIUS/Images/
├── models/
├── outputs/
├── MOBIUS/                   # Evaluation images, separate from unlabelled training data
├── SMD+SLD/
└── SynMOBIUS/
```

Confirm paths against the actual archive you receive. `build_pairs` and `get_mask_path` implement the source's naming conventions; T1 SBVPI masks use `_sclera.png`, while T2 mask names match image stems. The limited-label T2 layout may contain images without a corresponding mask.

On another machine, edit the relevant config paths in a local working copy and inspect scripts that define their own absolute paths. They are not all controlled by one configuration file. The source snapshot is deliberately not advertised as a portable package.

## Track 1 Training

Obtain SAM's `sam_vit_b_01ec64.pth` checkpoint through the official SAM project and place it in the configured model directory. The historical principal trainer supports:

```bash
python research/train.py --mode synthetic --lora-rank 16 --lora-alpha 32
python research/train.py --mode mixed --lora-rank 16 --lora-alpha 32
```

These are dataset-selection experiments, not two official competition protocols. `mixed` uses the provided T1 synthetic and real training datasets. Review the data split, sampler, output paths, and existing checkpoints before a run. A new training run is not guaranteed to reproduce a recovered checkpoint exactly.

## Track 2 Training

The main limited-label runs are:

```bash
python research/t2_dino_train.py
python research/t2_dino_large_train.py
```

Full-data retraining follows once a configuration has been selected using validation data:

```bash
python research/t2_fulltrain.py
python research/t2_fulltrain.py --large
```

The main configs use validation splits for model selection; full-data runs use all labelled examples and fixed epoch counts. The trainer can select SWA weights for `best_model.pth`, so inspect checkpoint contents and provenance rather than assuming that name means a validation-best student.

## Historical Inference

The original model-selection paths in `research/submit.py` are:

| Role | Checkpoint under `/home/jupyter/sclera/models/` |
| --- | --- |
| SAM backbone initialization | `sam_vit_b_01ec64.pth` |
| T1 output retained for the corrected submission | `t1_synthetic/best_model.pth` |
| Separate T1 comparison run | `t1_mixed/best_model.pth` |
| T2 ViT-B, preferred | `t2_dino_full/best_model.pth` |
| T2 ViT-B, fallback | `t2_dino/best_model.pth` |
| T2 ViT-L, preferred | `t2_dino_large_full/best_model.pth` |
| T2 ViT-L, fallback | `t2_dino_large/best_model.pth` |

To inspect the historical path, the original commands were equivalent to:

```bash
python research/submit.py --track 1 --test-dir /path/to/Evaluation --no-zip
python research/submit.py --track 2 --test-dir /path/to/Evaluation --no-zip
```

These commands require CUDA and the original dependencies and checkpoints. They write the legacy output trees under `/home/jupyter/sclera/outputs/submission`. The code can skip missing checkpoints/datasets and can overwrite existing prediction files, so inspect a run's output carefully. Read [the known scaling and geometry issues](results.md#confirmed-source-inconsistencies) before using them as a basis for new experiments. The archive's printed expected validation scores do not measure performance on the evaluation images.

Use the new packager on one selected tree:

```bash
python scripts/package_submission.py \
  --source /home/jupyter/sclera/outputs/submission/T1/Synthetic \
  --test-dir /path/to/Evaluation \
  --output outputs/T1_SAM_LoRA.zip

python scripts/package_submission.py \
  --source /home/jupyter/sclera/outputs/submission/T2/Synthetic \
  --test-dir /path/to/Evaluation \
  --output outputs/T2_DINOv2_Ensemble.zip
```

The packager refuses to overwrite an existing output. It validates the mask folders before writing and the resulting ZIP afterwards. The validator compares expected names against `--test-dir`, including names with spaces or parentheses, and expects exactly 3,539 MOBIUS, 489 SMD+SLD, and 4,771 SynMOBIUS images. It requires probabilistic masks as well as binary masks for this team's complete submission.

For unit fixtures or deliberately smaller local datasets, `--allow-subset` uses the images in `--test-dir` instead of enforcing competition counts. This mode is not a full competition validation and is labelled as such in the output.

The organizer also supplies an official checker through the Participant Instructions. Run it separately before any future submission. Our tool's success message is intentionally different and does not claim organizer approval.

## Verify the Release

```bash
python -m unittest discover -s tests -v
```

The tests generate their own grayscale masks and images in temporary directories. They cover correct layout, filenames, counts, image shape/type, binary value range, unsupported wrapper directories, corrupt entries, and overwrite protection. No licensed images or model downloads are needed.
