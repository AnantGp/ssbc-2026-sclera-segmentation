# Methods and Experiment Map

## Track 1: Parameter-Efficient SAM Adaptation

The goal was to turn a general segmentation foundation model into a prompt-free sclera segmenter while updating a limited part of the network.

`research/model.py` defines `SAMLoRASeg`. It constructs the SAM ViT-B image encoder, freezes the base parameters, and wraps the attention QKV projection, attention projection, and both MLP linear layers in each block with `LoRALinear`.

For a frozen linear layer W, the adapter computes:

```text
y = W x + (alpha / rank) B A x
```

The principal configuration specifies rank 16 and training alpha 32. The trainable decoder uses ConvTranspose2d, BatchNorm, GELU, and convolutional refinement to map 64 x 64 encoder features to a 256 x 256 single-channel logit map. The base decoder in `model.py` has no dropout; dropout belongs to the exploratory `model_v4.py` implementation.

Training uses RGB floats in the 0-255 range; the model applies SAM's pixel mean and standard deviation internally. `train.py` minimizes `0.3 * BCE + 0.7 * Dice`, with AdamW, a cosine schedule, gradient accumulation, and a fixed seed of 42. The main configuration uses 1024 input resolution, learning rate 5e-5, weight decay 1e-4, and 50 epochs.

The source retains `synthetic` and `mixed` training options, referring to dataset selection in local experiments. These are not official SSBC 2026 submission protocols. The corrected final Track 1 ZIP was made from the historical `T1/Synthetic` output tree, corresponding to `models/t1_synthetic/best_model.pth`. The mixed checkpoint was a separate experiment and was not combined with it in the corrected submission.

The historical submission path resizes directly to 1024 x 1024, applies sigmoid, resizes probabilities to 400 x 300, and thresholds at 0.50. There is no TTA on the final Track 1 path. The LoRA loader and geometry caveats are detailed in [results.md](results.md).

## Track 2: DINOv2 Mean Teacher

The limited-label track uses the T2 dataset versions only. The documented labelled subset has 357 examples: 82 MASD, 55 SBVPI, and 220 SynCROI images. Unlabelled MOBIUS images and maskless images from the other T2 datasets provide the additional training signal.

Two models are trained separately:

| Setting | ViT-B/14 | ViT-L/14 |
| --- | --- | --- |
| Intermediate block indices, zero-based | 2, 5, 8, 11 | 5, 11, 17, 23 |
| Input size | 518 x 518 | 518 x 518 |
| Decoder width | 256 | 256 |
| Base learning rate | 1e-3 | 5e-4 |
| Labelled / unlabelled batch size | 4 / 8 | 2 / 4 |
| Accumulation steps | 4 | 8 |
| Epochs | 150 | 120 |
| SWA start epoch | 120 | 90 |

Each backbone supplies four feature maps. Lateral projections combine them, and an FPN-style decoder with GroupNorm and GELU progressively upsamples the features into segmentation logits. Backbone weights are frozen except LayerNorm parameters in the principal training configuration. Decoder and lateral projection parameters are trainable.

The student receives supervised Dice + BCE on labelled data. On unlabelled data, weak and strong views feed the teacher and student. A confidence filter with threshold 0.95 gates an MSE consistency term. Its weight ramps up over 20 epochs. The teacher follows the student's parameters by EMA:

```text
teacher <- 0.999 * teacher + 0.001 * student
loss = supervised_loss + ramped_weight * consistency_loss
```

The two backbones use separate training runs. `t2_fulltrain.py` later uses all labelled examples without a validation holdout. Its `best_model.pth` may contain SWA weights under the `student_state_dict` key; the key's name alone does not identify the weight-selection procedure.

Inference averages the two model probabilities. Each model uses scales 0.75, 1.0, and 1.25, with and without horizontal flipping. Target dimensions are rounded down to multiples of the DINOv2 patch size, 14. The result is resized to the submission dimensions and thresholded at 0.50.

## Experiment Map

The files in `research/` are the recovered source snapshot, not a single unified library. The following map is the recommended reading order.

| Purpose | Files |
| --- | --- |
| Track 1 base architecture and training | `model.py`, `dataset.py`, `config.py`, `train.py` |
| Track 1 validation and prediction | `t1_eval.py`, `evaluate.py`, `inference.py` |
| Track 2 final architecture | `t2_dino_model.py`, `t2_dataset.py` |
| Track 2 principal training | `t2_dino_config.py`, `t2_dino_train.py` |
| Track 2 large backbone | `t2_dino_large_config.py`, `t2_dino_large_train.py` |
| Track 2 full-data retraining | `t2_fulltrain.py` |
| Final historical inference and model selection | `submit.py` |
| Ensemble and TTA comparisons | `t2_ensemble_eval.py`, `t2_enhanced_tta_eval.py`, `t2_dino_eval.py`, `t2_dino_large_eval.py` |
| Earlier EfficientNet/U-Net Mean Teacher baseline | `t2_model.py`, `t2_train.py`, `t2_config.py`, `t2_inference.py` |
| Supervised baseline | `baseline_train.py`, `baseline_evaluate.py` |
| Exploratory Track 1 regularization and adaptation | `model_v4.py`, `dataset_v4.py`, `train_v4.py`, `tta_inference.py` |
| Exploratory Track 2 distillation and postprocessing | `t2_distill.py`, `t2_distill_v2.py`, `t2_optimize.py`, `t2_crf_eval.py`, `t2_dino_lora.py` |
| Historical diagnostics and visualizations | `smoke_test.py`, `visualize.py`, `t2_dino_inference.py` |

Exploratory scripts may require extra packages and contain their own paths. Their presence does not mean they were used for the final submitted predictions. The release utilities in `scripts/` are the recommended way to package and validate masks.
