# Official Results and Limitations

## Official Evaluation

The public competition paper reports these aggregate values in Tables 3 and 4. F1 rankings use a harmonic mean across MOBIUS, SMD+SLD, and SynMOBIUS; do not replace these values with arithmetic means of the dataset scores.

| Entry | Rank | F1 | Precision | Recall | IoU | Optimal F1 | AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SAM-LoRA, T1 | 10 / 10 | 0.098 | 0.058 | 0.398 | 0.054 | 0.210 | 0.113 |
| DINOv2 Ensemble, T2 | 6 / 7 | 0.465 | 0.346 | 0.854 | 0.314 | 0.604 | 0.646 |

The organizer's May 2026 results email also supplied the following per-dataset figures:

| Entry | Dataset | F1 | Precision | Recall | IoU | Optimal F1 | AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SAM-LoRA | MOBIUS | 0.157 | 0.097 | 0.532 | 0.090 | 0.259 | 0.181 |
| SAM-LoRA | SMD+SLD | 0.055 | 0.032 | 0.249 | 0.029 | 0.148 | 0.062 |
| SAM-LoRA | SynMOBIUS | 0.172 | 0.104 | 0.611 | 0.099 | 0.273 | 0.207 |
| DINOv2 Ensemble | MOBIUS | 0.448 | 0.312 | 0.875 | 0.295 | 0.567 | 0.612 |
| DINOv2 Ensemble | SMD+SLD | 0.498 | 0.430 | 0.815 | 0.356 | 0.640 | 0.679 |
| DINOv2 Ensemble | SynMOBIUS | 0.453 | 0.318 | 0.874 | 0.299 | 0.608 | 0.651 |

Sources: [public summary paper](https://lmi.fe.uni-lj.si/wp-content/uploads/2026/09/IJCB_SSBC_2026.pdf), Tables 3-4; [competition site](https://sites.google.com/hyderabad.bits-pilani.ac.in/ssbc2026/home); final-results correspondence supplied by the team. This repository does not redistribute that email or private attachments.

## Internal Validation Is a Separate Measurement

The historical scripts report internal validation F1 values around 0.979 for a T1 checkpoint and 0.9564 for a T2 ensemble configuration. They are recorded experiment outputs, not claims of independently reproduced results. The splits, data distributions, checkpoint selection, and metric aggregation differ from the organizer's sequestered evaluation.

The official results show that the approach did not generalize successfully in its submitted form. Track 1's very low precision and AUC are severe failures. Track 2's high recall with relatively low precision is consistent with substantial false-positive segmentation. Neither result establishes a single root cause by itself.

## Confirmed Source Inconsistencies

### Track 1 LoRA Scaling

`research/config.py` sets `LORA_ALPHA = 32`. Both `research/train.py` and `research/t1_eval.py` pass alpha 32 to `SAMLoRASeg`. In contrast, `research/submit.py:load_t1_model` passes the rank but omits alpha, so `SAMLoRASeg` uses its default alpha of 1.

`LoRALinear` computes a plain Python scale attribute as `alpha / rank`. That scale is not a parameter or buffer in the PyTorch state dictionary. Loading a checkpoint's state dictionary therefore does not restore the training alpha automatically. At rank 16, the configured scales are 2.0 versus 0.0625, a factor of 32 apart.

This is a concrete train/inference mismatch in the recovered source. We have not rerun the submitted checkpoint with both scales against organizer ground truth, so its contribution to the score remains unmeasured. The source snapshot is preserved without a silent change to the original defaults.

### Validation and Submission Geometry

Track 1 validation in `dataset.py` preserves aspect ratio with longest-side resizing and padding. The T1 path in `submit.py` instead stretches each image to a square. Track 2's validation transform pads images to a square, but the submission path resizes the predicted square directly to 400 x 300 without an explicit inverse padding step.

Those operations can alter mask alignment. A controlled follow-up should compare geometry on a held-out labelled set and restore predictions to the original image coordinates before the final submission resize. No corrected competition score is claimed here.

### Checkpoint Provenance

The full-data Track 2 trainer can save SWA weights under `student_state_dict`. The submission loader prefers those full-data paths, falling back to validation-split checkpoints. Filenames and the `student_state_dict` key do not fully specify the weights used; original logs and hashes would be needed to identify them unambiguously.

### Historical Submission Layout

The original submission code writes `Synthetic/` and `Mixed/` wrapper directories. SSBC 2026 requires only the three dataset directories at the ZIP root. The corrected competition files were repackaged before organizer acknowledgement. The new packaging tools implement the flat layout and validate every output; they do not improve segmentation accuracy.

## Reproduction Boundaries

The original checkpoints and complete run environment are not bundled. The release checks validate code syntax and submission utility behavior using generated fixtures, rather than retraining the models or reproducing the published benchmark numbers. Random image splits in the historical training scripts also do not establish subject-disjoint evaluation.

A polarity inversion was suggested during an earlier discussion because of the low T1 AUC. That was a hypothesis, not a verified diagnosis. Masks should not be inverted, thresholds selected, or models tuned based on these test results and then represented as the original submission.

The most useful next research experiment is an audit on training/validation data: checkpoint identity, LoRA alpha, image/mask correspondence, foreground convention, aspect-ratio restoration, and direct numerical parity between validation and export. Any such corrected experiment should report its own protocol and scores.
