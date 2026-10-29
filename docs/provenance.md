# Source and Submission Provenance

## Team and Scope

This release is the Arogya Pandit team's research code for SSBC 2026. Team members are Gourav Gupta and Anant Gupta. The competition summary is a joint paper by the organizers and participating teams; this repository does not claim authorship of the other teams' methods.

`research/*.py` was recovered from the team's local SCLERA source folder. The original Python files are copied without changes. They contain older terminology, path assumptions, intermediate experiment comments, and validation scores that may differ from later documentation. The release's current instructions live in the root README and `docs/`.

[source_sha256.txt](source_sha256.txt) records SHA-256 checksums for all 37 archived Python files. From the repository root, verify them with `shasum -a 256 -c docs/source_sha256.txt` (macOS) or `sha256sum -c docs/source_sha256.txt` (Linux).

The public repository adds documentation, package metadata, submission packaging and validation tools, generated-fixture tests, and CI. Original internal planning notes, questionnaires, cloud setup commands, raw visualizations, and correspondence are excluded from this curated release.

## Submission Timeline

- In May 2026, the original pipeline generated output trees for both historical dataset-selection options. The old wrapper layout was carried forward from earlier instructions.
- The organizers clarified that SSBC 2026 has no separate Mixed and Synthetic submission protocols.
- The corrected T1 ZIP retained the predictions from `T1/Synthetic`, produced by the `t1_synthetic` checkpoint. The corrected T2 ZIP retained one output copy from `T2/Synthetic`; the old T2 regime loop used the same ensemble for both copies.
- The corrected ZIPs were named `T1_SAM_LoRA.zip` and `T2_DINOv2_Ensemble.zip`. Each contained 17,598 masks directly under the three required dataset folders.
- The organizer acknowledged receipt of the corrected submission on 8 May 2026 and circulated final results on 29 May 2026.
- A public version of the summary paper is now hosted by the University of Ljubljana organizer lab and listed as an IJCB 2026 proceedings article. See the links in the README.

This chronology is based on the team's source files and submission correspondence. The public paper is the source for aggregate benchmark results. Checkpoint hashes, the exact historical backbone revisions, and a complete environment lock are not available in this release. Exact numerical reproduction of the submission is therefore not claimed.

## Data and Redistribution

No biometric images, ground-truth masks, learned checkpoints, prediction ZIPs, personal access tokens, or private email exports are included. Data access is governed by the relevant dataset agreements. SAM and DINOv2 code and pretrained weights retain their upstream licenses; consult those projects before redistributing derived materials.
