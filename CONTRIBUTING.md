# Contributing

Useful follow-up work includes restoring masks to original image coordinates, making checkpoint metadata explicit, verifying LoRA scaling against training, and testing subject-disjoint validation. Start by reading [results.md](docs/results.md) so that a proposed change addresses a documented issue.

Please open an issue describing the question, affected source file, and evidence. Include Python and dependency versions for runtime problems. Do not upload licensed images, masks, access credentials, or private model files to issues or pull requests.

Keep the `research/` source snapshot unchanged in routine documentation and tooling contributions. For a corrected model implementation, add a clearly named experiment or propose a separately documented version with controlled validation. Do not present new scores as the original competition submission.

Changes to `scripts/` should pass `python -m unittest discover -s tests -v`. Report GPU experiments separately with their data split, checkpoint choice, preprocessing, and metrics.
