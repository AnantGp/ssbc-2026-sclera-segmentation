"""Package one selected prediction set in the flat SSBC 2026 ZIP layout."""

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from submission_utils import expected_names, validate_archive, validate_directory


def package_submission(source, test_dir, output, allow_subset=False):
    source = Path(source).resolve()
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists; choose a new filename: {output}")
    if output.resolve().is_relative_to(source):
        raise ValueError("Output ZIP must be outside the source mask tree")
    expected = expected_names(test_dir, allow_subset=allow_subset)
    files = validate_directory(source, expected)
    output.parent.mkdir(parents=True, exist_ok=True)

    created = False
    try:
        with output.open("xb") as stream:
            created = True
            with ZipFile(stream, "w", compression=ZIP_DEFLATED) as zipped:
                for path, name in files:
                    zipped.write(path, name)
        count = validate_archive(output, expected)
    except Exception:
        if created:
            output.unlink()
        raise
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="One model's dataset mask folders")
    parser.add_argument("--test-dir", type=Path, required=True, help="Root containing evaluation datasets")
    parser.add_argument("--output", type=Path, required=True, help="New ZIP file; existing files are refused")
    parser.add_argument("--allow-subset", action="store_true", help="Use local fixture sizes, not official counts")
    args = parser.parse_args()
    try:
        count = package_submission(args.source, args.test_dir, args.output, args.allow_subset)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Packaging failed: {error}\n")
    scope = "subset only" if args.allow_subset else "full dataset counts"
    print(f"Packaged and locally validated {count} PNG masks ({scope}): {args.output}")


if __name__ == "__main__":
    main()
