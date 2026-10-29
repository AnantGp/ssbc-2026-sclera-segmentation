"""Check this team's full SSBC prediction contract independently of the organizer tool."""

import argparse
from pathlib import Path

from submission_utils import expected_names, validate_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Submission ZIP to inspect")
    parser.add_argument("--test-dir", type=Path, required=True, help="Root containing evaluation datasets")
    parser.add_argument("--allow-subset", action="store_true", help="Use local fixture sizes, not official counts")
    args = parser.parse_args()
    try:
        expected = expected_names(args.test_dir, allow_subset=args.allow_subset)
        count = validate_archive(args.archive, expected)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Validation failed: {error}\n")
    scope = "subset only" if args.allow_subset else "full dataset counts"
    print(f"Local validation passed: {count} PNG masks ({scope})")
    print("This does not replace the organizer's checker or assess segmentation accuracy.")


if __name__ == "__main__":
    main()
