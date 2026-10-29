"""Validate this team's complete SSBC 2026 prediction layout."""

from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

from PIL import Image, UnidentifiedImageError


DATASET_COUNTS = {"MOBIUS": 3539, "SMD+SLD": 489, "SynMOBIUS": 4771}
MASK_KINDS = ("Predictions", "Binarised")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_SIZE = (400, 300)


class SubmissionError(ValueError):
    """A dataset or prediction file does not satisfy the submission contract."""


def expected_names(test_dir, allow_subset=False):
    """Get expected output names and check dataset sizes and stem collisions."""
    test_dir = Path(test_dir)
    expected = {}
    for dataset, official_count in DATASET_COUNTS.items():
        dataset_dir = test_dir / dataset
        image_dir = dataset_dir / "Images"
        if not image_dir.is_dir():
            image_dir = dataset_dir
        if not image_dir.is_dir():
            raise SubmissionError(f"Missing evaluation dataset: {dataset_dir}")

        images = sorted(
            path for path in image_dir.rglob("*")
            if path.is_file()
            and not path.name.startswith("._")
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        names = {path.stem + ".png" for path in images}
        if len(names) != len(images):
            raise SubmissionError(f"{dataset}: duplicate stems would overwrite masks")
        if not names:
            raise SubmissionError(f"{dataset}: evaluation dataset is empty")
        if not allow_subset and len(names) != official_count:
            raise SubmissionError(
                f"{dataset}: expected {official_count} evaluation images, got {len(names)}"
            )
        expected[dataset] = names
    return expected


def validate_mask(payload, name, kind):
    """Decode every mask and check format, geometry, mode, and binary values."""
    try:
        with Image.open(BytesIO(payload)) as mask:
            if mask.format != "PNG":
                raise SubmissionError(f"{name}: image encoding must be PNG")
            if mask.size != MASK_SIZE:
                raise SubmissionError(f"{name}: expected size {MASK_SIZE}, got {mask.size}")
            if mask.mode != "L":
                raise SubmissionError(f"{name}: expected 8-bit grayscale mode L, got {mask.mode}")
            mask.verify()
        with Image.open(BytesIO(payload)) as mask:
            mask.load()
            if kind == "Binarised":
                values = {value for _, value in mask.getcolors(maxcolors=256)}
                if not values <= {0, 255}:
                    raise SubmissionError(f"{name}: binary masks must contain only 0 and 255")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        if isinstance(error, SubmissionError):
            raise
        raise SubmissionError(f"{name}: cannot decode mask: {error}") from error


def validate_filenames(actual, expected, location):
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise SubmissionError(
            f"{location}: {len(missing)} missing, {len(extra)} unexpected masks; "
            f"first missing={missing[:3]}, first unexpected={extra[:3]}"
        )


def validate_directory(source, expected):
    """Return source/member pairs only after all folders and images pass."""
    source = Path(source)
    if not source.is_dir():
        raise SubmissionError(f"Missing mask source: {source}")
    if {path.name for path in source.iterdir()} != set(DATASET_COUNTS):
        raise SubmissionError(
            "Source root must contain only MOBIUS, SMD+SLD, and SynMOBIUS directories"
        )

    files = []
    for dataset in DATASET_COUNTS:
        dataset_dir = source / dataset
        if dataset_dir.is_symlink() or not dataset_dir.is_dir():
            raise SubmissionError(f"{dataset}: expected a regular dataset directory")
        if {path.name for path in dataset_dir.iterdir()} != set(MASK_KINDS):
            raise SubmissionError(f"{dataset}: expected only Predictions and Binarised folders")
        for kind in MASK_KINDS:
            folder = dataset_dir / kind
            if folder.is_symlink() or not folder.is_dir():
                raise SubmissionError(f"{folder}: expected a regular mask directory")
            entries = list(folder.iterdir())
            if any(not path.is_file() or path.is_symlink() for path in entries):
                raise SubmissionError(f"{folder}: nested directories and symlinks are not allowed")
            validate_filenames({path.name for path in entries}, expected[dataset], str(folder))
            for path in sorted(entries):
                member = f"{dataset}/{kind}/{path.name}"
                validate_mask(path.read_bytes(), member, kind)
                files.append((path, member))
    return files


def validate_archive(archive, expected):
    """Validate ZIP members, exact file correspondence, CRCs, and all masks."""
    allowed_dirs = set(DATASET_COUNTS)
    allowed_dirs.update(f"{dataset}/{kind}" for dataset in DATASET_COUNTS for kind in MASK_KINDS)
    actual = {(dataset, kind): set() for dataset in DATASET_COUNTS for kind in MASK_KINDS}
    seen = set()
    total = 0
    try:
        with ZipFile(archive) as zipped:
            for info in zipped.infolist():
                name = info.filename
                if name in seen:
                    raise SubmissionError(f"Duplicate ZIP member: {name}")
                seen.add(name)
                path = PurePosixPath(name)
                if (
                    path.is_absolute()
                    or "\\" in name
                    or ".." in path.parts
                    or path.as_posix() != name.rstrip("/")
                ):
                    raise SubmissionError(f"Invalid ZIP path: {name}")
                if info.is_dir():
                    if name.rstrip("/") not in allowed_dirs:
                        raise SubmissionError(f"Unexpected ZIP directory: {name}")
                    continue
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise SubmissionError(f"ZIP symlinks are not allowed: {name}")
                parts = path.parts
                if (
                    len(parts) != 3
                    or parts[0] not in DATASET_COUNTS
                    or parts[1] not in MASK_KINDS
                    or path.suffix != ".png"
                ):
                    raise SubmissionError(f"Unexpected ZIP member: {name}")
                dataset, kind, filename = parts
                validate_mask(zipped.read(info), name, kind)
                actual[dataset, kind].add(filename)
                total += 1
    except (BadZipFile, RuntimeError, EOFError) as error:
        raise SubmissionError(f"Cannot read ZIP archive: {error}") from error

    for (dataset, kind), names in actual.items():
        validate_filenames(names, expected[dataset], f"{dataset}/{kind}")
    return total
