"""Exercise the release tools using generated images; no biometric data required."""

from io import BytesIO
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
import warnings
from zipfile import ZipFile

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from package_submission import package_submission
from submission_utils import (
    DATASET_COUNTS, MASK_KINDS, SubmissionError, expected_names, validate_archive,
)


def png_bytes(mode="L", size=(400, 300), value=0, fmt="PNG"):
    stream = BytesIO()
    Image.new(mode, size, value).save(stream, format=fmt)
    return stream.getvalue()


class SubmissionToolsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.test_dir = self.root / "evaluation"
        self.source = self.root / "masks"
        self.output = self.root / "submission.zip"
        self.filename = "sample (1).png"
        for dataset in DATASET_COUNTS:
            images = self.test_dir / dataset / "Images"
            images.mkdir(parents=True)
            Image.new("RGB", (8, 6)).save(images / self.filename)
            for kind in MASK_KINDS:
                folder = self.source / dataset / kind
                folder.mkdir(parents=True)
                value = 128 if kind == "Predictions" else 255
                (folder / self.filename).write_bytes(png_bytes(value=value))
        self.expected = expected_names(self.test_dir, allow_subset=True)

    def package(self):
        return package_submission(self.source, self.test_dir, self.output, allow_subset=True)

    def write_archive(self, mutation):
        entries = []
        for dataset in DATASET_COUNTS:
            for kind in MASK_KINDS:
                name = f"{dataset}/{kind}/{self.filename}"
                entries.append((name, (self.source / name).read_bytes()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with ZipFile(self.output, "w") as zipped:
                for name, data in mutation(entries):
                    zipped.writestr(name, data)

    def test_valid_round_trip(self):
        self.assertEqual(self.package(), 6)
        self.assertEqual(validate_archive(self.output, self.expected), 6)
        with ZipFile(self.output) as zipped:
            self.assertEqual({n.split("/")[0] for n in zipped.namelist()}, set(DATASET_COUNTS))

    def test_cli_labels_subset_and_validates(self):
        package = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts/package_submission.py"),
             "--source", str(self.source), "--test-dir", str(self.test_dir),
             "--output", str(self.output), "--allow-subset"],
            capture_output=True, text=True,
        )
        self.assertEqual(package.returncode, 0, package.stderr)
        validate = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts/validate_submission.py"),
             str(self.output), "--test-dir", str(self.test_dir), "--allow-subset"],
            capture_output=True, text=True,
        )
        self.assertEqual(validate.returncode, 0, validate.stderr)
        self.assertIn("Local validation passed", validate.stdout)
        self.assertIn("subset only", validate.stdout)

    def test_official_counts_are_default(self):
        with self.assertRaisesRegex(SubmissionError, "expected 3539"):
            expected_names(self.test_dir)

    def test_missing_dataset_rejected(self):
        with self.assertRaises(SubmissionError):
            expected_names(self.root / "missing", allow_subset=True)

    def test_duplicate_image_stems_rejected(self):
        Image.new("RGB", (8, 6)).save(self.test_dir / "MOBIUS/Images/sample (1).jpg")
        with self.assertRaisesRegex(SubmissionError, "duplicate stems"):
            expected_names(self.test_dir, allow_subset=True)

    def test_rgb_masks_rejected(self):
        (self.source / "MOBIUS/Predictions" / self.filename).write_bytes(png_bytes(mode="RGB"))
        with self.assertRaisesRegex(SubmissionError, "mode L"):
            self.package()
        self.assertFalse(self.output.exists())

    def test_wrong_dimensions_rejected(self):
        (self.source / "MOBIUS/Predictions" / self.filename).write_bytes(png_bytes(size=(300, 400)))
        with self.assertRaisesRegex(SubmissionError, "expected size"):
            self.package()

    def test_nonbinary_values_rejected(self):
        (self.source / "MOBIUS/Binarised" / self.filename).write_bytes(png_bytes(value=127))
        with self.assertRaisesRegex(SubmissionError, "only 0 and 255"):
            self.package()

    def test_jpeg_disguised_as_png_rejected(self):
        (self.source / "MOBIUS/Predictions" / self.filename).write_bytes(png_bytes(fmt="JPEG"))
        with self.assertRaisesRegex(SubmissionError, "encoding must be PNG"):
            self.package()

    def test_missing_mask_rejected(self):
        (self.source / "MOBIUS/Binarised" / self.filename).unlink()
        with self.assertRaisesRegex(SubmissionError, "1 missing"):
            self.package()

    def test_wrong_filename_with_correct_count_rejected(self):
        folder = self.source / "MOBIUS/Binarised"
        (folder / self.filename).rename(folder / "wrong.png")
        with self.assertRaisesRegex(SubmissionError, "1 missing, 1 unexpected"):
            self.package()

    def test_nested_mask_directory_rejected(self):
        (self.source / "MOBIUS/Predictions/extra").mkdir()
        with self.assertRaisesRegex(SubmissionError, "nested directories"):
            self.package()

    def test_existing_output_is_preserved(self):
        self.output.write_bytes(b"existing file")
        with self.assertRaises(FileExistsError):
            self.package()
        self.assertEqual(self.output.read_bytes(), b"existing file")

    def test_legacy_source_wrapper_rejected(self):
        wrapped = self.root / "legacy"
        (wrapped / "Synthetic").mkdir(parents=True)
        with self.assertRaisesRegex(SubmissionError, "Source root"):
            package_submission(wrapped, self.test_dir, self.output, allow_subset=True)

    def test_duplicate_zip_member_rejected(self):
        self.write_archive(lambda entries: entries + [entries[0]])
        with self.assertRaisesRegex(SubmissionError, "Duplicate ZIP member"):
            validate_archive(self.output, self.expected)

    def test_legacy_zip_wrapper_rejected(self):
        self.write_archive(lambda entries: [(f"Synthetic/{n}", d) for n, d in entries])
        with self.assertRaisesRegex(SubmissionError, "Unexpected ZIP member"):
            validate_archive(self.output, self.expected)

    def test_extra_zip_file_rejected(self):
        self.write_archive(lambda entries: entries + [("README.txt", b"unexpected")])
        with self.assertRaisesRegex(SubmissionError, "Unexpected ZIP member"):
            validate_archive(self.output, self.expected)

    def test_corrupt_png_rejected(self):
        self.write_archive(lambda entries: [(entries[0][0], b"bad PNG")] + entries[1:])
        with self.assertRaisesRegex(SubmissionError, "cannot decode mask"):
            validate_archive(self.output, self.expected)

    def test_all_masks_checked_including_last_member(self):
        self.write_archive(lambda entries: entries[:-1] + [(entries[-1][0], b"bad PNG")])
        with self.assertRaisesRegex(SubmissionError, "cannot decode mask"):
            validate_archive(self.output, self.expected)

    def test_broken_zip_rejected(self):
        self.output.write_bytes(b"not a ZIP archive")
        with self.assertRaisesRegex(SubmissionError, "Cannot read ZIP archive"):
            validate_archive(self.output, self.expected)

    def test_zip_crc_corruption_rejected(self):
        self.write_archive(lambda entries: entries)
        payload = self.output.read_bytes()
        original = png_bytes(value=128)
        offset = payload.index(original)
        corrupted = payload[:offset] + bytes([payload[offset] ^ 1]) + payload[offset + 1:]
        self.output.write_bytes(corrupted)
        with self.assertRaisesRegex(SubmissionError, "Cannot read ZIP archive"):
            validate_archive(self.output, self.expected)

    def test_binary_mask_containing_both_endpoints_is_valid(self):
        mask = Image.new("L", (400, 300), 0)
        mask.putpixel((200, 150), 255)
        mask.save(self.source / "MOBIUS/Binarised" / self.filename)
        self.assertEqual(self.package(), 6)

    def test_output_inside_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the source mask tree"):
            package_submission(self.source, self.test_dir, self.source / "out.zip", allow_subset=True)

    def test_path_traversal_rejected(self):
        self.write_archive(lambda entries: entries + [("../out.png", png_bytes())])
        with self.assertRaisesRegex(SubmissionError, "Invalid ZIP path"):
            validate_archive(self.output, self.expected)

    def test_valid_explicit_zip_directories(self):
        self.package()
        with ZipFile(self.output, "a") as zipped:
            for dataset in DATASET_COUNTS:
                zipped.writestr(f"{dataset}/", b"")
                for kind in MASK_KINDS:
                    zipped.writestr(f"{dataset}/{kind}/", b"")
        self.assertEqual(validate_archive(self.output, self.expected), 6)


if __name__ == "__main__":
    unittest.main()
