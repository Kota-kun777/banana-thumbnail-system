import os
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

from gallery_persistence import (
    MANIFEST_FILENAME,
    append_gallery_images,
    clear_gallery,
    create_gallery_zip,
    load_gallery,
    normalize_retention_days,
)


class GalleryPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "replica_output"
        self.output_dir.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _image(self, name: str, content: bytes = b"image") -> Path:
        path = self.output_dir / name
        path.write_bytes(content)
        return path

    def test_migrates_existing_images_and_prunes_expired_files(self):
        old_image = self._image("old.png")
        recent_image = self._image("recent.png")
        old_time = time.time() - 10 * 24 * 60 * 60
        os.utime(old_image, (old_time, old_time))

        loaded = load_gallery(
            self.output_dir,
            retention_days=7,
            max_images=200,
        )

        self.assertEqual([recent_image], loaded)
        self.assertFalse(old_image.exists())
        self.assertTrue((self.output_dir / MANIFEST_FILENAME).exists())

    def test_append_merges_updates_and_clear_starts_a_new_gallery(self):
        first = self._image("first.png", b"first")
        second = self._image("second.png", b"second")

        append_gallery_images(self.output_dir, [first], max_images=200)
        merged = append_gallery_images(self.output_dir, [second], max_images=200)
        self.assertEqual([first, second], merged)

        clear_gallery(self.output_dir)
        self.assertEqual([], load_gallery(self.output_dir, max_images=200))
        # Resetting the view is recoverable until normal retention cleanup.
        self.assertTrue(first.exists())
        self.assertTrue(second.exists())

    def test_recovers_an_image_that_finishes_after_the_tab_closes(self):
        # First page load creates an empty manifest.
        self.assertEqual([], load_gallery(self.output_dir, max_images=200))
        background_image = self._image("background-finished.png", b"done")

        reopened = load_gallery(self.output_dir, max_images=200)

        self.assertEqual([background_image], reopened)

    def test_zip_contains_only_selected_gallery_images(self):
        first = self._image("first.png", b"first")
        second = self._image("second.jpg", b"second")
        self._image("not-selected.png", b"other")

        archive_path = create_gallery_zip(
            self.output_dir,
            [first, second],
            archive_key="test-session",
        )

        with zipfile.ZipFile(archive_path) as archive:
            self.assertEqual(["first.png", "second.jpg"], archive.namelist())
            self.assertEqual(b"first", archive.read("first.png"))

    def test_retention_is_bounded(self):
        self.assertEqual(1, normalize_retention_days(0))
        self.assertEqual(7, normalize_retention_days("invalid"))
        self.assertEqual(30, normalize_retention_days(365))


if __name__ == "__main__":
    unittest.main()
