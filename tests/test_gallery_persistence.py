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
    load_generation_batches,
    normalize_retention_days,
    save_generation_batch,
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

    def test_generation_batches_keep_prompt_and_images_after_gallery_reset(self):
        first = self._image("first.png", b"first")
        second = self._image("second.png", b"second")
        save_generation_batch(
            self.output_dir,
            batch_id="batch-1",
            prompt="最初のプロンプト",
            provider="gemini",
            images=[first, second],
            created_at=time.time(),
        )

        clear_gallery(self.output_dir)
        batches = load_generation_batches(self.output_dir)

        self.assertEqual(1, len(batches))
        self.assertEqual("最初のプロンプト", batches[0]["prompt"])
        self.assertEqual([first, second], batches[0]["images"])

    def test_generation_batches_are_newest_first_and_honor_image_cap(self):
        first = self._image("first.png", b"first")
        second = self._image("second.png", b"second")
        third = self._image("third.png", b"third")
        now = time.time()
        save_generation_batch(
            self.output_dir,
            batch_id="older",
            prompt="古いプロンプト",
            provider="gemini",
            images=[first, second],
            created_at=now - 10,
            max_images=10,
        )
        save_generation_batch(
            self.output_dir,
            batch_id="newer",
            prompt="新しいプロンプト",
            provider="openai",
            images=[third],
            created_at=now,
            max_images=10,
        )

        batches = load_generation_batches(self.output_dir, max_images=2)

        self.assertEqual(["newer", "older"], [batch["id"] for batch in batches])
        self.assertEqual([third], batches[0]["images"])
        self.assertEqual([first], batches[1]["images"])

    def test_retention_is_bounded(self):
        self.assertEqual(1, normalize_retention_days(0))
        self.assertEqual(7, normalize_retention_days("invalid"))
        self.assertEqual(30, normalize_retention_days(365))


if __name__ == "__main__":
    unittest.main()
