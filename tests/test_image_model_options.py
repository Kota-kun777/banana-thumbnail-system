import unittest

from image_model_options import (
    GEMINI_IMAGE_MODEL_DEFAULT,
    GEMINI_IMAGE_MODEL_OPTIONS,
    OPENAI_IMAGE_MODEL_DEFAULT,
    OPENAI_IMAGE_MODEL_OPTIONS,
    model_label,
    openai_size_options,
    openai_supports_custom_size,
)


class ImageModelOptionsTests(unittest.TestCase):
    def test_existing_models_remain_defaults(self):
        self.assertEqual("gemini-3-pro-image-preview", GEMINI_IMAGE_MODEL_DEFAULT)
        self.assertEqual("gpt-image-2", OPENAI_IMAGE_MODEL_DEFAULT)
        self.assertEqual(
            GEMINI_IMAGE_MODEL_DEFAULT,
            next(iter(GEMINI_IMAGE_MODEL_OPTIONS)),
        )
        self.assertEqual(
            OPENAI_IMAGE_MODEL_DEFAULT,
            next(iter(OPENAI_IMAGE_MODEL_OPTIONS)),
        )

    def test_lower_cost_comparison_models_are_available(self):
        self.assertIn("gemini-3.1-flash-image", GEMINI_IMAGE_MODEL_OPTIONS)
        self.assertIn("gemini-3.1-flash-lite-image", GEMINI_IMAGE_MODEL_OPTIONS)
        self.assertIn("gpt-image-1-mini", OPENAI_IMAGE_MODEL_OPTIONS)

    def test_legacy_openai_model_uses_only_fixed_sizes(self):
        self.assertEqual(
            ["1536x1024", "1024x1024", "1024x1536"],
            openai_size_options("gpt-image-1-mini"),
        )
        self.assertFalse(openai_supports_custom_size("gpt-image-1-mini"))
        self.assertTrue(openai_supports_custom_size("gpt-image-2"))

    def test_unknown_model_label_falls_back_to_raw_id(self):
        self.assertEqual("future-model", model_label("future-model"))


if __name__ == "__main__":
    unittest.main()
