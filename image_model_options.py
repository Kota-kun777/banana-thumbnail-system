"""Image model choices shown by the Streamlit app.

Defaults intentionally remain the models that were in use before the model
comparison UI was added.  Cheaper choices are opt-in.
"""

GEMINI_IMAGE_MODEL_DEFAULT = "gemini-3-pro-image-preview"
GEMINI_IMAGE_MODEL_OPTIONS = {
    GEMINI_IMAGE_MODEL_DEFAULT: "Gemini 3 Pro（現在の既定・高品質）",
    "gemini-3.1-flash-image": "Gemini 3.1 Flash（低コスト・バランス）",
    "gemini-3.1-flash-lite-image": "Gemini 3.1 Flash Lite（最安・1K）",
}

OPENAI_IMAGE_MODEL_DEFAULT = "gpt-image-2"
OPENAI_IMAGE_MODEL_OPTIONS = {
    OPENAI_IMAGE_MODEL_DEFAULT: "GPT Image 2（現在の既定）",
    "gpt-image-1-mini": "GPT Image 1 mini（低コスト・旧モデル）",
}

# GPT Image 2 supports flexible dimensions whose width and height are multiples
# of 16.  The legacy mini model uses the three documented fixed sizes.
OPENAI_IMAGE_2_SIZE_OPTIONS = [
    "2048x1152",   # 16:9 high resolution (current default)
    "1792x1008",   # 16:9 medium resolution
    "1536x864",    # 16:9 standard resolution
    "1024x576",    # 16:9 low resolution
    "1024x1024",   # 1:1
    "1024x1536",   # 2:3 portrait
]
OPENAI_LEGACY_SIZE_OPTIONS = [
    "1536x1024",   # 3:2; cropped to 16:9 by the existing default
    "1024x1024",
    "1024x1536",
]
OPENAI_QUALITY_OPTIONS = ["high", "medium", "low", "auto"]
OPENAI_QUALITY_LABELS = {
    "high": "high（高品質）",
    "medium": "medium（中品質・比較向け）",
    "low": "low（低コスト）",
    "auto": "auto（自動）",
}


def model_label(model_id: str) -> str:
    """Return the friendly label for a known model, or its raw model ID."""
    return (
        GEMINI_IMAGE_MODEL_OPTIONS.get(model_id)
        or OPENAI_IMAGE_MODEL_OPTIONS.get(model_id)
        or model_id
    )


def openai_size_options(model_id: str) -> list[str]:
    """Return supported UI sizes for the selected OpenAI image model."""
    if model_id == "gpt-image-1-mini":
        return list(OPENAI_LEGACY_SIZE_OPTIONS)
    return list(OPENAI_IMAGE_2_SIZE_OPTIONS)


def openai_supports_custom_size(model_id: str) -> bool:
    """Only GPT Image 2 exposes arbitrary multiple-of-16 dimensions here."""
    return model_id == OPENAI_IMAGE_MODEL_DEFAULT
