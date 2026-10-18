import json

import pytest

from backend.app.translation import (
    CaptionTranslationError,
    TranslationBatch,
    TranslatedCaption,
    translate_hindi_texts_to_english,
)


class FakeTranslationModel:
    def __init__(self, *, omit_index: bool = False) -> None:
        self.calls: list[list[tuple[str, str]]] = []
        self.omit_index = omit_index

    def invoke(self, messages: list[tuple[str, str]]) -> TranslationBatch:
        self.calls.append(messages)
        captions = json.loads(messages[-1][1])
        translations = [
            TranslatedCaption(index=item["index"], text=f"English {item['index']}")
            for item in reversed(captions)
        ]
        if self.omit_index:
            translations.pop()
        return TranslationBatch(translations=translations)


def test_translation_preserves_input_order_and_original_blank_segments() -> None:
    model = FakeTranslationModel()

    translated = translate_hindi_texts_to_english(
        ["नमस्ते", "", "दुनिया"], model=model
    )

    assert translated == ["English 0", "", "English 2"]
    assert len(model.calls) == 1


def test_translation_batches_large_caption_lists(monkeypatch) -> None:
    monkeypatch.setattr("backend.app.translation.MAX_SEGMENTS_PER_BATCH", 2)
    model = FakeTranslationModel()

    translated = translate_hindi_texts_to_english(["एक", "दो", "तीन"], model=model)

    assert translated == ["English 0", "English 1", "English 2"]
    assert len(model.calls) == 2


def test_translation_rejects_missing_caption_indices() -> None:
    with pytest.raises(CaptionTranslationError, match="missing"):
        translate_hindi_texts_to_english(
            ["पहला", "दूसरा"], model=FakeTranslationModel(omit_index=True)
        )


def test_translation_skips_model_for_empty_or_blank_input() -> None:
    assert translate_hindi_texts_to_english([]) == []
    assert translate_hindi_texts_to_english(["", "  "]) == ["", ""]
