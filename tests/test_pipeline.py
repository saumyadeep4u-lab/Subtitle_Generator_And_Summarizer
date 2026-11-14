from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.config_loader import load_config
from src.evaluator import PipelineEvaluator
from src.srt_formatter import SRTFormatter
from src.summarizer import LectureSummarizer
from src.topic_segmenter import TopicSegmenter


@pytest.fixture(scope="module")
def config():
    return load_config("config.yaml")


@pytest.fixture(scope="module")
def segmenter(config):
    return TopicSegmenter(config)


def test_config_parameters_loaded(config):
    """Verifies critical config.yaml keys exist and match project guidelines."""
    assert config["audio_extraction"]["target_sample_rate"] == 16000
    assert config["srt_formatting"]["max_chars_per_line"] == 42
    assert config["summarization"]["video_summary"]["max_words_limit"] == 100
    assert config["summarization"]["video_summary"]["target_min_sentences"] == 3
    assert config["summarization"]["video_summary"]["target_max_sentences"] == 5


def test_srt_timestamp_formatting():
    """Validates seconds to strict SRT timestamp conversion (HH:MM:SS,mmm)."""
    assert SRTFormatter.format_timestamp(0.0) == "00:00:00,000"
    assert SRTFormatter.format_timestamp(65.432) == "00:01:05,432"
    assert SRTFormatter.format_timestamp(3661.005) == "01:01:01,005"


def test_srt_generation_overlap_and_cleanup(config):
    """Tests that overlapping timestamps are resolved and cleans up temporary SRT file."""
    formatter = SRTFormatter(config)
    payload = {
        "video_id": "unit_test_video",
        "segments": [
            {"start": 0.0, "end": 4.0, "text": "Welcome to computer science."},
            {
                "start": 3.5,
                "end": 8.0,
                "text": "Overlapping timestamp test block.",
            },
        ],
    }
    res = formatter.build_srt(payload, "temp_unit_test.srt")
    srt_file = Path(res["srt_path"])
    try:
        assert res["validation"]["is_valid"] is True
        assert res["num_cues"] >= 2
    finally:
        if srt_file.exists():
            srt_file.unlink()


def test_srt_long_line_wrapping_and_empty_input(config):
    """Ensures extremely long single sentences wrap to <=42 chars and empty inputs fail safely."""
    formatter = SRTFormatter(config)
    long_text = (
        "In this lecture on deep learning architectures we examine how "
        "transformer self-attention layers process long-range dependencies "
        "across sequential tokens without recurrent bottlenecks."
    )
    payload = {
        "video_id": "wrap_test",
        "segments": [{"start": 0.0, "end": 12.0, "text": long_text}],
    }
    res = formatter.build_srt(payload, "temp_wrap_test.srt")
    srt_file = Path(res["srt_path"])
    try:
        assert res["validation"]["is_valid"] is True
        assert res["num_cues"] >= 2  # Split due to length and max_duration_sec
    finally:
        if srt_file.exists():
            srt_file.unlink()

    empty_check = formatter.validate_srt_content("")
    assert empty_check["is_valid"] is False


def test_topic_segmenter_token_bounds_and_overlap(segmenter):
    """Ensures TopicSegmenter enforces max_chunk_tokens and handles empty transcripts."""
    sample = {
        "video_id": "bound_test",
        "segments": [
            {
                "start": i * 2.0,
                "end": (i + 1) * 2.0,
                "text": (
                    f"Sentence {i} covers backpropagation, stochastic gradient "
                    "descent, and convolutional neural network optimization."
                ),
            }
            for i in range(1, 90)
        ],
    }
    sections = segmenter.segment_transcript(sample)
    assert len(sections) >= 2
    for sec in sections:
        assert sec["token_count"] <= segmenter.max_chunk_tokens

    assert segmenter.segment_transcript({"video_id": "empty", "segments": []}) == []


def test_summarizer_100_word_and_sentence_constraints(config):
    """Verifies that video summaries strictly obey <=100 words and 3-5 sentences."""
    summarizer = LectureSummarizer(config, model_override="lexrank")
    verbose_summary = " ".join(
        [
            f"Sentence {i} explains how neural networks optimize weights using gradient descent and loss functions across training epochs."
            for i in range(1, 12)
        ]
    )
    enforced = summarizer._enforce_video_constraints(verbose_summary)
    assert enforced["word_count"] <= 100
    assert 3 <= enforced["sentence_count"] <= 5
    assert enforced["post_processing_triggered"] is True


def test_evaluator_wer_and_rouge_metrics(config):
    """Validates WER, CER, normalization, ROUGE calculation, and empty-reference guard."""
    evaluator = PipelineEvaluator(config)
    asr_res = evaluator.evaluate_asr(
        "We can't optimize non-convex functions easily!",
        "we cannot optimize non convex functions easily",
    )
    assert asr_res["wer"] == 0.0  # Contraction & punctuation normalization should match

    rouge_res = evaluator.evaluate_summary(
        "Backpropagation computes gradients for neural network weights.",
        "Backpropagation calculates gradients to update neural network weights.",
        compute_bertscore=False,
    )
    assert 0.0 < rouge_res["rouge1_f1"] <= 1.0
    assert 0.0 < rouge_res["rougeL_f1"] <= 1.0

    with pytest.raises(ValueError):
        evaluator.evaluate_asr("   ", "hypothesis text")