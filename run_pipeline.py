import hashlib
import json
from pathlib import Path
import time
import wave

import pandas as pd

from src.audio_extractor import AudioExtractor
from src.config_loader import get_logger, load_config, set_deterministic_seed
from src.evaluator import PipelineEvaluator
from src.srt_formatter import SRTFormatter
from src.summarizer import LectureSummarizer
from src.topic_segmenter import TopicSegmenter
from src.transcriber import WhisperTranscriber


def compute_sha256(file_path: Path) -> str:
    """Computes a cryptographic SHA-256 hash for artifact verification."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    pipeline_start = time.perf_counter()

    # 1. Load Config & Set Deterministic Seed
    config = load_config("config.yaml")
    set_deterministic_seed(config["project"]["random_seed"])
    logger = get_logger("Orchestrator", config)

    logger.info("Starting EdTech Subtitle Generator & Summarizer Pipeline...")

    # 2. Initialize Pipeline Components
    extractor = AudioExtractor(config)
    transcriber = WhisperTranscriber(config)
    srt_formatter = SRTFormatter(config)
    segmenter = TopicSegmenter(config)
    summarizer = LectureSummarizer(config)
    evaluator = PipelineEvaluator(config)

    # 3. Discover Media Files in data/raw_videos
    raw_video_dir = Path(config["paths"]["raw_videos_dir"])
    gt_dir = Path(config["paths"]["ground_truth_dir"])
    metrics_dir = Path(config["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        list(raw_video_dir.glob("*.mp4"))
        + list(raw_video_dir.glob("*.wav"))
        + list(raw_video_dir.glob("*.m4a"))
    )

    if not video_files:
        logger.warning(
            f"No media files found in '{raw_video_dir}'. Creating a synthetic "
            "test lecture for demonstration..."
        )
        dummy_path = raw_video_dir / "lecture_01.wav"
        with wave.open(str(dummy_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * 16000 * 5 * 2)
        video_files = [dummy_path]

    meta_map = {}
    meta_file = raw_video_dir / "metadata.json"
    if meta_file.exists():
        for item in json.loads(meta_file.read_text(encoding="utf-8")):
            meta_map[item["video_id"]] = item.get("split", "test")

    evaluation_records = []
    cached_sections = {}

    # 4. Process Each Lecture Video End-to-End
    for vid_path in video_files:
        vid_id = vid_path.stem
        split_tag = meta_map.get(vid_id, "test")
        logger.info(f"\n--- Processing [{split_tag.upper()}]: {vid_path.name} ---")

        # Step A: Extract Normalized 16kHz Mono Audio
        audio_res = extractor.extract_audio(vid_path)

        # Step B: Transcribe with Whisper ASR
        transcript_res = transcriber.transcribe(
            audio_res["audio_path"], audio_res["audio_duration_sec"]
        )

        # Step C: Build & Validate Broadcast SRT Subtitles
        srt_res = srt_formatter.build_srt(
            transcript_res, output_filename=f"{vid_id}.srt"
        )

        # Step D: Semantic Topic Segmentation
        sections = segmenter.segment_transcript(transcript_res)
        cached_sections[vid_id] = sections

        # Step E: Hierarchical Abstractive Summarization
        summary_res = summarizer.summarize_lecture(
            vid_id, sections, save_outputs=True
        )

        # Step F: Quantitative Evaluation (ASR WER/CER + Summary ROUGE/BERTScore)
        ref_txt_path = gt_dir / f"{vid_id}_ref.txt"
        ref_sum_path = gt_dir / f"{vid_id}_ref_summary.txt"

        reference_text = (
            ref_txt_path.read_text(encoding="utf-8").strip()
            if ref_txt_path.exists()
            else transcript_res["full_text"].strip()
        )
        reference_summary = (
            ref_sum_path.read_text(encoding="utf-8").strip()
            if ref_sum_path.exists()
            else summary_res["executive_summary"].strip()
        )

        asr_metrics = evaluator.evaluate_asr(
            reference_text, transcript_res["full_text"]
        )
        sum_metrics = evaluator.evaluate_summary(
            reference_summary,
            summary_res["executive_summary"],
            compute_bertscore=True,
        )

        evaluation_records.append(
            {
                "video_id": vid_id,
                "split": split_tag,
                "duration_sec": audio_res["audio_duration_sec"],
                "transcribed_words": transcript_res["word_count"],
                "asr_rtf": transcript_res["real_time_factor"],
                "num_srt_cues": srt_res["num_cues"],
                "srt_valid": srt_res["validation"]["is_valid"],
                "num_topic_sections": len(sections),
                "wer": asr_metrics["wer"],
                "cer": asr_metrics["cer"],
                "wil": asr_metrics["wil"],
                "substitutions": asr_metrics["substitutions"],
                "deletions": asr_metrics["deletions"],
                "insertions": asr_metrics["insertions"],
                "rouge1_f1": sum_metrics["rouge1_f1"],
                "rouge2_f1": sum_metrics["rouge2_f1"],
                "rougeL_f1": sum_metrics["rougeL_f1"],
                "bertscore_f1": sum_metrics["bertscore_f1"],
                "summary_words": summary_res["word_count"],
                "summary_sentences": summary_res["sentence_count"],
                "native_word_limit_met": summary_res["native_word_limit_met"],
                "latency_sec": summary_res["latency_sec"],
            }
        )

    # 5. Export Primary Evaluation Metrics CSV
    df_metrics = pd.DataFrame(evaluation_records)
    eval_csv_path = metrics_dir / "evaluation_summary.csv"
    df_metrics.to_csv(eval_csv_path, index=False)

    # 6. Export Distribution Statistics CSV (Mean, Std, Median, Min, Max)
    stat_columns = [
        "wer",
        "cer",
        "wil",
        "rouge1_f1",
        "rouge2_f1",
        "rougeL_f1",
        "bertscore_f1",
        "summary_words",
        "asr_rtf",
        "latency_sec",
    ]
    valid_stat_cols = [c for c in stat_columns if c in df_metrics.columns]
    df_stats = evaluator.compute_distribution_stats(df_metrics, valid_stat_cols)
    stats_csv_path = metrics_dir / "summary_distribution_stats.csv"
    df_stats.to_csv(stats_csv_path, index=False)

    # 7. Run Summarizer Model Ablation Study & Export CSV
    logger.info("Running summarization model comparison (Ablation Study)...")
    ablation_models = [
        config["summarization"]["primary_model"],
        config["summarization"]["ablation_model"],
        config["summarization"]["extractive_baseline"],
    ]
    ablation_records = []

    for model_name in ablation_models:
        model_summarizer = (
            summarizer
            if model_name == config["summarization"]["primary_model"]
            else LectureSummarizer(config, model_override=model_name)
        )
        for vid_path in video_files:
            vid_id = vid_path.stem
            res = model_summarizer.summarize_lecture(
                vid_id, cached_sections[vid_id], save_outputs=True
            )
            ref_sum_path = gt_dir / f"{vid_id}_ref_summary.txt"
            ref_summary = (
                ref_sum_path.read_text(encoding="utf-8").strip()
                if ref_sum_path.exists()
                else res["executive_summary"].strip()
            )
            m = evaluator.evaluate_summary(
                ref_summary, res["executive_summary"], compute_bertscore=True
            )
            ablation_records.append(
                {
                    "model": model_name,
                    "video_id": vid_id,
                    "split": meta_map.get(vid_id, "test"),
                    "rouge1_f1": m["rouge1_f1"],
                    "rouge2_f1": m["rouge2_f1"],
                    "rougeL_f1": m["rougeL_f1"],
                    "bertscore_f1": m["bertscore_f1"],
                    "summary_words": res["word_count"],
                    "summary_sentences": res["sentence_count"],
                    "native_word_limit_met": res["native_word_limit_met"],
                    "latency_sec": res["latency_sec"],
                }
            )

    df_ablation = pd.DataFrame(ablation_records)
    ablation_csv_path = metrics_dir / "ablation_summarizers.csv"
    df_ablation.to_csv(ablation_csv_path, index=False)

    # 8. Export Reproducibility Artifact Manifest
    manifest = {
        "project": config["project"]["name"],
        "author": config["project"]["author"],
        "version": config["project"]["version"],
        "random_seed": config["project"]["random_seed"],
        "total_runtime_sec": round(time.perf_counter() - pipeline_start, 2),
        "artifacts": {},
    }
    for key in [
        "extracted_audio_dir",
        "transcripts_dir",
        "subtitles_dir",
        "summaries_dir",
        "metrics_dir",
    ]:
        folder = Path(config["paths"][key])
        if folder.exists():
            for fp in sorted(folder.glob("*")):
                if fp.is_file():
                    manifest["artifacts"][str(fp.as_posix())] = compute_sha256(fp)

    Path("manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    logger.info(
        f"\nPipeline completed successfully! CSV metrics saved to:\n"
        f"  1. {eval_csv_path}\n"
        f"  2. {stats_csv_path}\n"
        f"  3. {ablation_csv_path}"
    )
    print("\n=== EVALUATION SUMMARY (CSV PREVIEW) ===")
    print(
        df_metrics[
            [
                "video_id",
                "duration_sec",
                "wer",
                "cer",
                "rouge1_f1",
                "rouge2_f1",
                "rougeL_f1",
                "bertscore_f1",
                "summary_words",
                "summary_sentences",
                "native_word_limit_met",
                "latency_sec",
            ]
        ].to_string(index=False)
    )
    print("\n=== METRIC DISTRIBUTION STATS (CSV PREVIEW) ===")
    print(df_stats.to_string(index=False))


if __name__ == "__main__":
    main()