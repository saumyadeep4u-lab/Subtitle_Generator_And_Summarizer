from pathlib import Path
import sys

# Automatically adds the project root folder to Python's import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import whisper

from src.config_loader import get_logger


class WhisperTranscriber:
    """Transcribes 16kHz lecture audio using OpenAI Whisper with telemetry logging."""

    def __init__(self, config: Dict[str, Any], model_override: Optional[str] = None):
        self.config = config
        self.asr_cfg = config["asr_whisper"]
        self.paths_cfg = config["paths"]
        self.logger = get_logger(self.__class__.__name__, config)

        self.model_name = model_override or self.asr_cfg["primary_model"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fp16 = bool(self.asr_cfg.get("fp16", False)) and (self.device == "cuda")

        self.logger.info(
            f"Loading Whisper model '{self.model_name}' on device='{self.device}' (fp16={self.fp16})..."
        )
        self.model = whisper.load_model(self.model_name, device=self.device)

    def transcribe(
        self, audio_path: str | Path, audio_duration_sec: Optional[float] = None
    ) -> Dict[str, Any]:
        """Runs Whisper transcription and saves structured JSON and TXT outputs."""
        wav_path = Path(audio_path)
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found for transcription: {wav_path}")

        start_time = time.perf_counter()
        self.logger.info(f"Starting transcription for '{wav_path.name}' using Whisper-{self.model_name}...")

        raw_result = self.model.transcribe(
            str(wav_path),
            language=self.asr_cfg["language"],
            task=self.asr_cfg["task"],
            word_timestamps=bool(self.asr_cfg["word_timestamps"]),
            fp16=self.fp16,
            condition_on_previous_text=bool(self.asr_cfg["condition_on_previous_text"]),
            no_speech_threshold=float(self.asr_cfg["no_speech_threshold"]),
            logprob_threshold=float(self.asr_cfg["logprob_threshold"]),
        )

        inference_sec = round(time.perf_counter() - start_time, 3)

        # Clean and normalize segments
        segments = []
        for idx, seg in enumerate(raw_result.get("segments", []), start=1):
            text = seg.get("text", "").strip()
            if not text:
                continue
            segments.append(
                {
                    "id": idx,
                    "start": round(float(seg["start"]), 3),
                    "end": round(float(seg["end"]), 3),
                    "text": text,
                    "avg_logprob": round(float(seg.get("avg_logprob", 0.0)), 4),
                    "no_speech_prob": round(float(seg.get("no_speech_prob", 0.0)), 4),
                    "words": [
                        {
                            "word": w.get("word", "").strip(),
                            "start": round(float(w.get("start", seg["start"])), 3),
                            "end": round(float(w.get("end", seg["end"])), 3),
                            "probability": round(float(w.get("probability", 1.0)), 4),
                        }
                        for w in seg.get("words", [])
                        if w.get("word", "").strip()
                    ],
                }
            )

        full_text = " ".join(s["text"] for s in segments).strip()
        total_duration = audio_duration_sec or (segments[-1]["end"] if segments else 1.0)
        rtf = round(inference_sec / max(total_duration, 0.1), 4)

        # Save outputs
        out_dir = Path(self.paths_cfg["transcripts_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = wav_path.stem
        suffix = "" if self.model_name == self.asr_cfg["primary_model"] else f"_{self.model_name}"

        json_path = out_dir / f"{stem}_transcript{suffix}.json"
        txt_path = out_dir / f"{stem}_transcript{suffix}.txt"

        payload = {
            "video_id": stem,
            "asr_model": f"whisper-{self.model_name}",
            "device": self.device,
            "audio_duration_sec": round(total_duration, 2),
            "transcription_latency_sec": inference_sec,
            "real_time_factor": rtf,
            "num_segments": len(segments),
            "word_count": len(full_text.split()),
            "full_text": full_text,
            "segments": segments,
        }

        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(payload, jf, indent=2, ensure_ascii=False)

        with open(txt_path, "w", encoding="utf-8") as tf:
            tf.write(full_text + "\n")

        self.logger.info(
            f"Saved transcript: {json_path.name} | Segments: {len(segments)} | Words: {payload['word_count']} | RTF: {rtf}"
        )
        return payload