from pathlib import Path
import sys

# Automatically adds the project root folder to Python's import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import subprocess
import time
import wave
from pathlib import Path
from typing import Any, Dict

from src.config_loader import get_logger


class AudioExtractor:
    """Extracts 16kHz mono WAV audio from lecture videos with retry resilience."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.audio_cfg = config["audio_extraction"]
        self.paths_cfg = config["paths"]
        self.logger = get_logger(self.__class__.__name__, config)

    def extract_audio(self, video_path: str | Path) -> Dict[str, Any]:
        """
        Extracts audio from a video file (or normalizes an existing audio file)
        into a 16kHz mono WAV file required by Whisper ASR.
        """
        input_path = Path(video_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input media file not found: {input_path}")

        output_dir = Path(self.paths_cfg["extracted_audio_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_wav = output_dir / f"{input_path.stem}.{self.audio_cfg['audio_format']}"

        sample_rate = int(self.audio_cfg["target_sample_rate"])
        channels = int(self.audio_cfg["channels"])
        codec = str(self.audio_cfg["codec"])
        max_retries = int(self.audio_cfg["max_retries"])
        backoff = float(self.audio_cfg["retry_backoff_seconds"])

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-vn",
            "-acodec", codec,
            "-ar", str(sample_rate),
            "-ac", str(channels),
            str(output_wav),
        ]

        start_time = time.perf_counter()
        attempt = 0
        last_error = None

        while attempt < max_retries:
            attempt += 1
            try:
                self.logger.info(
                    f"Extracting audio from '{input_path.name}' (Attempt {attempt}/{max_retries})..."
                )
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                duration_sec = self._verify_wav_file(output_wav, sample_rate, channels)
                elapsed_sec = round(time.perf_counter() - start_time, 3)

                self.logger.info(
                    f"Audio extracted: {output_wav.name} | Duration: {duration_sec:.2f}s | Latency: {elapsed_sec}s"
                )
                return {
                    "video_id": input_path.stem,
                    "input_path": str(input_path),
                    "audio_path": str(output_wav),
                    "audio_duration_sec": round(duration_sec, 2),
                    "extraction_latency_sec": elapsed_sec,
                    "attempts_used": attempt,
                    "status": "SUCCESS",
                }
            except Exception as exc:
                last_error = str(exc)
                self.logger.warning(
                    f"Audio extraction attempt {attempt} failed for {input_path.name}: {last_error}"
                )
                if attempt < max_retries:
                    sleep_time = backoff * (2 ** (attempt - 1))
                    time.sleep(sleep_time)

        raise RuntimeError(
            f"Failed to extract audio from {input_path.name} after {max_retries} attempts. Last error: {last_error}"
        )

    @staticmethod
    def _verify_wav_file(wav_path: Path, expected_sr: int, expected_channels: int) -> float:
        """Validates WAV header properties and returns audio duration in seconds."""
        if not wav_path.exists() or wav_path.stat().st_size == 0:
            raise ValueError(f"Generated WAV file is missing or empty: {wav_path}")

        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            frames = wf.getnframes()
            if sr != expected_sr or ch != expected_channels:
                raise ValueError(
                    f"WAV header mismatch: got {sr}Hz/{ch}ch, expected {expected_sr}Hz/{expected_channels}ch"
                )
            duration = frames / float(sr)
            if duration <= 0.1:
                raise ValueError(f"Audio duration too short ({duration:.3f}s) in {wav_path.name}")
            return duration