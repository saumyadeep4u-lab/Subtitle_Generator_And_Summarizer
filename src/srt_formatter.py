from pathlib import Path
import sys

# Automatically adds the project root folder to Python's import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List

from src.config_loader import get_logger


class SRTFormatter:
    """Generates and validates broadcast-compliant .srt files from ASR segments."""

    TIMESTAMP_REGEX = re.compile(
        r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})\s-->\s(\d{2}):(\d{2}):(\d{2}),(\d{3})$"
    )

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.srt_cfg = config["srt_formatting"]
        self.paths_cfg = config["paths"]
        self.logger = get_logger(self.__class__.__name__, config)

        self.max_chars_per_line = int(self.srt_cfg["max_chars_per_line"])
        self.max_lines_per_block = int(self.srt_cfg["max_lines_per_block"])
        self.min_duration_sec = float(self.srt_cfg["min_duration_sec"])
        self.max_duration_sec = float(self.srt_cfg["max_duration_sec"])
        self.min_gap_sec = float(self.srt_cfg["min_gap_ms"]) / 1000.0

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Converts seconds (float) to strict SRT timestamp format HH:MM:SS,mmm."""
        seconds = max(0.0, float(seconds))
        total_ms = int(round(seconds * 1000))
        hours = total_ms // 3_600_000
        remainder = total_ms % 3_600_000
        minutes = remainder // 60_000
        remainder %= 60_000
        secs = remainder // 1_000
        millis = remainder % 1_000
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _split_long_segment(self, seg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Splits segments that exceed max characters per block or max duration."""
        max_block_chars = self.max_chars_per_line * self.max_lines_per_block
        text = " ".join(seg["text"].split())
        start = float(seg["start"])
        end = max(start + self.min_duration_sec, float(seg["end"]))
        duration = end - start

        if len(text) <= max_block_chars and duration <= self.max_duration_sec:
            return [{"start": start, "end": end, "text": text}]

        # Split proportionally by words
        words = text.split()
        if not words:
            return []

        num_chunks = max(
            (len(text) + max_block_chars - 1) // max_block_chars,
            int(duration // self.max_duration_sec) + 1,
        )
        chunk_size = max(1, (len(words) + num_chunks - 1) // num_chunks)
        word_chunks = [words[i : i + chunk_size] for i in range(0, len(words), chunk_size)]

        sub_segments = []
        total_words = len(words)
        cursor = start

        for idx, w_chunk in enumerate(word_chunks):
            chunk_text = " ".join(w_chunk)
            proportion = len(w_chunk) / total_words
            chunk_dur = max(self.min_duration_sec, duration * proportion)
            chunk_end = end if idx == len(word_chunks) - 1 else min(end, cursor + chunk_dur)
            if chunk_end <= cursor:
                chunk_end = cursor + self.min_duration_sec

            sub_segments.append(
                {
                    "start": round(cursor, 3),
                    "end": round(chunk_end, 3),
                    "text": chunk_text,
                }
            )
            cursor = chunk_end + self.min_gap_sec

        return sub_segments

    def _wrap_lines(self, text: str) -> str:
        """Wraps subtitle text to max_chars_per_line and max_lines_per_block."""
        wrapped = textwrap.wrap(
            text,
            width=self.max_chars_per_line,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if len(wrapped) > self.max_lines_per_block:
            # Merge overflow safely into the last allowed line and re-wrap
            kept = wrapped[: self.max_lines_per_block - 1]
            tail = " ".join(wrapped[self.max_lines_per_block - 1 :])
            kept.append(tail[: self.max_chars_per_line])
            wrapped = kept
        return "\n".join(wrapped)

    def build_srt(self, transcript_payload: Dict[str, Any], output_filename: str | None = None) -> Dict[str, Any]:
        """Converts a transcript payload into a validated .srt file."""
        raw_segments = transcript_payload.get("segments", [])
        expanded_cues: List[Dict[str, Any]] = []

        for seg in raw_segments:
            if not seg.get("text", "").strip():
                continue
            expanded_cues.extend(self._split_long_segment(seg))

        # Enforce strict monotonic non-overlapping timestamps with min_gap_sec
        cleaned_cues: List[Dict[str, Any]] = []
        prev_end = 0.0
        for cue in expanded_cues:
            start = max(cue["start"], prev_end + (self.min_gap_sec if prev_end > 0 else 0.0))
            end = max(cue["end"], start + self.min_duration_sec)
            if end - start > self.max_duration_sec:
                end = start + self.max_duration_sec
            cleaned_cues.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": self._wrap_lines(cue["text"]),
                }
            )
            prev_end = end

        # Render SRT blocks
        srt_blocks = []
        for idx, cue in enumerate(cleaned_cues, start=1):
            ts_line = f"{self.format_timestamp(cue['start'])} --> {self.format_timestamp(cue['end'])}"
            srt_blocks.append(f"{idx}\n{ts_line}\n{cue['text']}")

        srt_content = "\n\n".join(srt_blocks) + ("\n" if srt_blocks else "")

        out_dir = Path(self.paths_cfg["subtitles_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        video_id = transcript_payload.get("video_id", "lecture")
        srt_path = out_dir / (output_filename or f"{video_id}.srt")

        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        validation = self.validate_srt_content(srt_content)
        self.logger.info(
            f"Generated SRT: {srt_path.name} | Cues: {len(cleaned_cues)} | Valid: {validation['is_valid']}"
        )
        return {
            "video_id": video_id,
            "srt_path": str(srt_path),
            "num_cues": len(cleaned_cues),
            "validation": validation,
        }

    def validate_srt_content(self, srt_content: str) -> Dict[str, Any]:
        """Audits an SRT string for timestamp format, monotonic ordering, and line length."""
        if not srt_content.strip():
            return {"is_valid": False, "errors": ["Empty SRT content"], "num_blocks": 0}

        blocks = [b.strip() for b in srt_content.strip().split("\n\n") if b.strip()]
        errors = []
        prev_end_ms = -1

        for idx, block in enumerate(blocks, start=1):
            lines = block.splitlines()
            if len(lines) < 3:
                errors.append(f"Block {idx} has fewer than 3 lines.")
                continue

            if not lines[0].strip().isdigit() or int(lines[0].strip()) != idx:
                errors.append(f"Block {idx} has non-sequential index '{lines[0]}'.")

            match = self.TIMESTAMP_REGEX.match(lines[1].strip())
            if not match:
                errors.append(f"Block {idx} invalid timestamp format: '{lines[1]}'")
                continue

            h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, match.groups())
            start_ms = ((h1 * 3600 + m1 * 60 + s1) * 1000) + ms1
            end_ms = ((h2 * 3600 + m2 * 60 + s2) * 1000) + ms2

            if end_ms <= start_ms:
                errors.append(f"Block {idx} end timestamp <= start timestamp.")
            if start_ms < prev_end_ms:
                errors.append(f"Block {idx} overlaps with previous subtitle block.")
            prev_end_ms = end_ms

            text_lines = lines[2:]
            if len(text_lines) > self.max_lines_per_block:
                errors.append(f"Block {idx} exceeds max lines ({len(text_lines)} > {self.max_lines_per_block}).")
            for t_line in text_lines:
                if len(t_line) > self.max_chars_per_line:
                    errors.append(
                        f"Block {idx} line exceeds {self.max_chars_per_line} chars ({len(t_line)} chars)."
                    )

        return {
            "is_valid": len(errors) == 0,
            "num_blocks": len(blocks),
            "error_count": len(errors),
            "errors": errors[:10],
        }

if __name__ == "__main__":
  from src.config_loader import load_config

  cfg = load_config()
  formatter = SRTFormatter(cfg)
  sample_payload = {
      "video_id": "smoke_test",
      "segments": [
          {
              "start": 0.0,
              "end": 3.5,
              "text": (
                  "Welcome to the machine learning lecture on neural networks."
              ),
          },
          {
              "start": 3.6,
              "end": 7.2,
              "text": (
                  "Today we will cover backpropagation and gradient descent."
              ),
          },
      ],
  }
  result = formatter.build_srt(sample_payload, "smoke_test.srt")
  print("SRT Smoke Test Result:", result)