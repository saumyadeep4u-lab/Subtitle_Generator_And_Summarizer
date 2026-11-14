from pathlib import Path
import sys
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from src.config_loader import get_logger, load_config


class TopicSegmenter:
  """Splits lecture transcripts into coherent topic sections with strict token bounds."""

  def __init__(self, config: Dict[str, Any]):
    self.config = config
    self.seg_cfg = config["topic_segmentation"]
    self.sum_cfg = config["summarization"]
    self.logger = get_logger(self.__class__.__name__, config)

    for resource in ["punkt", "punkt_tab"]:
      try:
        nltk.data.find(f"tokenizers/{resource}")
      except LookupError:
        nltk.download(resource, quiet=True)

    self.embed_model_name = self.seg_cfg["embedding_model"]
    self.window_size = int(self.seg_cfg["window_size_sentences"])
    self.sim_threshold = float(self.seg_cfg["similarity_drop_threshold"])
    self.max_chunk_tokens = int(self.seg_cfg["max_chunk_tokens"])
    self.chunk_overlap_tokens = int(self.seg_cfg["chunk_overlap_tokens"])
    self.min_section_tokens = int(self.seg_cfg["min_section_tokens"])

    self.logger.info(
        f"Loading topic embedding model: {self.embed_model_name}..."
    )
    self.embedder = SentenceTransformer(self.embed_model_name)
    self.tokenizer = AutoTokenizer.from_pretrained(
        self.sum_cfg["primary_model"]
    )

  def count_tokens(self, text: str) -> int:
    """Returns exact subword token count using the primary summarization tokenizer."""
    if not text.strip():
      return 0
    return len(self.tokenizer.encode(text, add_special_tokens=False))

  def segment_transcript(
      self, transcript_payload: Dict[str, Any]
  ) -> List[Dict[str, Any]]:
    """Splits a transcript payload into topic-coherent, token-bounded sections."""
    segments = transcript_payload.get("segments", [])
    if not segments:
      full_text = transcript_payload.get("full_text", "").strip()
      if not full_text:
        return []
      sentences = nltk.sent_tokenize(full_text)
      segments = [
          {"start": 0.0, "end": 0.0, "text": s} for s in sentences if s.strip()
      ]

    # Group raw ASR segments into sentence-level units with timestamps
    sentence_units: List[Dict[str, Any]] = []
    for seg in segments:
      text = seg.get("text", "").strip()
      if not text:
        continue
      sentence_units.append({
          "start": float(seg.get("start", 0.0)),
          "end": float(seg.get("end", 0.0)),
          "text": text,
          "tokens": self.count_tokens(text),
      })

    if not sentence_units:
      return []

    # Compute sliding-window embeddings to detect semantic topic shifts
    texts = [u["text"] for u in sentence_units]
    embeddings = self.embedder.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True
    )

    topic_Sections: List[Dict[str, Any]] = []
    current_units: List[Dict[str, Any]] = []
    current_tokens = 0

    for idx, unit in enumerate(sentence_units):
      unit_tokens = unit["tokens"]

      # Check semantic shift between preceding window and upcoming window
      is_topic_shift = False
      if idx >= self.window_size and idx + self.window_size <= len(
          sentence_units
      ):
        prev_vec = np.mean(embeddings[idx - self.window_size : idx], axis=0)
        next_vec = np.mean(embeddings[idx : idx + self.window_size], axis=0)
        denom = np.linalg.norm(prev_vec) * np.linalg.norm(next_vec) + 1e-8
        cosine_sim = float(np.dot(prev_vec, next_vec) / denom)
        if (
            cosine_sim < self.sim_threshold
            and current_tokens >= self.min_section_tokens
        ):
          is_topic_shift = True

      exceeds_token_limit = (current_tokens + unit_tokens) > (
          self.max_chunk_tokens - 10
      )

      if current_units and (is_topic_shift or exceeds_token_limit):
        topic_Sections.append(
            self._finalize_section(len(topic_Sections) + 1, current_units)
        )

        # Carry over token-bounded overlap when split was forced by token ceiling
        if exceeds_token_limit and self.chunk_overlap_tokens > 0:
          overlap_units: List[Dict[str, Any]] = []
          overlap_toks = 0
          for prev_u in reversed(current_units):
            if overlap_toks + prev_u["tokens"] <= self.chunk_overlap_tokens:
              overlap_units.insert(0, prev_u)
              overlap_toks += prev_u["tokens"]
            else:
              break
          current_units = overlap_units
          current_tokens = overlap_toks
        else:
          current_units = []
          current_tokens = 0

      current_units.append(unit)
      current_tokens += unit_tokens

    if current_units:
      # Merge trailing tiny fragment into the previous section if within token limit
      if (
          topic_Sections
          and current_tokens < self.min_section_tokens
          and (topic_Sections[-1]["token_count"] + current_tokens)
          <= self.max_chunk_tokens
      ):
        merged_units = topic_Sections[-1]["_raw_units"] + current_units
        topic_Sections[-1] = self._finalize_section(
            len(topic_Sections), merged_units
        )
      else:
        topic_Sections.append(
            self._finalize_section(len(topic_Sections) + 1, current_units)
        )

    for sec in topic_Sections:
      sec.pop("_raw_units", None)

    self.logger.info(
        f"Segmented '{transcript_payload.get('video_id', 'lecture')}' into"
        f" {len(topic_Sections)} topic sections (Max tokens enforced:"
        f" {self.max_chunk_tokens})."
    )
    return topic_Sections

  def _finalize_section(
      self, section_id: int, units: List[Dict[str, Any]]
  ) -> Dict[str, Any]:
    text = " ".join(u["text"] for u in units).strip()
    token_count = self.count_tokens(text)
    if token_count > self.max_chunk_tokens:
      encoded = self.tokenizer.encode(text, add_special_tokens=False)[
          : self.max_chunk_tokens
      ]
      text = self.tokenizer.decode(encoded, skip_special_tokens=True).strip()
      token_count = len(encoded)

    return {
        "section_id": section_id,
        "start_sec": round(units[0]["start"], 2),
        "end_sec": round(units[-1]["end"], 2),
        "token_count": token_count,
        "word_count": len(text.split()),
        "text": text,
        "_raw_units": units,
    }


if __name__ == "__main__":
  cfg = load_config()
  segmenter = TopicSegmenter(cfg)
  dummy_transcript = {
      "video_id": "test_lecture",
      "segments": [
          {
              "start": i * 5.0,
              "end": (i + 1) * 5.0,
              "text": (
                  f"Sentence {i} explains gradient descent optimization and"
                  " learning rates in deep neural networks."
              ),
          }
          for i in range(1, 25)
      ],
  }
  sections = segmenter.segment_transcript(dummy_transcript)
  print(
      f"TopicSegmenter Smoke Test: Created {len(sections)} section(s) | First"
      f" section tokens: {sections[0]['token_count']}"
  )