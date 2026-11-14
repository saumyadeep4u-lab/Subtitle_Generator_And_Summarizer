from pathlib import Path
import re
import sys
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bert_score import score as bert_score_fn
import jiwer
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

from src.config_loader import get_logger, load_config


class PipelineEvaluator:
  """Evaluates ASR (WER, CER, WIL) and Summarization (ROUGE-1/2/L, BERTScore) with full summary statistics."""

  CONTRACTIONS = {
      "can't": "cannot",
      "won't": "will not",
      "don't": "do not",
      "doesn't": "does not",
      "isn't": "is not",
      "aren't": "are not",
      "wasn't": "was not",
      "weren't": "were not",
      "it's": "it is",
      "that's": "that is",
      "there's": "there is",
      "we're": "we are",
      "they're": "they are",
      "you're": "you are",
      "i'm": "i am",
      "let's": "let us",
  }

  def __init__(self, config: Dict[str, Any]):
    self.config = config
    self.eval_cfg = config["evaluation"]
    self.logger = get_logger(self.__class__.__name__, config)
    self.rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

  def normalize_text(self, text: str) -> str:
    """Normalizes reference and hypothesis strings according to config.yaml."""
    norm_cfg = self.eval_cfg["wer_normalization"]
    out = text.strip()
    if norm_cfg.get("lowercase", True):
      out = out.lower()
    if norm_cfg.get("expand_contractions", True):
      for k, v in self.CONTRACTIONS.items():
        out = re.sub(rf"\b{re.escape(k)}\b", v, out)
    if norm_cfg.get("remove_punctuation", True):
      out = re.sub(r"[^\w\s]", " ", out)
    return re.sub(r"\s+", " ", out).strip()

  def evaluate_asr(self, reference: str, hypothesis: str) -> Dict[str, float]:
    """Computes WER, CER, WIL, and exact error counts (Substitutions, Deletions, Insertions)."""
    ref_norm = self.normalize_text(reference)
    hyp_norm = self.normalize_text(hypothesis)

    if not ref_norm:
      raise ValueError("Reference transcript cannot be empty for WER evaluation.")

    word_output = jiwer.process_words(ref_norm, hyp_norm)
    cer_score = jiwer.cer(ref_norm, hyp_norm)

    return {
        "wer": round(float(word_output.wer), 4),
        "cer": round(float(cer_score), 4),
        "wil": round(float(word_output.wil), 4),
        "substitutions": int(word_output.substitutions),
        "deletions": int(word_output.deletions),
        "insertions": int(word_output.insertions),
        "hits": int(word_output.hits),
        "ref_words": len(ref_norm.split()),
    }

  def evaluate_summary(
      self,
      reference_summary: str,
      generated_summary: str,
      compute_bertscore: bool = True,
  ) -> Dict[str, float]:
    """Computes ROUGE-1, ROUGE-2, ROUGE-L F1 scores and optional BERTScore F1."""
    r_scores = self.rouge.score(reference_summary, generated_summary)
    metrics = {
        "rouge1_precision": round(float(r_scores["rouge1"].precision), 4),
        "rouge1_recall": round(float(r_scores["rouge1"].recall), 4),
        "rouge1_f1": round(float(r_scores["rouge1"].fmeasure), 4),
        "rouge2_f1": round(float(r_scores["rouge2"].fmeasure), 4),
        "rougeL_f1": round(float(r_scores["rougeL"].fmeasure), 4),
        "summary_words": len(generated_summary.split()),
    }

    if compute_bertscore:
      _, _, f1 = bert_score_fn(
          [generated_summary],
          [reference_summary],
          model_type=self.eval_cfg["bertscore_model"],
          verbose=False,
      )
      metrics["bertscore_f1"] = round(float(f1.mean().item()), 4)

    return metrics

  @staticmethod
  def compute_distribution_stats(
      df: pd.DataFrame, metric_cols: List[str]
  ) -> pd.DataFrame:
    """Computes Mean, Median, Std Dev, Min, and Max across evaluation samples."""
    rows = []
    for col in metric_cols:
      if col not in df.columns:
        continue
      vals = df[col].dropna().to_numpy(dtype=float)
      if len(vals) == 0:
        continue
      rows.append({
          "metric": col,
          "mean": round(float(np.mean(vals)), 4),
          "median": round(float(np.median(vals)), 4),
          "std": round(float(np.std(vals, ddof=1 if len(vals) > 1 else 0)), 4),
          "min": round(float(np.min(vals)), 4),
          "max": round(float(np.max(vals)), 4),
      })
    return pd.DataFrame(rows)


if __name__ == "__main__":
  cfg = load_config()
  evaluator = PipelineEvaluator(cfg)
  asr_res = evaluator.evaluate_asr(
      "Gradient descent minimizes the loss function iteratively.",
      "Gradient decent minimizes the loss function iteratively.",
  )
  sum_res = evaluator.evaluate_summary(
      "Gradient descent optimizes neural networks by updating weights using"
      " loss gradients.",
      "Gradient descent updates neural network weights to minimize the loss"
      " function.",
      compute_bertscore=False,
  )
  print("ASR Evaluation Smoke Test:", asr_res)
  print("Summary Evaluation Smoke Test:", sum_res)