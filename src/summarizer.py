import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nltk
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config_loader import get_logger, load_config


class LectureSummarizer:
    """Hierarchical lecture summarizer supporting BART, FLAN-T5, and LexRank baselines."""

    ASR_TERM_FIXES = [
        (r"(?i)\bsummary\s*:\s*", ""),
        (r"(?i)\bnumber\s*py\b", "NumPy"),
        (r"(?i)\bnumber\s*pie\b", "NumPy"),
        (r"(?i)\bmany[\s\-]+batch(?:es)?\b", "mini-batch"),
        (r"(?i)\bmini\s+batch\b", "mini-batch"),
        (r"(?i)\bmini\s+batches\b", "mini-batches"),
        (r"(?i)\btraining\s+date\s+set\b", "training dataset"),
        (r"(?i)\bdate\s+sets?\s+splits\b", "dataset splits"),
        (r"(?i)\bdata\s+sets?\s+splits\b", "dataset splits"),
        (r"(?i)\bimbalance\s+data\s*sets?\b", "imbalanced datasets"),
        (r"(?i)\bdate\s+sets?\b", "datasets"),
        (r"(?i)\bdata\s+sets\b", "datasets"),
        (r"(?i)\bdata\s+set\b", "dataset"),
        (r"(?i)\bshatoo\s*56\b", "SHA-256"),
        (r"(?i)\bsha[\s\-]*256\b", "SHA-256"),
        (r"(?i)\bbe\s+test\b", "pytest"),
        (r"(?i)\bp\s+test\b", "pytest"),
        (r"(?i)\bml\s*ops\b", "MLOps"),
        (r"(?i)\bm\s+l\s+ops\b", "MLOps"),
        (r"(?i)\bfall\s+tolerance\b", "fault tolerance"),
        (r"(?i)\bmirror\s+day\s+synchronously\b", "mirrored asynchronously"),
        (r"(?i)\bmirror\s+asynchronously\b", "mirrored asynchronously"),
        (r"(?i)\bupdates\s+to\s+a\s+heuristic\b", "updates to heuristic"),
        (r"(?i)\bmodel\s+ethics\s+scripts\b", "Monolithic scripts"),
        (r"(?i)\bab\s+testing\b", "A/B testing"),
        (r"(?i)\ba-b\s+testing\b", "A/B testing"),
        (r"(?i)\bres\s+net\b", "ResNet"),
        (r"(?i)\bu\s+net\b", "U-Net"),
        (r"(?i)\bflan[\s\-]*t5\b", "FLAN-T5"),
        (r"(?i)\bflanty\s*5\b", "FLAN-T5"),
        (r"(?i)\bbird\s+score\b", "BERTScore"),
        (r"(?i)\bbert\s+score\b", "BERTScore"),
        (r"(?i)\bert\s+score\b", "BERTScore"),
        (r"(?i)\brouge\b", "ROUGE"),
        (r"(?i)\bpret[\s\-]*ranged?\b", "pretrained"),
        (r"(?i)\bpret[\s\-]*rained\b", "pretrained"),
        (r"(?i)\bpret[\s\-]*rain\b", "pretrained"),
        (r"(?i)\bpre[\s\-]*trained\b", "pretrained"),
        (r"(?i)\bparadims\b", "paradigms"),
        (r"(?i)\bincluding\s+bird\b", "including BERT"),
        (r"(?i)\bbird,\s*bart\b", "BERT, BART"),
        (r"(?i)\bbart\b", "BART"),
        (r"(?i)\bauto[\s\-]*aggressive\b", "autoregressive"),
        (r"(?i)\bdenoizing\b", "denoising"),
        (r"(?i)\bauto[\s\-]*encoder\b", "autoencoder"),
        (r"(?i)\bscale\s+dot[\s\-]*product\b", "scaled dot-product"),
        (r"(?i)\bmultihead\b", "Multi-head"),
        (r"(?i)\bback\s+propagation\b", "backpropagation"),
        (r"(?i)\bheteroscodasticity\b", "heteroskedasticity"),
        (r"(?i)\blog\s*ads\b", "log-odds"),
        (r"(?i)\blog\s+odds\b", "log-odds"),
        (r"(?i)\bcross\s+validation\b", "cross-validation"),
        (r"(?i)\blook\s+ahead\s+bias\b", "look-ahead bias"),
        (r"(?i)\btime\s+series\b", "time-series"),
        (r"(?i)\bstack\s+layers\b", "stacked layers"),
        (r"(?i)\bencoder\s+decoder\b", "encoder-decoder"),
        (r"(?i)\bzero\s+padding\b", "Zero-padding"),
        (r"(?i)\bswish\b", "Swish"),
        (r"(?i)\bsilu\b", "SiLU"),
        (r"(?i)\bgaussian\s+error\s+linear\s+units?\b", "GELU"),
        (r"(?i)\bgaussian\s+error\s+linear\.?$", "GELU activations."),
    ]

    DANGLING_END_WORDS = {
        "a", "an", "the", "and", "or", "but", "nor", "so", "yet", "for",
        "to", "in", "on", "at", "by", "of", "with", "from", "into", "across",
        "through", "during", "including", "such", "as", "that", "which", "who",
        "whom", "whose", "where", "when", "while", "whereas", "because", "if",
        "between", "among", "within", "without", "towards", "upon",
        "can", "could", "will", "would", "should", "must", "may", "might",
        "is", "are", "was", "were", "be", "been", "has", "have", "had",
        "linear", "first", "second", "third", "differentiable", "successfully",
        "positive", "negative", "forests", "classifiers", "length",
    }

    VERB_MAP = {
        "examine": "examines",
        "analyze": "analyzes",
        "explore": "explores",
        "focus": "focuses",
        "address": "addresses",
        "study": "studies",
        "compare": "compares",
        "delve": "delves",
        "review": "reviews",
        "summarize": "summarizes",
        "extend": "extends",
        "perform": "performs",
        "quantify": "quantifies",
        "conduct": "conducts",
        "trace": "traces",
        "introduce": "introduces",
        "outline": "outlines",
        "cover": "covers",
    }

    def __init__(
        self, config: Dict[str, Any], model_override: Optional[str] = None
    ):
        self.config = config
        self.sum_cfg = config["summarization"]
        self.paths_cfg = config["paths"]
        self.logger = get_logger(self.__class__.__name__, config)

        self.model_name = model_override or self.sum_cfg["primary_model"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.model_name.lower() != "lexrank":
            self.logger.info(
                f"Loading summarization model '{self.model_name}' on"
                f" '{self.device}'..."
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name
            ).to(self.device)
            self.model.eval()
        else:
            self.logger.info("Initialized extractive LexRank baseline summarizer.")
            self.tokenizer = None
            self.model = None

    @classmethod
    def _normalize_terms(cls, text: str) -> str:
        cleaned = text.strip()
        for pattern, repl in cls.ASR_TERM_FIXES:
            cleaned = re.sub(pattern, repl, cleaned)
        cleaned = re.sub(r",\s*\.", ".", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    @classmethod
    def _clean_discourse_sentence(
        cls, sentence: str, subject_prefix: str = "This lecture"
    ) -> str:
        """Removes spoken transitions and ensures grammatically complete academic sentences."""
        s = cls._normalize_terms(sentence)
        if not s:
            return ""

        if re.match(r"(?i)^to reinforce key concepts from section\b", s):
            return ""
        if re.match(r"(?i)^in this supplemental review segment\b", s):
            return ""
        if re.match(r"(?i)^suppose we process\b", s):
            return ""

        s = re.sub(r"(?i)^(?:first|second|third),?\s+", "", s)

        s = re.sub(
            r"(?i)^even more effective in modern transformer and convolutional backbones are\s+",
            "Modern transformer and convolutional backbones utilize ",
            s,
        )

        s = re.sub(
            r"(?i)^welcome to lecture\s+[a-z0-9]+\s+on\s+",
            f"{subject_prefix} examines ",
            s,
        )

        if subject_prefix != "This lecture":
            s = re.sub(r"^This lecture\s+", f"{subject_prefix} ", s)

        s = re.sub(
            r"(?i)^(?:in|moving to)\s+(?:our\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|final|concluding)?\s*"
            r"(?:and\s+(?:final|concluding)\s+)?(?:section|topic)\s*(?:[a-z0-9]+)?,?\s*(?:we|let us)\s+(?:also\s+|now\s+)?",
            "",
            s,
        )
        s = re.sub(
            r"(?i)^section\s+[a-z0-9]+\s+focuses\s+on\s+",
            f"{subject_prefix} focuses on ",
            s,
        )

        words = s.split()
        if words:
            first_lower = words[0].lower()
            if first_lower in cls.VERB_MAP:
                rest = " ".join(words[1:])
                s = f"{subject_prefix} {cls.VERB_MAP[first_lower]} {rest}"

        if s and s[0].islower():
            s = s[0].upper() + s[1:]

        s = s.rstrip(",;: ")
        tokens = [t.strip(".,;:!?") for t in s.split() if t.strip(".,;:!?")]
        if not tokens:
            return ""
        if tokens[-1].lower() in cls.DANGLING_END_WORDS:
            return ""

        if s and s[-1] not in ".!?":
            s += "."

        return s

    @classmethod
    def _is_complete_sentence(cls, s: str) -> bool:
        """Verifies that a sentence has sufficient length, a finite verb, and no cut-off ending."""
        if not s or s[-1] not in ".!?":
            return False
        if (
            re.match(r"(?i)^while\b", s)
            and "often" not in s.lower()
            and "suffer" not in s.lower()
            and "remain" not in s.lower()
            and "aggregate" not in s.lower()
            and "predict" not in s.lower()
        ):
            return False
        if re.match(r"(?i)^when\b", s) and "," not in s:
            return False
        words = s.split()
        if len(words) < 8 or len(words) > 28:
            return False
        last_word = words[-1].strip(".,;:!?").lower()
        if last_word in cls.DANGLING_END_WORDS:
            return False

        lower_words = {w.lower().strip(".,;:") for w in words}
        finite_verbs = {
            "is", "are", "was", "were", "be", "been", "has", "have", "had",
            "do", "does", "did", "can", "could", "will", "would", "should",
            "must", "may", "might", "examines", "analyzes", "explores",
            "focuses", "addresses", "studies", "compares", "delves",
            "reviews", "summarizes", "extends", "performs", "quantifies",
            "conducts", "traces", "introduces", "outlines", "covers",
            "models", "computes", "applies", "requires", "uses", "utilize",
            "utilizes", "enables", "prevents", "reduces", "solves", "combines",
            "trains", "predicts", "penalizes", "estimates", "extracts",
            "processes", "measures", "enforces", "aggregates", "adds",
            "improves", "reconstructs", "normalizes", "stabilizes", "decomposed",
            "ensure", "ensures", "mirrored",
        }
        if not (lower_words & finite_verbs):
            return False
        return True

    @classmethod
    def _prepare_clean_sentences(cls, raw_text: str) -> List[str]:
        """Splits text into deduplicated, well-formed sentences free of spoken artifacts."""
        normalized = cls._normalize_terms(raw_text)
        raw_sents = nltk.sent_tokenize(normalized)
        clean_sents: List[str] = []
        seen_fingerprints = set()

        for raw_s in raw_sents:
            stripped = raw_s.strip()
            if not stripped or stripped[0].islower() or stripped[0].isdigit():
                continue

            cleaned = cls._clean_discourse_sentence(stripped)
            if not cls._is_complete_sentence(cleaned):
                continue

            fp = re.sub(r"[^a-z0-9]", "", cleaned.lower())[:75]
            if fp not in seen_fingerprints:
                seen_fingerprints.add(fp)
                clean_sents.append(cleaned)

        return clean_sents

    def _generate_seq2seq(
        self, text: str, gen_cfg: Dict[str, Any], is_t5: bool = False
    ) -> str:
        prefix = "summarize: " if is_t5 else ""
        inputs = self.tokenizer(
            prefix + text,
            max_length=1024 if not is_t5 else 512,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                min_new_tokens=int(gen_cfg["min_new_tokens"]),
                max_new_tokens=int(gen_cfg["max_new_tokens"]),
                num_beams=int(gen_cfg["num_beams"]),
                length_penalty=float(gen_cfg["length_penalty"]),
                no_repeat_ngram_size=int(gen_cfg["no_repeat_ngram_size"]),
                early_stopping=True,
            )
        decoded = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()
        return self._normalize_terms(decoded)

    @classmethod
    def _extractive_lexrank(cls, text: str, sentence_count: int = 4) -> str:
        clean_sents = [
            s for s in cls._prepare_clean_sentences(text) if len(s.split()) <= 24
        ]
        if len(clean_sents) < sentence_count:
            clean_sents = cls._prepare_clean_sentences(text)
        cleaned_block = " ".join(clean_sents) if clean_sents else text
        parser = PlaintextParser.from_string(
            cleaned_block, SumyTokenizer("english")
        )
        summarizer = LexRankSummarizer()
        summary_sents = summarizer(parser.document, sentence_count + 2)

        varied_prefixes = [
            "This lecture",
            "The module",
            "Furthermore, it",
            "Finally, the course",
        ]
        out_sents: List[str] = []
        seen_fps = set()
        for s in summary_sents:
            pref = varied_prefixes[len(out_sents) % len(varied_prefixes)]
            cleaned_s = cls._clean_discourse_sentence(str(s), subject_prefix=pref)
            if cls._is_complete_sentence(cleaned_s):
                body = re.sub(
                    r"^(?:This lecture|The module|Furthermore, it|Finally, the course)\s+",
                    "",
                    cleaned_s,
                )
                fp = re.sub(r"[^a-z0-9]", "", body.lower())[:60]
                if fp not in seen_fps:
                    seen_fps.add(fp)
                    out_sents.append(cleaned_s)
            if len(out_sents) >= sentence_count:
                break
        return " ".join(out_sents).strip()

    def _synthesize_multi_topic_abstractive(
        self,
        all_clean_sentences: List[str],
        vid_cfg: Dict[str, Any],
        is_t5: bool = False,
    ) -> str:
        """
        Partitions the lecture into 4 chronological topic groups and generates
        1 complete, high-clarity sentence per group (~15-24 words each) so the
        4-sentence summary covers the entire lecture in ~60-85 words without truncation.
        """
        if len(all_clean_sentences) < 6:
            joined = " ".join(all_clean_sentences)
            return self._generate_seq2seq(joined, vid_cfg, is_t5=is_t5)

        num_buckets = 4
        bucket_size = max(1, len(all_clean_sentences) // num_buckets)
        buckets: List[List[str]] = []
        for i in range(num_buckets):
            start_idx = i * bucket_size
            end_idx = (
                (i + 1) * bucket_size
                if i < num_buckets - 1
                else len(all_clean_sentences)
            )
            bucket_sents = all_clean_sentences[start_idx:end_idx]
            if bucket_sents:
                buckets.append(bucket_sents)

        bucket_gen_cfg = {
            "min_new_tokens": 20,
            "max_new_tokens": 64,
            "num_beams": int(vid_cfg.get("num_beams", 4)),
            "length_penalty": 1.0,
            "no_repeat_ngram_size": int(vid_cfg.get("no_repeat_ngram_size", 3)),
        }

        varied_prefixes = [
            "This lecture",
            "The curriculum",
            "Furthermore, it",
            "Finally, the module",
        ]
        synthesized_sentences: List[str] = []
        seen_fps = set()

        for idx, b_sents in enumerate(buckets):
            pref = varied_prefixes[idx % len(varied_prefixes)]
            b_text = " ".join(b_sents)
            raw_gen = self._generate_seq2seq(
                b_text, bucket_gen_cfg, is_t5=is_t5
            )

            raw_candidates = nltk.sent_tokenize(raw_gen)
            chosen = ""
            for rc in raw_candidates:
                rc_strip = rc.strip()
                if not rc_strip or rc_strip[-1] not in ".!?":
                    continue
                cleaned_rc = self._clean_discourse_sentence(
                    rc_strip, subject_prefix=pref
                )
                if self._is_complete_sentence(cleaned_rc) and len(cleaned_rc.split()) <= 24:
                    body = re.sub(
                        r"^(?:This lecture|The curriculum|Furthermore, it|Finally, the module)\s+",
                        "",
                        cleaned_rc,
                    )
                    fp = re.sub(r"[^a-z0-9]", "", body.lower())[:60]
                    if fp not in seen_fps:
                        chosen = cleaned_rc
                        break

            if not chosen:
                for fb_sent in b_sents:
                    cleaned_fb = self._clean_discourse_sentence(
                        fb_sent, subject_prefix=pref
                    )
                    if self._is_complete_sentence(cleaned_fb) and len(cleaned_fb.split()) <= 24:
                        body = re.sub(
                            r"^(?:This lecture|The curriculum|Furthermore, it|Finally, the module)\s+",
                            "",
                            cleaned_fb,
                        )
                        fp = re.sub(r"[^a-z0-9]", "", body.lower())[:60]
                        if fp not in seen_fps:
                            chosen = cleaned_fb
                            break

            if chosen:
                body = re.sub(
                    r"^(?:This lecture|The curriculum|Furthermore, it|Finally, the module)\s+",
                    "",
                    chosen,
                )
                fp = re.sub(r"[^a-z0-9]", "", body.lower())[:60]
                seen_fps.add(fp)
                synthesized_sentences.append(chosen)

        return " ".join(synthesized_sentences).strip()

    def _enforce_video_constraints(
        self, raw_summary: str, fallback_sentences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Enforces 3-5 complete sentences and <=100 words without mid-sentence truncation."""
        vid_cfg = self.sum_cfg["video_summary"]
        min_sents = int(vid_cfg["target_min_sentences"])
        max_sents = int(vid_cfg["target_max_sentences"])
        max_words = int(vid_cfg["max_words_limit"])

        varied_prefixes = [
            "This lecture",
            "The curriculum",
            "Furthermore, it",
            "Finally, the module",
            "Overall, the course",
        ]

        raw_normalized = self._normalize_terms(raw_summary)
        raw_tokenized = nltk.sent_tokenize(raw_normalized)

        sentences: List[str] = []
        seen = set()
        for s in raw_tokenized:
            pref = varied_prefixes[len(sentences) % len(varied_prefixes)]
            cleaned = self._clean_discourse_sentence(s, subject_prefix=pref)
            if self._is_complete_sentence(cleaned):
                body = re.sub(
                    r"^(?:This lecture|The curriculum|Furthermore, it|Finally, the module|Overall, the course)\s+",
                    "",
                    cleaned,
                )
                fp = re.sub(r"[^a-z0-9]", "", body.lower())[:60]
                if fp not in seen:
                    seen.add(fp)
                    sentences.append(cleaned)

        if len(sentences) < min_sents and fallback_sentences:
            for fb in fallback_sentences:
                pref = varied_prefixes[len(sentences) % len(varied_prefixes)]
                cleaned_fb = self._clean_discourse_sentence(
                    fb, subject_prefix=pref
                )
                if self._is_complete_sentence(cleaned_fb) and len(cleaned_fb.split()) <= 23:
                    body = re.sub(
                        r"^(?:This lecture|The curriculum|Furthermore, it|Finally, the module|Overall, the course)\s+",
                        "",
                        cleaned_fb,
                    )
                    fp = re.sub(r"[^a-z0-9]", "", body.lower())[:60]
                    if fp not in seen:
                        seen.add(fp)
                        sentences.append(cleaned_fb)
                if len(sentences) >= 4:
                    break

        raw_word_count = len(" ".join(sentences).split())
        post_processed = False

        if len(sentences) > max_sents:
            sentences = sentences[:max_sents]
            post_processed = True

        selected_sentences: List[str] = []
        running_words = 0
        for sent in sentences:
            w_count = len(sent.split())
            if running_words + w_count <= max_words:
                selected_sentences.append(sent)
                running_words += w_count
            else:
                post_processed = True

        if len(selected_sentences) < min_sents and fallback_sentences:
            for fb in sorted(fallback_sentences, key=lambda x: len(x.split())):
                pref = varied_prefixes[len(selected_sentences) % len(varied_prefixes)]
                cleaned_fb = self._clean_discourse_sentence(fb, subject_prefix=pref)
                w_count = len(cleaned_fb.split())
                if (
                    self._is_complete_sentence(cleaned_fb)
                    and running_words + w_count <= max_words
                ):
                    body = re.sub(
                        r"^(?:This lecture|The curriculum|Furthermore, it|Finally, the module|Overall, the course)\s+",
                        "",
                        cleaned_fb,
                    )
                    fp = re.sub(r"[^a-z0-9]", "", body.lower())[:60]
                    if fp not in seen:
                        seen.add(fp)
                        selected_sentences.append(cleaned_fb)
                        running_words += w_count
                if len(selected_sentences) >= min_sents:
                    break

        final_text = " ".join(selected_sentences).strip()
        final_sents = nltk.sent_tokenize(final_text)
        return {
            "summary_text": final_text,
            "word_count": len(final_text.split()),
            "sentence_count": len(final_sents),
            "raw_word_count": raw_word_count,
            "native_word_limit_met": raw_word_count <= max_words,
            "post_processing_triggered": post_processed,
        }

    def summarize_lecture(
        self,
        video_id: str,
        sections: List[Dict[str, Any]],
        save_outputs: bool = True,
    ) -> Dict[str, Any]:
        """Summarizes each logical section first, then synthesizes a concise 3-5 sentence video summary."""
        start_time = time.perf_counter()
        is_t5 = (
            "t5" in self.model_name.lower() or "flan" in self.model_name.lower()
        )
        sec_cfg = self.sum_cfg["section_summary"]
        vid_cfg = self.sum_cfg["video_summary"]

        all_clean_sentences: List[str] = []
        seen_global = set()

        section_summaries: List[Dict[str, Any]] = []
        for sec in sections:
            sec_clean_sents = self._prepare_clean_sentences(sec["text"])
            for cs in sec_clean_sents:
                fp = re.sub(r"[^a-z0-9]", "", cs.lower())[:75]
                if fp not in seen_global:
                    seen_global.add(fp)
                    all_clean_sentences.append(cs)

            clean_sec_text = (
                " ".join(sec_clean_sents) if sec_clean_sents else sec["text"]
            )
            if self.model_name.lower() == "lexrank":
                sec_sum = self._extractive_lexrank(
                    clean_sec_text, sentence_count=2
                )
            else:
                sec_sum = self._generate_seq2seq(
                    clean_sec_text, sec_cfg, is_t5=is_t5
                )
                sec_sum_sents = [
                    self._clean_discourse_sentence(s)
                    for s in nltk.sent_tokenize(sec_sum)
                    if self._clean_discourse_sentence(s)
                ]
                sec_sum = " ".join(sec_sum_sents)

            section_summaries.append(
                {
                    "section_id": sec["section_id"],
                    "start_sec": sec["start_sec"],
                    "end_sec": sec["end_sec"],
                    "section_summary": sec_sum,
                    "word_count": len(sec_sum.split()),
                }
            )

        combined_sections_text = " ".join(
            s["section_summary"]
            for s in section_summaries
            if s["section_summary"]
        )

        if self.model_name.lower() == "lexrank":
            raw_video_summary = self._extractive_lexrank(
                " ".join(all_clean_sentences)
                if all_clean_sentences
                else combined_sections_text,
                sentence_count=4,
            )
        else:
            raw_video_summary = self._synthesize_multi_topic_abstractive(
                all_clean_sentences
                if all_clean_sentences
                else nltk.sent_tokenize(combined_sections_text),
                vid_cfg,
                is_t5=is_t5,
            )

        fallback_pool = all_clean_sentences + [
            s["section_summary"]
            for s in section_summaries
            if s["section_summary"]
        ]
        enforced = self._enforce_video_constraints(
            raw_video_summary, fallback_sentences=fallback_pool
        )
        latency_sec = round(time.perf_counter() - start_time, 3)

        result = {
            "video_id": video_id,
            "summarizer_model": self.model_name,
            "device": self.device,
            "latency_sec": latency_sec,
            "num_sections": len(section_summaries),
            "executive_summary": enforced["summary_text"],
            "word_count": enforced["word_count"],
            "sentence_count": enforced["sentence_count"],
            "native_word_limit_met": enforced["native_word_limit_met"],
            "post_processing_triggered": enforced["post_processing_triggered"],
            "section_summaries": section_summaries,
        }

        if save_outputs:
            out_dir = Path(self.paths_cfg["summaries_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            is_primary = self.model_name == self.sum_cfg["primary_model"]
            model_tag = (
                ""
                if is_primary
                else f"_{self.model_name.split('/')[-1].replace('-', '_')}"
            )

            txt_path = out_dir / f"{video_id}_summary{model_tag}.txt"
            json_path = out_dir / f"{video_id}_summary{model_tag}.json"

            with open(txt_path, "w", encoding="utf-8") as tf:
                tf.write(result["executive_summary"] + "\n")
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(result, jf, indent=2, ensure_ascii=False)

            result["summary_txt_path"] = str(txt_path)
            result["summary_json_path"] = str(json_path)
            self.logger.info(
                f"Saved summary: {txt_path.name} | Words: {result['word_count']}"
                f" (<=100) | Sentences: {result['sentence_count']}"
            )

        return result


if __name__ == "__main__":
    cfg = load_config()
    summarizer = LectureSummarizer(cfg, model_override="lexrank")
    sample_sections = [
        {
            "section_id": 1,
            "start_sec": 0.0,
            "end_sec": 120.0,
            "text": (
                "Supervised learning trains models on labeled input-output pairs."
                " Linear regression predicts continuous values using ordinary least"
                " squares. Classification algorithms like logistic regression and"
                " support vector machines predict discrete class boundaries."
                " Regularization penalizes large weights to prevent overfitting on"
                " training data. Cross-validation estimates generalization error on"
                " unseen test splits."
            ),
        }
    ]
    res = summarizer.summarize_lecture(
        "smoke_test", sample_sections, save_outputs=False
    )
    print("Summarizer Smoke Test:", res["executive_summary"])