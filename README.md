# Subtitle Generator and Summarizer
**(EdTech Subtitle Generator & Hierarchical Lecture Summarizer)**

**Author:** Saumyadeep Bardhan  
**Organization:** Newton AI  
**Project Deliverable:** End-to-End ASR Transcription, `.srt` Subtitle Conversion, Topic Segmentation, Summarization & Quantitative Evaluation Pipeline

---

## 1. Project Overview & Architecture
This repository implements an automated, deterministic AI pipeline designed to process long-form EdTech video lectures (10–12 minutes) through five modular stages:

1. **Audio Extraction (`src/audio_extractor.py`):** Standardizes media tracks into $16\text{ kHz}$, mono-channel, 16-bit PCM WAV audio via FFmpeg with fault-tolerant retry logic.
2. **Speech-to-Text Transcription (`src/transcriber.py`):** Transcribes audio using **OpenAI Whisper (`base`)** with deterministic beam search (`temperature=0.0`, `beam_size=5`), capturing timestamped segments and Real-Time Factor (RTF) telemetry:
   $$\text{RTF} = \frac{T_{\text{inference}}}{T_{\text{audio\_duration}}}$$
3. **Broadcast Subtitle Formatting (`src/srt_formatter.py`):** Converts transcripts into SubRip (`.srt`) subtitles adhering to broadcast readability constraints ($\le 42$ characters/line, $\le 2$ lines/cue, $0.8\text{ s}$ to $7.0\text{ s}$ duration, and zero timestamp overlap).
4. **Logical Topic Segmentation (`src/topic_segmenter.py`):** Splits long transcripts into coherent thematic sections ($\le 512$ tokens with a $64$-token sliding overlap) using sentence embeddings (`all-MiniLM-L6-v2`).
5. **Hierarchical Summarization (`src/summarizer.py`):** Summarizes sections independently (Map step) and synthesizes a topic-focused **3–5 sentence executive summary ($\le 100$ words)** using **`facebook/bart-large-cnn`**, benchmarked against **`google/flan-t5-base`** and extractive **`LexRank`** (Reduce step).

---

## 2. Dataset & Large Media Assets
Due to portal file size limits, large source video assets (`raw_videos`) and raw extracted audio waveforms (`extracted_audio`) are hosted externally:

* **[Download Full Media Assets (Google Drive)](https://drive.google.com/drive/folders/1-2zRexLCCSaKQsDBvmNYQkBAIe3YScCz?usp=sharing)**

To run the pipeline locally on the full lecture corpus (`lecture_01_neural_networks.mp4` through `lecture_05_mlops_deployment.mp4`), download the media folders from the Google Drive link above and place them inside your project's `data/` directory as shown in the directory tree below. *(Note: If the media folders are left empty, the pipeline automatically synthesizes a fallback test waveform on the fly so execution never halts).*

---

## 3. Project Directory Structure
```text
Subtitle_Generator_And_Summarizer_By_Saumyadeep_Bardhan/
├── data/
│   ├── extracted_audio/          # [Download from Google Drive Link Above] 16kHz mono WAV audio tracks
│   ├── outputs/
│   │   ├── metrics/              # Evaluation CSV summaries & distribution stats
│   │   ├── subtitles/            # Broadcast-compliant .srt files
│   │   ├── summaries/            # Executive summaries (.txt & .json) across models
│   │   └── transcripts/          # JSON & TXT ASR transcripts
│   ├── raw_videos/               # [Download from Google Drive Link Above] Source lecture MP4 videos & metadata.json
│   └── reference_ground_truth/   # Reference texts, summaries, and section splits
├── logs/                         # Execution logs
├── reports/                      # Final 3-page project report PDF & documentation
├── src/                          # Core pipeline Python modules
│   ├── audio_extractor.py
│   ├── config_loader.py
│   ├── evaluator.py
│   ├── srt_formatter.py
│   ├── summarizer.py
│   ├── topic_segmenter.py
│   └── transcriber.py
├── tests/                        # Unit and boundary test suites
├── config.yaml                   # Centralized hyperparameter configuration
├── manifest.json                 # Cryptographic SHA-256 reproducibility artifact manifest
├── README.md                     # Project documentation, architecture & execution guide
├── requirements.txt              # Pinned Python package dependencies
├── run_pipeline.py               # Main pipeline orchestration script
└── Subtitle_Generator_And_Summarizer.ipynb # Presentation & evaluation notebook
```

---

## 4. Environment Setup & Installation

> **Prerequisite:** Ensure `FFmpeg` is installed and available on your system `PATH` for audio extraction and Whisper ASR processing.

1. **Clone or Extract the Repository:**
   ```bash
   cd Subtitle_Generator_And_Summarizer_By_Saumyadeep_Bardhan
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv .venv
   # On Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 5. Running the Pipeline, Notebook & Tests

1. **Execute the End-to-End Pipeline:**
   ```bash
   python run_pipeline.py
   ```
   This script automatically:
   * Fixes deterministic random seeds (`seed: 42`) across Python, NumPy, and PyTorch for $100\%$ reproducibility.
   * Processes all lectures found in `data/raw_videos/`.
   * Generates transcripts (`data/outputs/transcripts/`), validated `.srt` subtitles (`data/outputs/subtitles/`), and hierarchical summaries (`data/outputs/summaries/`).
   * Computes quantitative evaluation metrics (`WER`, `CER`, `WIL`, `ROUGE-1/2/L`, `BERTScore`).
   * Runs the summarizer model ablation study (`facebook/bart-large-cnn` vs. `google/flan-t5-base` vs. `LexRank`).
   * Exports evaluation CSVs (`evaluation_summary.csv`, `summary_distribution_stats.csv`, `ablation_summarizers.csv`) to `data/outputs/metrics/` and generates a cryptographic `SHA-256` reproducibility `manifest.json`.

2. **Explore the Interactive Evaluation Notebook:**
   ```bash
   jupyter notebook Subtitle_Generator_And_Summarizer.ipynb
   ```

3. **Run the Automated Test Suite:**
   ```bash
   pytest tests/
   ```

---

## 6. Summary of Quantitative Results

* **ASR Accuracy & Telemetry:** Across the 5 full-length ($10.5\text{–}11.3\text{ minute}$) technical lectures, OpenAI Whisper (`base`) achieves a **mean Word Error Rate (WER) of $7.65\%$ ($0.0765$)**, a **mean Character Error Rate (CER) of $3.28\%$ ($0.0328$)**, a **mean Word Information Lost (WIL) of $11.97\%$ ($0.1197$)**, and a **mean Real-Time Factor (RTF) of $0.1270$**.
* **Subtitle Structural Compliance:** **$100\%$** of generated `.srt` subtitle files pass automated structural validation ($\le 42$ characters/line, $\le 2$ lines/cue, $0.8\text{ s}$ to $7.0\text{ s}$ duration, and zero overlapping timestamps).
* **Summarization Quality (ROUGE & BERTScore):** Hierarchical `facebook/bart-large-cnn` summaries achieve **$100\%$ compliance** with the $\le 100$-word (corpus mean: **$56.2\text{ words}$**, range: $50\text{–}62\text{ words}$) and 3–5 sentence (**$4\text{ sentences}$** per summary) constraints, achieving:
  * **Mean ROUGE-1 F1:** $0.3300$ (max: $0.3793$)
  * **Mean ROUGE-2 F1:** $0.0741$ (max: $0.1228$)
  * **Mean ROUGE-L F1:** $0.2176$ (max: $0.2500$)
  * **Mean BERTScore F1:** $0.8291$ ($> 0.82$ threshold across all lectures)

| Lecture Video ID | Split | Duration (s) | WER | CER | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BERTScore F1 | Summary Words | SRT Valid |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `lecture_01_neural_networks` | dev | $629.38$ | $0.0919$ | $0.0428$ | $0.3793$ | $0.1228$ | $0.2414$ | $0.8236$ | $51$ | `True` |
| `lecture_02_nlp_transformers` | dev | $633.98$ | $0.0816$ | $0.0275$ | $0.3577$ | $0.0826$ | $0.2114$ | $0.8321$ | $50$ | `True` |
| `lecture_03_supervised_learning` | test | $668.42$ | $0.0393$ | $0.0148$ | $0.3500$ | $0.0678$ | $0.2500$ | $0.8245$ | $56$ | `True` |
| `lecture_04_computer_vision_cnns` | test | $657.98$ | $0.1008$ | $0.0592$ | $0.2677$ | $0.0640$ | $0.2047$ | $0.8367$ | $62$ | `True` |
| `lecture_05_mlops_deployment` | test | $678.27$ | $0.0688$ | $0.0199$ | $0.2951$ | $0.0333$ | $0.1803$ | $0.8284$ | $62$ | `True` |
| **Corpus Mean** | **All** | **$653.61$** | **$0.0765$** | **$0.0328$** | **$0.3300$** | **$0.0741$** | **$0.2176$** | **$0.8291$** | **$56.2$** | **$100\%$** |