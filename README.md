# RAI Content Moderation Pipeline

A five-part Responsible AI (RAI) project that builds, audits, debiases, and deploys a production-grade content moderation classifier on top of DistilBERT, with a three-layer guardrail pipeline.

---

## Environment

| Item | Value |
|------|-------|
| Python | 3.12.12 |
| Platform | Kaggle Notebooks (dockerImageVersionId `31329`) |
| GPU (training parts) | NVIDIA Tesla T4 (16 GB VRAM) |
| GPU (inference / pipeline) | Tesla T4 or CPU fallback |
| CUDA | 12.x (provided by Kaggle image) |

---

## Project Structure

```
├── part-1-rai.ipynb      # Baseline DistilBERT fine-tune + evaluation
├── part-2-rai.ipynb      # Fairness audit (AIF360 metrics, group disparity)
├── part-3-rai.ipynb      # Bias mitigation — data-level interventions
├── part-4-rai.ipynb      # Bias mitigation — Reweighing / Threshold Optimiser / Oversampling
├── part-5-rai.ipynb      # Probability calibration (IsotonicRegression) + pipeline assembly
├── pipeline.ipynb        # Production ModerationPipeline class + self-test
└── requirements.txt      # Pinned dependencies
```

### Artefacts produced (saved to `/kaggle/working/`)

| File / Folder | Produced by | Consumed by |
|---|---|---|
| `part1_model/` | part-1 | part-2, part-3, part-4 |
| `part4_rw_model/`, `part4_os_model/` | part-4 | part-4 (selection) |
| `part4_best_model/` | part-4 | part-5, pipeline |
| `best_probs.npy` | part-4 | part-5 |
| `calibrator.pkl` | part-5 | pipeline |

---

## How to Reproduce

> **Run the notebooks in order.** Each part depends on artefacts saved by the previous one.

### 1. Set up the environment

```bash
# Clone / download the notebooks into a single directory, then:
pip install -r requirements.txt
```

For GPU training (recommended for parts 1, 3, 4):
- Ensure CUDA 12.x drivers are installed.
- The `torch==2.5.1` wheel in `requirements.txt` targets CUDA 12.x.
- CPU-only alternative: replace the `torch` line — see the comment inside `requirements.txt`.

### 2. Prepare the dataset

The project uses the **Jigsaw Toxicity** dataset hosted as a Kaggle dataset (`sourceId: 15840184`).

- In Kaggle: add the dataset via *Add Data → Dataset* in each notebook's settings.
- Locally: download via the Kaggle API and place files at the path the notebooks expect:

```bash
kaggle datasets download -d <dataset-slug> -p ./data --unzip
```

Update the dataset path constants at the top of each notebook if running outside Kaggle.

### 3. Run the notebooks

```
part-1-rai.ipynb   →   part-2-rai.ipynb   →   part-3-rai.ipynb
                                                      ↓
                   pipeline.ipynb         ←   part-5-rai.ipynb   ←   part-4-rai.ipynb
```

Each notebook prints its GPU name and device on startup — verify `Device: cuda` appears if you intend to use GPU.

### 4. Run the pipeline self-test

After parts 4 and 5 have completed and saved their artefacts:

```bash
python pipeline.py   # or execute the pipeline.ipynb notebook
```

Expected output:
```
=== pipeline.py Self-test ===
  BLOCK  | layer=input_filter   | conf=1.000 | I will kill you if you show up
  BLOCK  | layer=input_filter   | conf=1.000 | go kill yourself
  BLOCK  | layer=input_filter   | conf=1.000 | I know where you live and I'll post your address
  ALLOW  | layer=model          | conf=0.082 | Have a wonderful day, hope you're doing well!
  ALLOW  | layer=model          | conf=0.069 | This policy is absolutely stupid and wrong
```

---

## Pipeline Overview

The `ModerationPipeline` applies three layers in sequence:

| Layer | Mechanism | Decision |
|-------|-----------|----------|
| 1 — Input Filter | Regex blocklist (5 harm categories) | `block` (conf = 1.0) |
| 2 — Model | Calibrated DistilBERT · prob ≥ 0.6 → block · prob ≤ 0.4 → allow | `block` / `allow` |
| 3 — Human Review | Uncertain cases (0.4 < prob < 0.6) | `review` |

```python
from pipeline import load_pipeline

pipe = load_pipeline(
    model_dir       = "/kaggle/working/part4_best_model",
    calibrator_path = "/kaggle/working/calibrator.pkl",
)
result = pipe.predict("some comment text")
# → {"decision": "block"|"allow"|"review",
#    "layer": "input_filter"|"model"|"human_review",
#    "confidence": float,
#    "category": str}   # only present for input_filter blocks
```

---

## Reproducibility Notes

- Global seed `42` is set in every notebook (`random`, `numpy`, `torch`, `torch.cuda`).
- Kaggle's environment pins the full image; the versions in `requirements.txt` match `dockerImageVersionId 31329`.
- Parts 2 and 5 (no GPU required) can be run on CPU by setting `accelerator: none` in Kaggle settings or simply running locally without a GPU.
