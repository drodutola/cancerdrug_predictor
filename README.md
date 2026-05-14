# Cancer Drug Predictor 🧬

A machine learning application that predicts optimal drug candidates for **breast, ovarian, and colon cancers** based on tumor characteristics and patient features.

## Overview

This tool leverages supervised ML models trained on oncology datasets to assist in identifying likely drug responders — supporting the kind of precision medicine decisions that are increasingly relevant in clinical trial settings.

## Features

- Multi-cancer type support: breast, ovarian, and colon cancer
- Drug response prediction based on molecular and clinical features
- Clean Python-based pipeline from data ingestion to prediction output

## Tech Stack

- **Language:** Python
- **Libraries:** scikit-learn, Pandas, NumPy
- **Deployment:** Heroku (see `cancer_app_heroku`)

## How to Run

```bash
git clone https://github.com/drodutola/cancerdrug_predictor
cd cancerdrug_predictor
pip install -r requirements.txt
python predict.py
```

## Background

Built as part of an applied AI initiative to explore ML-driven oncology decision support tools. Complements hands-on involvement in 8+ Phase 2/3 oncology clinical trials.

## Author

**Dr. Peter Odutola, M.D.** — Physician, AI developer, and clinical researcher.  
[GitHub Profile](https://github.com/drodutola)
