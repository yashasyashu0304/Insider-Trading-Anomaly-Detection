# Insider Trading Anomaly Detection
### Unsupervised ML on Real SEBI PIT Regulatory Data | Python · Scikit-learn · TensorFlow · Tableau

[![Tableau Dashboard](https://img.shields.io/badge/Tableau-Dashboard-blue)](https://public.tableau.com/app/profile/siddanth.s8417/viz/AnomolyDetectioinusingNSEdata/Dashboard3)

---

## What This Project Is

An end-to-end anomaly detection pipeline that analyses **210,392 real insider trading disclosures** filed with SEBI (Securities and Exchange Board of India) under the Prohibition of Insider Trading (PIT) Regulations 2015.

Three unsupervised ML models flag statistically unusual trades. Their outputs are validated against subsequent stock price movement — because no ground-truth "confirmed insider trading" labels exist in the real world.

> **Why unsupervised?** SEBI enforcement orders are rare, public, and lag by years. Framing this as supervised classification would be incorrect — the absence of a label does not mean a trade was clean.

---

## Key Findings

| Company | Insider | Trade | 30-Day Price Move |
|---------|---------|-------|-------------------|
| **Marine Electricals (India) Ltd** | KDU Enterprises Pvt Ltd | Buy ₹1.87Cr | **+74.9%** |
| **GPT Infraprojects Ltd** | GPT Sons Private Ltd | Buy ₹5.5Cr | **+24.7%** |
| **Whirlpool of India Ltd** | Whirlpool Mauritius Ltd | Sale ₹14.9Cr | **-20.5%** |
| **Almondz Global Securities** | Avonmore Capital | Sale/Pledge | **+44.1%** |
| **Sumeet Industries Ltd** | Padmini Polytex Pvt Ltd | Buy ₹43.9Cr | +2.3% *(false positive — highest model score, lowest price move)* |

**3 of 17 flagged trades preceded price moves of 20%+ within 30 days.**
The Sumeet Industries case demonstrates that a high anomaly score ≠ confirmed insider trading — the model surfaces candidates for human review, not verdicts.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data cleaning & EDA | Python, Pandas, NumPy |
| Traditional ML | Scikit-learn (Isolation Forest, One-Class SVM + Nystroem kernel) |
| Deep Learning | TensorFlow / Keras (Autoencoder) |
| Price validation | yfinance |
| Visualisation | Tableau Public |
| Storage | SQLite, CSV |

---

## Pipeline

```
Raw SEBI PIT CSV (210K rows)
        │
        ▼
 1. DATA CLEANING
    - Strip Indian-format commas ("1,24,60,392" → 12460392)
    - Replace "-" with 0, parse dates, filter to Equity only
    - Derive transaction_type (buy / sale_or_pledge_invoke / pledge / mixed)
        │
        ▼
 2. FEATURE ENGINEERING
    - trade_value: combine buy/sale/pledge into one number
    - trade_value_zscore: how unusual vs this insider's own history
    - days_since_last_trade: time gap between this insider's trades
    - cluster_signal: co-traders at same company within ±5 days
    - is_promoter / is_designated_person: category encoding
        │
        ▼
 3. MODELLING (Unsupervised — no labels)
    ┌─────────────────┐  ┌──────────────┐  ┌─────────────────────┐
    │ Isolation Forest│  │One-Class SVM │  │ Autoencoder (Keras) │
    │ n_estimators=200│  │SGD + Nystroem│  │ 8→4→2→4→8 layers   │
    │                 │  │ kernel approx│  │ 25 epochs, MSE loss │
    └────────┬────────┘  └──────┬───────┘  └──────────┬──────────┘
             └──────────────────┼──────────────────────┘
                                ▼
                    Ensemble Score (normalised avg)
                    models_agreed (0-3 model votes)
        │
        ▼
 4. PRICE VALIDATION (post-hoc, not a training feature)
    - Map company names → NSE/BSE tickers via yfinance Search
    - Compute 5 / 10 / 30-day forward returns after each flagged trade
    - Validate: do high-scoring trades precede large price moves?
        │
        ▼
 5. TABLEAU DASHBOARD
    - Company ranking by anomaly score
    - Scatter: ensemble score vs 30-day price return (validation proof)
    - Flagged trades table with forward returns
    - Transaction type breakdown (donut chart)
```

---

## Model Design Decisions

**Why Isolation Forest?**
Random splits isolate anomalies in fewer steps than normal points. No distance metric needed — works well with mixed feature scales.

**Why One-Class SVM with Nystroem kernel?**
Classic One-Class SVM is O(n²) — unusable on 210K rows. Nystroem approximates an RBF kernel via random feature maps, giving non-linear boundaries at O(n) cost.

**Why an Autoencoder?**
Trained to reconstruct its own input, it learns the "grammar" of normal trades. Anomalous trades produce high reconstruction error because their patterns were never reinforced during training. Bottleneck of 2 nodes forces compression — if the 2-number representation can't capture a trade's pattern, reconstruction fails.

**Why ensemble?**
Each model has different inductive biases. A trade flagged by 2+ models is a stronger signal than one flagged by only one.

---

## Known Limitations (Data Literacy)

| Limitation | Detail |
|-----------|--------|
| **Conflated columns** | SEBI's `SALE / PLEDGE INVOCATION` column combines ordinary sales with forced pledge seizures — indistinguishable from disclosure data alone |
| **Pledge ambiguity** | `PLEDGE CREATION / RELEASE` combines new pledges (bearish signal) with pledge payoffs (neutral) |
| **No ground-truth labels** | Model scores statistical outliers, not confirmed violations — validated by price proxy only |
| **Z-score includes current trade** | In-sample z-score slightly dilutes anomaly scores for insiders with few trades (production system would use expanding window) |

---

## Results Summary

```
Total trades analysed:    210,392
Unique companies:         ~4,800
Flagged by 2+ models:     17 trades
Flagged by all 3 models:  0 trades  ← models disagree on edge cases (expected)

Of 17 flagged trades:
  - 3 preceded 20%+ price moves within 30 days  (strong signal)
  - 5 preceded 10-20% price moves               (moderate signal)
  - 9 showed <10% price movement                (weak / false positive)
```

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Siddanth-S/Insider-Trading-Anomaly-Detection.git
cd Insider-Trading-Anomaly-Detection

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install pandas numpy scikit-learn tensorflow yfinance tqdm openpyxl

# 4. Download raw data
# Go to: https://www.nseindia.com/companies-listing/corporate-filings-insider-trading
# Download: "CF-PIT-Annual-equities" CSV for your desired date range
# Place in the project root folder

# 5. Run the notebook
jupyter lab insider_trading_anomaly.ipynb
# Run cells top to bottom — each step saves its output as CSV
```

---

## Data Source

**SEBI PIT (Prohibition of Insider Trading) Annual Disclosures**
NSE India Corporate Filings → Insider Trading (PIT) → Annual
[nseindia.com/companies-listing/corporate-filings-insider-trading](https://www.nseindia.com/companies-listing/corporate-filings-insider-trading)

Real regulatory data — not a Kaggle dataset.

---

## Dashboard

**Live Tableau Public Dashboard →** [View Here](https://public.tableau.com/app/profile/siddanth.s8417/viz/AnomolyDetectioinusingNSEdata/Dashboard3)

