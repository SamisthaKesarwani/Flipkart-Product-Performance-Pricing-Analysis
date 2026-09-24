# 🛒 Flipkart Product Performance & Pricing Analysis

> **AICTE IBM SkillsBuild Internship Project**  
> Identify top-performing and underperforming products, uncover pricing patterns, determine drivers of high ratings, and generate actionable sales recommendations — all from real customer review data.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Analysis Sections](#analysis-sections)
- [Key Findings](#key-findings)
- [Charts Generated](#charts-generated)
- [Tech Stack](#tech-stack)
- [Screenshots](#screenshots)

---

## Project Overview

This project performs an end-to-end product performance analysis on Flipkart customer reviews. It answers four core business questions:

| Question | Method |
|---|---|
| Which products are top performers vs underperformers? | Composite score = z(rating) + z(sentiment%) + z(log(reviews)) |
| What pricing patterns exist across categories? | Price tiers (Budget / Mid / Premium) via tertile quantiles |
| What characteristics drive high ratings? | Gradient Boosting Regressor — feature importance |
| Why do products underperform? | TF-IDF on negative reviews → complaint keyword extraction |

---

## Dataset

| Property | Value |
|---|---|
| File | `Dataset-SA.csv` |
| Source | [Kaggle — Flipkart Product Customer Reviews](https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset) by niraliivaghani |
| Rows | ~170,000 (after cleaning) |
| Product types | 104 |
| Sentiment | Pre-labelled as Positive / Neutral / Negative |

**Columns:**

| Column | Description |
|---|---|
| `product_name` | Full product name as listed on Flipkart |
| `price` | Listed price in ₹ (may contain currency symbols) |
| `rate` | Customer star rating (1–5) |
| `review` | Full review text (may be null if summary exists) |
| `summary` | Short review headline (may be null if review exists) |
| `sentiment` | Pre-labelled sentiment: Positive / Neutral / Negative |

---

## Project Structure

```
.
├── Samistha_flipkartproductperformance&pricinganalysis.py   # Main Streamlit app
├── Dataset-SA.csv                                           # Dataset
├── requirements.txt                                         # Pinned dependencies
├── project_report.docx                                      # Full project report (Word)
├── README.md                                                # This file
└── charts/                                                  # Auto-generated chart PNGs
    ├── aggregated_data_sample.png
    ├── category_distribution.png
    ├── top10_products_chart.png
    ├── bottom10_products_chart.png
    ├── price_vs_rating_scatter.png
    ├── price_by_category_boxplot.png
    ├── price_tier_rating_chart.png
    ├── price_tier_sentiment_chart.png
    ├── correlation_heatmap.png
    ├── feature_importance_chart.png
    ├── negative_keywords_bar.png
    ├── negative_keywords_wordcloud.png
    └── notebook_widget_demo.png
```

---

## Setup & Installation

**Prerequisites:** Python 3.11+

```bash
# 1. Clone or download this project folder

# 2. (Recommended) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Option A — Streamlit Web App (recommended)

```bash
streamlit run "Samistha_flipkartproductperformance&pricinganalysis.py"
```

Opens at **http://localhost:8501** in your browser.

- Use the **sidebar file uploader** to upload any Flipkart-style CSV, or leave it empty to use `Dataset-SA.csv` automatically.
- Use the **Category dropdown**, **Min Reviews slider**, and **Top/Bottom toggle** in the sidebar to explore products interactively.

### Option B — Save all charts to disk (no browser needed)

```bash
python "Samistha_flipkartproductperformance&pricinganalysis.py" --save-charts
```

This runs the full pipeline headlessly and saves all 13 chart PNGs into the `charts/` folder.

---

## Analysis Sections

| Section | What it covers |
|---|---|
| **1 — Setup & Data Loading** | Load CSV, print column names, display shape and head |
| **2 — Data Cleaning** | Standardise columns, clean price, handle nulls, deduplicate, normalise sentiment |
| **3 — Category Extraction** | Assign 13 categories from product name via keyword dictionary |
| **4 — Product Aggregation** | Group by product: avg rating, review count, avg price, sentiment % |
| **5 — Top vs Underperforming** | Composite-score ranking, top-10 and bottom-10 tables and charts |
| **6 — Pricing Patterns** | Scatter, box plot, price tier rating bar, sentiment-per-tier stacked bar |
| **7 — What Drives High Ratings** | Pearson correlation heatmap + Gradient Boosting feature importance |
| **8 — Root-Cause Analysis** | TF-IDF on negative reviews, keyword bar chart, word cloud |
| **9 — Recommendations** | 7 auto-generated data-backed bullet recommendations |
| **10 — Interactive Frontend** | Live-filtering explorer: category dropdown, min-reviews slider, top/bottom toggle |

---

## Key Findings

- **Positive Sentiment % is the strongest rating predictor** (feature importance = 0.57, correlation r = 0.82). Quality perception matters far more than price.
- **Mid-range tier products achieve the highest average ratings.** Premium products attract more negative sentiment when quality doesn't justify price.
- **Top categories:** Watches and Smartphone Accessories (avg rating > 4.3, positive sentiment > 70%).
- **Underperforming categories:** Home & Kitchen and Clothing (avg rating < 3.0, negative sentiment > 40%).
- **Top complaint keywords:** "poor quality", "damaged packaging", "not as described", "size issue", "refund delayed".
- **Model accuracy:** MAE = 0.115, R² = 0.892 on held-out test set (20% split).

---

## Charts Generated

| File | Description |
|---|---|
| `category_distribution.png` | Review count per category (horizontal bar) |
| `aggregated_data_sample.png` | Top-10 products KPI table figure |
| `top10_products_chart.png` | Top 10: avg rating + sentiment bars |
| `bottom10_products_chart.png` | Bottom 10: avg rating + sentiment bars |
| `price_vs_rating_scatter.png` | Price vs Rating scatter, coloured by category |
| `price_by_category_boxplot.png` | Price distribution box plot per category |
| `price_tier_rating_chart.png` | Avg rating per price tier bar chart |
| `price_tier_sentiment_chart.png` | Sentiment % breakdown per price tier (stacked bar) |
| `correlation_heatmap.png` | Pearson correlation heatmap |
| `feature_importance_chart.png` | GBR feature importance bar chart |
| `negative_keywords_bar.png` | Top 20 TF-IDF negative keyword bar chart |
| `negative_keywords_wordcloud.png` | Word cloud of negative keywords |
| `notebook_widget_demo.png` | Interactive explorer demo chart |

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | 1.59.2 | Web app framework — interactive frontend |
| `pandas` | 3.0.3 | Data ingestion, cleaning, aggregation |
| `numpy` | 2.4.6 | Numerical computation |
| `matplotlib` | 3.11.2 | Static chart generation |
| `seaborn` | 0.13.2 | Statistical visualisations (heatmap, boxplot) |
| `scikit-learn` | 1.9.1 | GBR model, TF-IDF, LabelEncoder, train/test split |
| `wordcloud` | 1.9.6 | Negative keyword word cloud |

---

## Screenshots

| Chart | Preview path |
|---|---|
| Category Distribution | `charts/category_distribution.png` |
| Top 10 Products | `charts/top10_products_chart.png` |
| Feature Importance | `charts/feature_importance_chart.png` |
| Word Cloud | `charts/negative_keywords_wordcloud.png` |
| Interactive Explorer | `charts/notebook_widget_demo.png` |

---

## Report

The full project write-up is in **`project_report.docx`**, covering:
- Business problem & objectives
- Dataset description
- Methodology (5-stage pipeline)
- All analysis sections with embedded chart images
- 7 data-backed recommendations
- Conclusion & limitations

---

*AICTE IBM SkillsBuild Internship — Flipkart Product Performance & Pricing Analysis*
