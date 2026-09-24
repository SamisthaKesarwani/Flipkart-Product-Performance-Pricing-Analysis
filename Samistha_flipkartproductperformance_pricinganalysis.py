"""
Flipkart Product Performance Analysis

Author: Samistha Kesarwani (Riya)
Project: AICTE IBM SkillsBuild Data Analytics with AI Internship
Built with: IBM BOB

Run:  streamlit run analysis.py
"""

# Section 1: Setup & Data Loading
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

# Only import streamlit when not running in chart-save mode
_SAVE_MODE = "--save-charts" in sys.argv
if not _SAVE_MODE:
    import streamlit as st

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150})

CSV_PATH    = "Dataset-SA.csv"   # fallback when no file is uploaded
CHARTS_DIR  = "charts"           # folder where PNGs are saved


# Section 2: Data Cleaning
def _load_and_clean_impl(path) -> pd.DataFrame:
    """Accept a file path (str) or a file-like object (UploadedFile)."""
    raw = pd.read_csv(path, encoding="utf-8", low_memory=False)

    # Standardise column names
    raw.columns = (
        raw.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    _col_map = {
        "product_name": ["product_name", "productname", "product name", "name"],
        "price":        ["price", "product_price", "productprice"],
        "rate":         ["rate", "rating", "ratings", "product_rating"],
        "review":       ["review", "reviews", "review_text"],
        "summary":      ["summary", "review_summary"],
        "sentiment":    ["sentiment", "sentiments"],
    }
    rename = {}
    for std, variants in _col_map.items():
        for v in variants:
            if v in raw.columns and std not in raw.columns:
                rename[v] = std
    raw.rename(columns=rename, inplace=True)

    # Clean Price
    raw["price"] = (
        raw["price"].astype(str)
        .str.replace(r"[Rs.$,\s]", "", regex=True)
        .str.replace(r"[^0-9.]", "", regex=True)
    )
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")

    # Clean Rate
    raw["rate"] = pd.to_numeric(raw["rate"], errors="coerce")

    # Keep rows with at least one text field OR a rating
    has_text = raw["review"].notna() | raw["summary"].notna()
    raw = raw[has_text | raw["rate"].notna()].copy()

    # Standardise Sentiment
    raw["sentiment"] = raw["sentiment"].astype(str).str.strip().str.capitalize()
    valid = {"Positive", "Neutral", "Negative"}
    raw.loc[~raw["sentiment"].isin(valid), "sentiment"] = np.nan

    # Deduplicate
    raw.drop_duplicates(inplace=True)
    raw.reset_index(drop=True, inplace=True)
    return raw


if not _SAVE_MODE:
    import streamlit as _st_ref
    @_st_ref.cache_data(show_spinner="Loading & cleaning data...")
    def load_and_clean(path: str) -> pd.DataFrame:
        return _load_and_clean_impl(path)
else:
    def load_and_clean(path: str) -> pd.DataFrame:
        return _load_and_clean_impl(path)


# Section 3: Category Extraction
KEYWORD_MAP = {
    "Smartphone & Accessories": [
        "phone", "mobile", "smartphone", "iphone", "redmi", "samsung", "oneplus",
        "realme", "oppo", "vivo", "motorola", "nokia", "mi ", "poco", "iqoo",
        "charger", "power bank", "earphone", "headphone", "earbuds", "cable",
        "screen guard", "back cover", "phone case",
    ],
    "Laptop & Computer": [
        "laptop", "notebook", "computer", "desktop", "chromebook", "macbook",
        "keyboard", "mouse", "monitor", "webcam", "hard disk", "ssd", "pendrive",
        "ram", "processor", "router", "modem",
    ],
    "Watches": [
        "watch", "smartwatch", "fitness band", "wristwatch", "timepiece",
    ],
    "Footwear": [
        "shoe", "shoes", "sneaker", "sandal", "slipper", "boot", "loafer",
        "heels", "flip flop", "chappal",
    ],
    "Clothing & Apparel": [
        "shirt", "t-shirt", "tshirt", "kurta", "saree", "dress", "jeans",
        "trouser", "pant", "legging", "jacket", "hoodie", "top ", "tops",
        "suit", "ethnic", "kurti", "palazzo", "skirt", "blouse", "salwar",
        "underwear", "innerwear", "bra", "briefs", "socks",
    ],
    "Bags & Luggage": [
        "bag", "handbag", "backpack", "purse", "wallet", "luggage", "trolley",
        "suitcase", "clutch", "tote",
    ],
    "Home & Kitchen": [
        "sofa", "bed", "mattress", "pillow", "curtain", "bedsheet", "towel",
        "blanket", "cushion", "chair", "table", "lamp", "light", "fan",
        "mixer", "grinder", "juicer", "oven", "microwave", "fridge",
        "refrigerator", "washing machine", "iron", "cooker", "cookware",
        "utensil", "pan ", "pot ", "knife", "bottle", "container", "flask",
        "kettle", "toaster", "coffee maker", "water purifier", "vacuum",
    ],
    "Home Decor": [
        "decor", "decoration", "wall art", "photo frame", "candle", "vase",
        "showpiece", "figurine", "idol", "painting", "wall clock", "poster",
        "doormat", "rug", "carpet",
    ],
    "Beauty & Personal Care": [
        "cream", "lotion", "serum", "shampoo", "conditioner", "moisturizer",
        "face wash", "sunscreen", "lipstick", "makeup", "foundation", "mascara",
        "perfume", "deodorant", "razor", "trimmer", "hair dryer", "hair oil",
        "body wash", "soap", "toothbrush", "toothpaste",
    ],
    "Sports & Fitness": [
        "cricket", "football", "badminton", "tennis", "gym", "dumbbell",
        "yoga", "cycle", "bicycle", "treadmill", "skipping rope", "gloves",
        "jersey", "sports", "fitness", "protein", "supplement",
    ],
    "Books & Stationery": [
        "book", "novel", "diary", "pen ", "pencil", "marker",
        "highlighter", "stationery",
    ],
    "Toys & Baby": [
        "toy", "doll", "lego", "puzzle", "baby", "infant", "diaper",
        "stroller", "board game", "action figure",
    ],
    "Automotive": [
        "car ", "bike ", "helmet", "tyre", "tire", "seat cover", "dashboard",
        "motor oil", "air freshener", "vehicle",
    ],
}


def assign_category(name: str) -> str:
    if not isinstance(name, str):
        return "Other"
    nl = name.lower()
    for cat, keywords in KEYWORD_MAP.items():
        for kw in keywords:
            if kw in nl:
                return cat
    return "Other"


# Section 4: Product-Level Aggregation
def _build_aggregates_impl(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["category"] = df["product_name"].apply(assign_category)
    for s in ["Positive", "Neutral", "Negative"]:
        df[f"is_{s.lower()}"] = (df["sentiment"] == s).astype(int)

    agg = (
        df.groupby(["product_name", "category"], as_index=False)
        .agg(
            avg_rating   =("rate",         "mean"),
            review_count =("rate",         "count"),
            avg_price    =("price",        "mean"),
            min_price    =("price",        "min"),
            max_price    =("price",        "max"),
            pos_pct      =("is_positive",  "mean"),
            neu_pct      =("is_neutral",   "mean"),
            neg_pct      =("is_negative",  "mean"),
        )
    )
    for col in ["pos_pct", "neu_pct", "neg_pct"]:
        agg[col] = (agg[col] * 100).round(1)
    agg["avg_rating"] = agg["avg_rating"].round(2)
    agg["avg_price"]  = agg["avg_price"].round(2)
    return agg


if not _SAVE_MODE:
    @_st_ref.cache_data(show_spinner="Aggregating products...")
    def build_aggregates(df: pd.DataFrame) -> pd.DataFrame:
        return _build_aggregates_impl(df)
else:
    def build_aggregates(df: pd.DataFrame) -> pd.DataFrame:
        return _build_aggregates_impl(df)


# Helpers
def z_norm(s: pd.Series) -> pd.Series:
    std = s.std()
    return (s - s.mean()) / std if std > 0 else s * 0


def composite_score(df: pd.DataFrame) -> pd.Series:
    return (
        z_norm(df["avg_rating"])
        + z_norm(df["pos_pct"])
        + z_norm(np.log1p(df["review_count"]))
    )


def short_name(n: str, max_len: int = 30) -> str:
    return n[:max_len] + "..." if len(n) > max_len else n


def savefig(fig, name: str):
    """Save figure to CHARTS_DIR/<name>.png and close it."""
    os.makedirs(CHARTS_DIR, exist_ok=True)
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  Saved → {path}")
    plt.close(fig)


# Section 5: Top vs Underperforming Products
def get_top_bottom(prod_agg: pd.DataFrame, min_reviews: int = 30):
    sufficient = prod_agg[prod_agg["review_count"] >= min_reviews].copy()
    sufficient["_score"] = composite_score(sufficient)
    top10    = sufficient.nlargest(10, "_score").reset_index(drop=True)
    bottom10 = sufficient.nsmallest(10, "_score").reset_index(drop=True)
    return top10, bottom10


def plot_top_bottom(data: pd.DataFrame, title: str, color: str,
                    save_as: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    labels = [short_name(n) for n in data["product_name"]]

    axes[0].barh(labels, data["avg_rating"], color=color, edgecolor="white")
    axes[0].set_xlim(0, 5.5)
    axes[0].set_xlabel("Average Rating")
    axes[0].set_title(f"{title} - Avg Rating", fontweight="bold")
    for i, v in enumerate(data["avg_rating"]):
        axes[0].text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8)

    y_pos = np.arange(len(labels))
    axes[1].barh(y_pos, data["pos_pct"], color="#2ecc71", label="Positive", edgecolor="white")
    axes[1].barh(y_pos, data["neu_pct"], left=data["pos_pct"],
                 color="#f39c12", label="Neutral", edgecolor="white")
    axes[1].barh(y_pos, data["neg_pct"], left=data["pos_pct"] + data["neu_pct"],
                 color="#e74c3c", label="Negative", edgecolor="white")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel("Sentiment %")
    axes[1].set_title(f"{title} - Sentiment Breakdown", fontweight="bold")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].set_xlim(0, 105)
    plt.tight_layout()
    if save_as:
        savefig(fig, save_as)
        return None
    return fig


# Section 6: Pricing Pattern Analysis
def pricing_analysis(df: pd.DataFrame):
    df_p = df.dropna(subset=["price", "rate"]).copy()
    df_p["price_tier"] = pd.qcut(
        df_p["price"], q=3, labels=["Budget", "Mid", "Premium"]
    )
    return df_p


def plot_scatter(df_p: pd.DataFrame, save_as: str = None):
    sample = df_p.sample(min(8000, len(df_p)), random_state=42)
    cats = sample["category"].unique()
    palette = dict(zip(cats, sns.color_palette("tab20", len(cats))))
    fig, ax = plt.subplots(figsize=(12, 6))
    for cat, grp in sample.groupby("category"):
        ax.scatter(grp["price"], grp["rate"], alpha=0.35, s=18,
                   label=cat, color=palette[cat])
    ax.set_xlabel("Price (Rs.)")
    ax.set_ylabel("Rating")
    ax.set_title("Price vs Rating by Category (sample)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rs.{int(x):,}"))
    plt.tight_layout()
    if save_as:
        savefig(fig, save_as)
        return None
    return fig


def plot_boxplot(df_p: pd.DataFrame, save_as: str = None):
    cat_counts = df_p["category"].value_counts()
    major_cats = cat_counts[cat_counts >= 200].index.tolist()
    bp_data = df_p[df_p["category"].isin(major_cats)]
    order = bp_data.groupby("category")["price"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=bp_data, x="category", y="price", order=order,
                palette="tab20", fliersize=2, ax=ax)
    ax.set_ylim(0, bp_data["price"].quantile(0.97))
    ax.set_xlabel("")
    ax.set_ylabel("Price (Rs.)")
    ax.set_title("Price Distribution by Category (capped at 97th percentile)", fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rs.{int(x):,}"))
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    if save_as:
        savefig(fig, save_as)
        return None
    return fig


def plot_tier_rating(df_p: pd.DataFrame, save_as: str = None):
    tier_rating = df_p.groupby("price_tier", observed=True)["rate"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(tier_rating["price_tier"], tier_rating["rate"],
                  color=["#3498db", "#f39c12", "#8e44ad"], edgecolor="white", width=0.5)
    ax.set_ylim(0, 5.2)
    ax.set_ylabel("Average Rating")
    ax.set_title("Average Rating by Price Tier", fontweight="bold")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{bar.get_height():.2f}", ha="center", fontsize=10)
    plt.tight_layout()
    if save_as:
        savefig(fig, save_as)
        return None
    return fig


def plot_tier_sentiment(df_p: pd.DataFrame, save_as: str = None):
    sent_tier = (
        df_p.dropna(subset=["sentiment"])
        .groupby(["price_tier", "sentiment"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    sent_tier_pct = sent_tier.div(sent_tier.sum(axis=1), axis=0) * 100
    for col in ["Positive", "Neutral", "Negative"]:
        if col not in sent_tier_pct.columns:
            sent_tier_pct[col] = 0
    sent_tier_pct = sent_tier_pct[["Positive", "Neutral", "Negative"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    sent_tier_pct.plot(kind="bar", stacked=True, ax=ax,
                       color=["#2ecc71", "#f39c12", "#e74c3c"], edgecolor="white", rot=0)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Sentiment Breakdown by Price Tier", fontweight="bold")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    if save_as:
        savefig(fig, save_as)
        return None
    return fig


def plot_category_dist(raw: pd.DataFrame, save_as: str = None):
    cat_dist = raw["category"].value_counts()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(cat_dist.index[::-1], cat_dist.values[::-1],
            color=sns.color_palette("tab20", len(cat_dist)))
    ax.set_title("Product Review Count by Category", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Reviews")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    if save_as:
        savefig(fig, save_as)
        return None
    return fig


def plot_aggregated_sample(prod_agg: pd.DataFrame, save_as: str = None):
    """Render the top-10 aggregated products table as a figure."""
    sample = prod_agg.sort_values("review_count", ascending=False).head(10)
    cols   = ["product_name", "category", "avg_rating",
              "review_count", "avg_price", "pos_pct", "neg_pct"]
    sample = sample[cols].copy()
    sample["product_name"] = sample["product_name"].apply(lambda x: short_name(x, 35))

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    tbl = ax.table(
        cellText=sample.values,
        colLabels=["Product", "Category", "Avg Rating",
                   "Reviews", "Avg Price", "Pos%", "Neg%"],
        cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#dce6f0")
    ax.set_title("Top 10 Products by Review Count - Aggregated KPIs",
                 fontweight="bold", pad=10, fontsize=11)
    plt.tight_layout()
    if save_as:
        savefig(fig, save_as)
        return None
    return fig


# Section 7: What Drives High Ratings
def plot_heatmap(prod_agg: pd.DataFrame, save_as: str = None):
    corr_cols = ["avg_rating", "avg_price", "review_count", "pos_pct"]
    corr_df = prod_agg[corr_cols].dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("Correlation: Price, Rating, Review Count, Positive Sentiment %",
                 fontweight="bold", pad=12)
    plt.tight_layout()
    if save_as:
        savefig(fig, save_as)
        return None
    return fig


def _train_model_impl(prod_agg: pd.DataFrame):
    model_df = prod_agg[["avg_rating", "avg_price", "review_count",
                          "pos_pct", "category"]].dropna().copy()
    le = LabelEncoder()
    model_df["category_enc"] = le.fit_transform(model_df["category"])
    features = ["avg_price", "review_count", "pos_pct", "category_enc"]
    X = model_df[features]
    y = model_df["avg_rating"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                      learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    return model, features, mae, r2


if not _SAVE_MODE:
    @_st_ref.cache_data(show_spinner="Training model...")
    def train_model(prod_agg: pd.DataFrame):
        return _train_model_impl(prod_agg)
else:
    def train_model(prod_agg: pd.DataFrame):
        return _train_model_impl(prod_agg)


def plot_feature_importance(model, feat_labels, save_as: str = None):
    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([feat_labels[i] for i in sorted_idx],
            importances[sorted_idx], color="#3b82d4", edgecolor="white")
    ax.set_xlabel("Feature Importance")
    ax.set_title("Drivers of High Ratings (Gradient Boosting)", fontweight="bold")
    for i, (idx, v) in enumerate(zip(sorted_idx, importances[sorted_idx])):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    if save_as:
        savefig(fig, save_as)
        return None, importances
    return fig, importances


# Section 8: Root-Cause Analysis of Underperformance
def _run_tfidf_impl(df: pd.DataFrame, prod_agg: pd.DataFrame):
    cat_avg = prod_agg.groupby("category")["avg_rating"].mean()
    underperf_cats = cat_avg[cat_avg <= cat_avg.quantile(0.25)].index.tolist()

    neg = df[
        (df["sentiment"] == "Negative") &
        (df["category"].isin(underperf_cats))
    ].copy()
    neg["text"] = (neg["review"].fillna("") + " " + neg["summary"].fillna("")).str.strip()
    neg = neg[neg["text"].str.len() > 3]

    if len(neg) < 10:
        return pd.DataFrame(), underperf_cats

    sample = neg["text"].sample(min(20000, len(neg)), random_state=42)
    tfidf = TfidfVectorizer(stop_words="english", max_features=100,
                             ngram_range=(1, 2), min_df=3)
    mat    = tfidf.fit_transform(sample)
    scores = np.asarray(mat.mean(axis=0)).flatten()
    vocab  = tfidf.get_feature_names_out()
    top_df = (
        pd.DataFrame({"term": vocab, "score": scores})
        .sort_values("score", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    return top_df, underperf_cats


if not _SAVE_MODE:
    @_st_ref.cache_data(show_spinner="Running TF-IDF...")
    def run_tfidf(df: pd.DataFrame, prod_agg: pd.DataFrame):
        return _run_tfidf_impl(df, prod_agg)
else:
    def run_tfidf(df: pd.DataFrame, prod_agg: pd.DataFrame):
        return _run_tfidf_impl(df, prod_agg)


def plot_tfidf_bar(top_df: pd.DataFrame, save_as: str = None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top_df["term"][::-1], top_df["score"][::-1],
            color="#e74c3c", edgecolor="white")
    ax.set_xlabel("Mean TF-IDF Score")
    ax.set_title("Top 20 Keywords in Negative Reviews (Underperforming Categories)",
                 fontweight="bold")
    plt.tight_layout()
    if save_as:
        savefig(fig, save_as)
        return None
    return fig


def plot_wordcloud(top_df: pd.DataFrame, save_as: str = None):
    if not HAS_WORDCLOUD or top_df.empty:
        return None
    freq = dict(zip(top_df["term"], top_df["score"]))
    wc   = WordCloud(width=900, height=400, background_color="white",
                     colormap="Reds", max_words=80).generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud - Negative Review Keywords", fontweight="bold", pad=10)
    plt.tight_layout()
    if save_as:
        savefig(fig, save_as)
        return None
    return fig


# Section 9: Recommendations (auto-generated)
def generate_recommendations(prod_agg, df_p, importances, feat_labels):
    recs = []
    cat_stats = prod_agg.groupby("category").agg(
        avg_rating   =("avg_rating",   "mean"),
        avg_price    =("avg_price",    "mean"),
        avg_neg_pct  =("neg_pct",      "mean"),
        avg_pos_pct  =("pos_pct",      "mean"),
        total_reviews=("review_count", "sum"),
    ).round(2)

    global_avg_rating = cat_stats["avg_rating"].mean()
    global_avg_price  = cat_stats["avg_price"].mean()

    for cat, row in cat_stats.iterrows():
        if row["avg_price"] > global_avg_price * 1.2 and row["avg_rating"] < global_avg_rating:
            recs.append(
                f"💰 **PRICING** - *{cat}* is priced above average "
                f"(Rs.{row['avg_price']:,.0f}) yet rated below average "
                f"({row['avg_rating']:.2f}/5). Consider price rationalisation or quality improvement."
            )

    threshold_rc = prod_agg["review_count"].quantile(0.75)
    high_vol_low = prod_agg[
        (prod_agg["review_count"] >= threshold_rc) &
        (prod_agg["avg_rating"] < global_avg_rating - 0.3)
    ].nlargest(5, "review_count")
    for _, row in high_vol_low.iterrows():
        recs.append(
            f"⚠️  **ATTENTION** - *{row['product_name'][:50]}* ({row['category']}) "
            f"has {int(row['review_count']):,} reviews but only {row['avg_rating']:.2f}/5. "
            f"Prioritise quality control or product revision."
        )

    tier_rating = df_p.groupby("price_tier", observed=True)["rate"].mean()
    best_tier   = str(tier_rating.idxmax())
    recs.append(
        f"📊 **PRICE TIER** - *{best_tier}* tier products show the highest average rating. "
        f"Promote more products in this tier and highlight value proposition in marketing."
    )

    for cat, row in cat_stats.iterrows():
        if row["avg_neg_pct"] > 30 and row["avg_rating"] >= global_avg_rating - 0.2:
            recs.append(
                f"🔍 **SENTIMENT MISMATCH** - *{cat}* has {row['avg_neg_pct']:.1f}% negative sentiment "
                f"despite avg rating {row['avg_rating']:.2f}. Investigate specific pain points."
            )

    for cat, row in cat_stats.iterrows():
        if row["avg_rating"] >= global_avg_rating + 0.3 and row["avg_pos_pct"] >= 60:
            recs.append(
                f"✅ **REPLICATE SUCCESS** - *{cat}* has high ratings ({row['avg_rating']:.2f}) "
                f"and {row['avg_pos_pct']:.1f}% positive sentiment. "
                f"Use as benchmark for seller onboarding."
            )

    top_driver = feat_labels[np.argmax(importances)]
    recs.append(
        f"🤖 **ML INSIGHT** - *{top_driver}* is the strongest predictor of product rating. "
        f"Focus improvement efforts on this dimension for maximum rating lift."
    )
    return recs


# CHART SAVE MODE  -  python analysis.py --save-charts
def save_all_charts():
    """Run the full pipeline and save every chart as a PNG to ./charts/"""
    print(f"\n{'='*60}")
    print("  Flipkart Analysis - Chart Export Mode")
    print(f"  Output folder: ./{CHARTS_DIR}/")
    print(f"{'='*60}\n")

    print("[1] Loading & cleaning data...")
    raw = load_and_clean(CSV_PATH)
    raw["category"] = raw["product_name"].apply(assign_category)
    print(f"  Rows: {len(raw):,}  |  Columns: {list(raw.columns)}")

    print("[2] Building product aggregates...")
    prod_agg = build_aggregates(raw)
    print(f"  Unique products: {len(prod_agg):,}")

    print("[3] Saving: aggregated_data_sample.png")
    plot_aggregated_sample(prod_agg, save_as="aggregated_data_sample.png")

    print("[4] Saving: category_distribution.png")
    plot_category_dist(raw, save_as="category_distribution.png")

    print("[5] Saving: top10_products_chart.png")
    top10, bottom10 = get_top_bottom(prod_agg, min_reviews=30)
    plot_top_bottom(top10,    "Top 10 Products",    "#2ecc71",
                   save_as="top10_products_chart.png")

    print("[6] Saving: bottom10_products_chart.png")
    plot_top_bottom(bottom10, "Bottom 10 Products", "#e74c3c",
                   save_as="bottom10_products_chart.png")

    print("[7] Building price tier data...")
    df_p = pricing_analysis(raw)

    print("[8] Saving: price_vs_rating_scatter.png")
    plot_scatter(df_p, save_as="price_vs_rating_scatter.png")

    print("[9] Saving: price_by_category_boxplot.png")
    plot_boxplot(df_p, save_as="price_by_category_boxplot.png")

    print("[10] Saving: price_tier_rating_chart.png")
    plot_tier_rating(df_p, save_as="price_tier_rating_chart.png")

    print("[11] Saving: price_tier_sentiment_chart.png")
    plot_tier_sentiment(df_p, save_as="price_tier_sentiment_chart.png")

    print("[12] Saving: correlation_heatmap.png")
    plot_heatmap(prod_agg, save_as="correlation_heatmap.png")

    print("[13] Training model & saving: feature_importance_chart.png")
    model, features, mae, r2 = train_model(prod_agg)
    feat_labels = ["Avg Price", "Review Count", "Positive Sent %", "Category"]
    _, importances = plot_feature_importance(model, feat_labels,
                                             save_as="feature_importance_chart.png")
    print(f"  Model MAE: {mae:.3f}  R2: {r2:.3f}")

    print("[14] Running TF-IDF...")
    top_df, underperf_cats = run_tfidf(raw, prod_agg)
    if not top_df.empty:
        print("[15] Saving: negative_keywords_bar.png")
        plot_tfidf_bar(top_df, save_as="negative_keywords_bar.png")
        if HAS_WORDCLOUD:
            print("[16] Saving: negative_keywords_wordcloud.png")
            plot_wordcloud(top_df, save_as="negative_keywords_wordcloud.png")
        else:
            print("  (wordcloud not installed - skipping wordcloud)")
    else:
        print("  Not enough negative review text for TF-IDF.")

    print(f"\n{'='*60}")
    print(f"  All charts saved to ./{CHARTS_DIR}/")
    print(f"{'='*60}\n")
    return importances, feat_labels, df_p, prod_agg


# Section 10: Streamlit Interactive Frontend
def render_interactive(prod_agg: pd.DataFrame):
    st.markdown("---")
    st.header("🛍️ Section 10 - Interactive Product Explorer")
    st.markdown(
        "Use the controls in the **sidebar** to filter products dynamically. "
        "All charts and tables update instantly."
    )

    all_cats = sorted(prod_agg["category"].unique().tolist())
    sel_cat  = st.sidebar.selectbox("📂 Category", ["All"] + all_cats)
    min_rev  = st.sidebar.slider("🔢 Min Review Count", 0, 500, 20, step=10)
    show_top = st.sidebar.radio("📊 Show", ["Top Products", "Bottom Products"])

    filtered = prod_agg.copy()
    if sel_cat != "All":
        filtered = filtered[filtered["category"] == sel_cat]
    filtered = filtered[filtered["review_count"] >= min_rev]

    if filtered.empty:
        st.warning("No products match the current filters. Try lowering Min Reviews.")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Products",   f"{len(filtered):,}")
    c2.metric("Avg Price",  f"Rs.{filtered['avg_price'].mean():,.0f}")
    c3.metric("Avg Rating", f"{filtered['avg_rating'].mean():.2f} / 5")
    c4.metric("Positive %", f"{filtered['pos_pct'].mean():.1f}%")
    c5.metric("Negative %", f"{filtered['neg_pct'].mean():.1f}%")

    filt_scored = filtered.copy()
    filt_scored["_score"] = composite_score(filt_scored)

    n = 10
    if show_top == "Top Products":
        table     = filt_scored.nlargest(n, "_score").reset_index(drop=True)
        title_str = f"Top {min(n, len(filt_scored))} Products"
        bar_color = "#2ecc71"
    else:
        table     = filt_scored.nsmallest(n, "_score").reset_index(drop=True)
        title_str = f"Bottom {min(n, len(filt_scored))} Products"
        bar_color = "#e74c3c"

    st.subheader(f"📋 {title_str}")
    st.dataframe(
        table[["product_name", "category", "avg_rating",
               "review_count", "avg_price", "pos_pct", "neg_pct"]].rename(columns={
            "product_name": "Product", "category": "Category",
            "avg_rating": "Avg Rating", "review_count": "Reviews",
            "avg_price": "Avg Price (Rs.)", "pos_pct": "Positive %",
            "neg_pct": "Negative %",
        }),
        use_container_width=True
    )

    st.subheader(f"📊 {title_str} - Charts")
    col_left, col_right = st.columns(2)

    fig_r, ax_r = plt.subplots(figsize=(7, 5))
    labels = [short_name(p) for p in table["product_name"]]
    ax_r.barh(labels, table["avg_rating"], color=bar_color, edgecolor="white")
    ax_r.set_xlim(0, 5.5)
    ax_r.set_xlabel("Average Rating")
    ax_r.set_title(f"{title_str} - Avg Rating", fontweight="bold")
    for i, v in enumerate(table["avg_rating"]):
        ax_r.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8)
    plt.tight_layout()
    col_left.pyplot(fig_r)
    plt.close(fig_r)

    fig_s, ax_s = plt.subplots(figsize=(7, 5))
    y_pos = np.arange(len(labels))
    ax_s.barh(y_pos, table["pos_pct"],  color="#2ecc71", label="Positive", edgecolor="white")
    ax_s.barh(y_pos, table["neu_pct"],  left=table["pos_pct"],
              color="#f39c12", label="Neutral", edgecolor="white")
    ax_s.barh(y_pos, table["neg_pct"],  left=table["pos_pct"] + table["neu_pct"],
              color="#e74c3c", label="Negative", edgecolor="white")
    ax_s.set_yticks(y_pos)
    ax_s.set_yticklabels(labels, fontsize=8)
    ax_s.set_xlabel("Sentiment %")
    ax_s.set_title(f"{title_str} - Sentiment Breakdown", fontweight="bold")
    ax_s.legend(loc="lower right", fontsize=8)
    ax_s.set_xlim(0, 105)
    plt.tight_layout()
    col_right.pyplot(fig_s)
    plt.close(fig_s)

    # Save interactive screenshot (widget_demo)
    # Composite screenshot of both charts side by side
    fig_demo, ax_demo = plt.subplots(1, 2, figsize=(14, 5))
    ax_demo[0].barh(labels, table["avg_rating"], color=bar_color, edgecolor="white")
    ax_demo[0].set_xlim(0, 5.5)
    ax_demo[0].set_xlabel("Average Rating")
    ax_demo[0].set_title(f"Interactive Explorer: {title_str} - Rating", fontweight="bold")
    for i, v in enumerate(table["avg_rating"]):
        ax_demo[0].text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8)

    ax_demo[1].barh(y_pos, table["pos_pct"],  color="#2ecc71", label="Positive", edgecolor="white")
    ax_demo[1].barh(y_pos, table["neu_pct"],  left=table["pos_pct"],
                    color="#f39c12", label="Neutral", edgecolor="white")
    ax_demo[1].barh(y_pos, table["neg_pct"],  left=table["pos_pct"] + table["neu_pct"],
                    color="#e74c3c", label="Negative", edgecolor="white")
    ax_demo[1].set_yticks(y_pos)
    ax_demo[1].set_yticklabels(labels, fontsize=8)
    ax_demo[1].set_xlabel("Sentiment %")
    ax_demo[1].set_title(f"Interactive Explorer: {title_str} - Sentiment", fontweight="bold")
    ax_demo[1].legend(loc="lower right", fontsize=8)
    ax_demo[1].set_xlim(0, 105)
    plt.tight_layout()
    savefig(fig_demo, "notebook_widget_demo.png")


# STREAMLIT MAIN APP
def main_streamlit():
    st.set_page_config(
        page_title="Flipkart Product Performance Analysis",
        page_icon="🛒",
        layout="wide",
    )

    st.sidebar.title("🛒 Flipkart Analysis")
    st.sidebar.markdown(
        "**Dataset:** flipkart_reviews.csv\n\n"
        "~205K rows · 104 product types\n\n"
        "Source: Kaggle - niraliivaghani"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Explorer Controls")

    st.title("🛒 Flipkart Product Performance & Pricing Analysis")
    st.markdown(
        "Identify top-performing and underperforming products, uncover pricing patterns, "
        "determine drivers of high ratings, and generate actionable recommendations."
    )

    # Data is always loaded from the local CSV_PATH — no upload widget.
    try:
        raw = load_and_clean(CSV_PATH)
    except Exception as exc:
        st.error(
            f"**`{CSV_PATH}` not found or could not be read.**\n\n"
            f"Place your CSV in the same folder as `analysis.py`, named "
            f"`{CSV_PATH}`, and re-run.\n\nError detail: {exc}"
        )
        st.stop()

    raw["category"] = raw["product_name"].apply(assign_category)
    prod_agg = build_aggregates(raw)

    # Dashboard overview (KPI cards)
    st.markdown("---")
    st.header("📊 Dashboard Overview")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Products", f"{len(prod_agg):,}")
    k2.metric("Total Reviews", f"{int(prod_agg['review_count'].sum()):,}")
    k3.metric("Avg Rating", f"{prod_agg['avg_rating'].mean():.2f} / 5")
    k4.metric("Avg Price", f"Rs. {prod_agg['avg_price'].mean():,.0f}")
    k5.metric("Avg Positive Sentiment", f"{prod_agg['pos_pct'].mean()*100:.1f}%")

    dk1, dk2 = st.columns(2)
    with dk1:
        st.subheader("Category Mix")
        fig_cat_dash = plot_category_dist(raw)
        st.pyplot(fig_cat_dash); plt.close(fig_cat_dash)
    with dk2:
        st.subheader("Avg Rating by Price Tier")
        df_p_dash = pricing_analysis(raw)
        fig_tier_dash = plot_tier_rating(df_p_dash)
        st.pyplot(fig_tier_dash); plt.close(fig_tier_dash)

    # Section 1
    st.markdown("---")
    st.header("📂 Section 1 - Setup & Data Loading")

    st.success(
        f"Loaded **{CSV_PATH}** — "
        f"**{raw.shape[0]:,} rows × {raw.shape[1]} columns**"
    )

    st.code(str(raw.columns.tolist()))
    st.dataframe(raw.head(), use_container_width=True)

    # Section 2
    st.markdown("---")
    st.header("🧹 Section 2 - Data Cleaning")
    null_df = raw.isnull().sum().rename("Null Count").to_frame()
    null_df["% Null"] = (null_df["Null Count"] / len(raw) * 100).round(1)
    st.dataframe(null_df, use_container_width=True)

    # Section 3
    st.markdown("---")
    st.header("🏷️ Section 3 - Category Extraction")
    fig_cat = plot_category_dist(raw)
    st.pyplot(fig_cat); plt.close(fig_cat)
    cat_counts_df = raw["category"].value_counts().rename_axis("Category").reset_index(name="Reviews")
    st.dataframe(cat_counts_df, use_container_width=True)

    # Section 4
    st.markdown("---")
    st.header("📊 Section 4 - Product-Level Aggregation")
    st.success(f"Unique products: **{len(prod_agg):,}**")
    fig_agg = plot_aggregated_sample(prod_agg)
    st.pyplot(fig_agg); plt.close(fig_agg)

    # Section 5
    st.markdown("---")
    st.header("🏆 Section 5 - Top vs Underperforming Products")
    top10, bottom10 = get_top_bottom(prod_agg, min_reviews=30)

    st.subheader("🥇 Top 10 Products")
    st.dataframe(top10[["product_name", "category", "avg_rating",
                         "review_count", "avg_price", "pos_pct", "neg_pct"]],
                 use_container_width=True)
    fig_top = plot_top_bottom(top10, "Top 10 Products", "#2ecc71")
    st.pyplot(fig_top); plt.close(fig_top)

    st.subheader("⚠️ Bottom 10 Products")
    st.dataframe(bottom10[["product_name", "category", "avg_rating",
                            "review_count", "avg_price", "pos_pct", "neg_pct"]],
                 use_container_width=True)
    fig_bot = plot_top_bottom(bottom10, "Bottom 10 Products", "#e74c3c")
    st.pyplot(fig_bot); plt.close(fig_bot)

    # Section 6
    st.markdown("---")
    st.header("💰 Section 6 - Pricing Pattern Analysis")
    df_p = pricing_analysis(raw)

    st.subheader("Price vs Rating (scatter)")
    fig_sc = plot_scatter(df_p); st.pyplot(fig_sc); plt.close(fig_sc)

    st.subheader("Price Distribution by Category")
    fig_bx = plot_boxplot(df_p); st.pyplot(fig_bx); plt.close(fig_bx)

    st.subheader("Avg Rating per Price Tier")
    fig_tr = plot_tier_rating(df_p); st.pyplot(fig_tr); plt.close(fig_tr)

    st.subheader("Sentiment Breakdown per Price Tier")
    fig_ts = plot_tier_sentiment(df_p); st.pyplot(fig_ts); plt.close(fig_ts)

    # Section 7
    st.markdown("---")
    st.header("🔬 Section 7 - What Drives High Ratings")

    st.subheader("Correlation Heatmap")
    fig_hm = plot_heatmap(prod_agg); st.pyplot(fig_hm); plt.close(fig_hm)

    model, features, mae, r2 = train_model(prod_agg)
    feat_labels = ["Avg Price", "Review Count", "Positive Sent %", "Category"]
    st.subheader("Feature Importance - Gradient Boosting Regressor")
    st.markdown(f"**MAE:** {mae:.3f}  &nbsp;&nbsp;  **R2:** {r2:.3f}")
    fig_fi, importances = plot_feature_importance(model, feat_labels)
    st.pyplot(fig_fi); plt.close(fig_fi)

    # Section 8
    st.markdown("---")
    st.header("🔍 Section 8 - Root-Cause Analysis of Underperformance")
    top_df, underperf_cats = run_tfidf(raw, prod_agg)
    st.markdown(f"**Underperforming categories:** {', '.join(underperf_cats)}")
    if not top_df.empty:
        fig_tf = plot_tfidf_bar(top_df); st.pyplot(fig_tf); plt.close(fig_tf)
        if HAS_WORDCLOUD:
            fig_wc = plot_wordcloud(top_df)
            if fig_wc:
                st.pyplot(fig_wc); plt.close(fig_wc)

    # Section 9
    st.markdown("---")
    st.header("📝 Section 9 - Auto-Generated Recommendations")
    for rec in generate_recommendations(prod_agg, df_p, importances, feat_labels):
        st.markdown(f"- {rec}")

    # Section 10
    render_interactive(prod_agg)

    st.markdown("---")
    st.caption("Flipkart Product Performance Analysis · AICTE IBM SkillsBuild Internship · Built with Streamlit")


# Entry point
if _SAVE_MODE:
    save_all_charts()
else:
    # Streamlit always runs this file with __name__ == "__main__",
    # so main_streamlit() must be called unconditionally here (not
    # gated behind an "else: imported" branch, which never fires).
    main_streamlit()
