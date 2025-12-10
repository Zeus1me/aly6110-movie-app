# app.py
# ================================================
# Amazon Movie Review Dashboard – Streamlit App
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------- CONFIG -----------------

st.set_page_config(
    page_title="Amazon Movie Review Dashboard",
    page_icon="🎬",
    layout="wide",
)

# Use the folder that contains this file (works on Streamlit Cloud & locally)
PROJECT_ROOT = Path(__file__).parent
# For deployment we keep the data file(s) in the same folder as app.py
DATA_DIR = PROJECT_ROOT

sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "axes.titleweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    }
)

COLORS_MIXED = [
    "#E63946",
    "#457B9D",
    "#2A9D8F",
    "#F4A261",
    "#A8DADC",
    "#1D3557",
]

# ---------------- DATA LOADING -----------------


def load_parquet_folder(folder: Path) -> pd.DataFrame:
    """Load all parquet/CSV files from a folder into a single DataFrame."""
    parquet_files = list(folder.glob("*.parquet"))
    csv_files = list(folder.glob("*.csv"))

    if not parquet_files and not csv_files:
        raise FileNotFoundError(f"No parquet or CSV files found in {folder}")

    dfs = []

    for f in parquet_files:
        dfs.append(pd.read_parquet(f))

    for f in csv_files:
        dfs.append(pd.read_csv(f))

    df = pd.concat(dfs, ignore_index=True)
    return df


@st.cache_data(show_spinner=True)
def get_data() -> pd.DataFrame:
    df = load_parquet_folder(DATA_DIR)

    # Standardise column names (lowercase)
    df.columns = [c.strip() for c in df.columns]

    # Ensure key columns exist
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    if "helpful_ratio" in df.columns:
        df["helpful_ratio"] = pd.to_numeric(df["helpful_ratio"], errors="coerce")

    if "total_votes" in df.columns:
        df["total_votes"] = pd.to_numeric(df["total_votes"], errors="coerce")

    # Build date / year information from unixTime where possible
    if "unixTime" in df.columns:
        df["unixTime"] = pd.to_numeric(df["unixTime"], errors="coerce")
        df["review_date"] = pd.to_datetime(df["unixTime"], unit="s", errors="coerce")
    elif "unixtime" in df.columns:
        df["unixtime"] = pd.to_numeric(df["unixtime"], errors="coerce")
        df["review_date"] = pd.to_datetime(df["unixtime"], unit="s", errors="coerce")
    else:
        df["review_date"] = pd.NaT

    # If review_year is missing, derive it from review_date
    if "review_year" not in df.columns:
        df["review_year"] = df["review_date"].dt.year
    else:
        df["review_year"] = pd.to_numeric(df["review_year"], errors="coerce")

    return df


# ---------------- SIDEBAR FILTERS -----------------


def sidebar_filters(df_full: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    # Sample size control (for speed)
    max_rows = len(df_full)
    default_sample = min(50000, max_rows)

    sample_n = st.sidebar.number_input(
        "Sample size (0 = full dataset)",
        min_value=0,
        max_value=max_rows,
        value=default_sample,
        step=1000,
    )

    if sample_n and 0 < sample_n < max_rows:
        df = df_full.sample(sample_n, random_state=42)
    else:
        df = df_full.copy()

    # Year range slider (robust to edge cases)
    year_range = None
    if "review_year" in df.columns:
        years = (
            df["review_year"]
            .dropna()
            .astype(int)
        )
        if len(years) > 0:
            min_year = int(years.min())
            max_year = int(years.max())

            if min_year < max_year:
                year_range = st.sidebar.slider(
                    "Review year range",
                    min_value=min_year,
                    max_value=max_year,
                    value=(min_year, max_year),
                    step=1,
                )
            else:
                st.sidebar.write(f"Only one review year in data: {min_year}")
                year_range = (min_year, max_year)

    if year_range is not None:
        df = df[
            (df["review_year"] >= year_range[0])
            & (df["review_year"] <= year_range[1])
        ]

    # Rating filter
    if "rating" in df.columns:
        unique_ratings = sorted(df["rating"].dropna().unique())
        rating_options = st.sidebar.multiselect(
            "Ratings to include",
            options=unique_ratings,
            default=unique_ratings,
        )
        if rating_options:
            df = df[df["rating"].isin(rating_options)]

    # Sentiment filter (if available)
    if "sentiment" in df.columns:
        sentiments = sorted(df["sentiment"].dropna().unique())
        sentiment_default = sentiments
        sentiment_sel = st.sidebar.multiselect(
            "Sentiment",
            options=sentiments,
            default=sentiment_default,
        )
        if sentiment_sel:
            df = df[df["sentiment"].isin(sentiment_sel)]

    # Minimum total votes (to focus on more engaged reviews)
    if "total_votes" in df.columns:
        min_votes = int(df["total_votes"].fillna(0).min())
        max_votes = int(df["total_votes"].fillna(0).max())
        vote_thresh = st.sidebar.slider(
            "Minimum total votes",
            min_value=min_votes,
            max_value=max_votes,
            value=min_votes,
            step=1,
        )
        df = df[df["total_votes"].fillna(0) >= vote_thresh]

    return df


# ---------------- VISUAL HELPERS -----------------


def plot_reviews_over_time(df: pd.DataFrame):
    if "review_year" not in df.columns or df["review_year"].dropna().empty:
        st.info("No year information available for time-based view.")
        return

    yearly = (
        df.dropna(subset=["review_year"])
        .groupby("review_year")
        .agg(
            review_count=("rating", "count"),
            avg_rating=("rating", "mean"),
        )
        .reset_index()
        .sort_values("review_year")
    )

    fig, ax1 = plt.subplots()
    ax1.bar(
        yearly["review_year"],
        yearly["review_count"],
        alpha=0.7,
        label="Review count",
        color=COLORS_MIXED[1],
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of reviews")

    ax2 = ax1.twinx()
    ax2.plot(
        yearly["review_year"],
        yearly["avg_rating"],
        marker="o",
        linewidth=2,
        label="Average rating",
        color=COLORS_MIXED[0],
    )
    ax2.set_ylabel("Average rating (1–5)")

    ax1.set_title("Review Volume and Average Rating Over Time")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    st.pyplot(fig)

    st.markdown(
        """
**Interpretation**

- The bars show how many reviews were written each year.  
- The line shows how the average rating changed over time.  
Use this to spot years with unusual spikes in activity or changes in audience sentiment.
"""
    )


def plot_rating_distribution(df: pd.DataFrame):
    if "rating" not in df.columns:
        st.info("No rating column available.")
        return

    rating_counts = (
        df["rating"]
        .dropna()
        .value_counts()
        .sort_index()
        .reset_index()
    )
    rating_counts.columns = ["rating", "count"]
    total = rating_counts["count"].sum()
    rating_counts["share"] = rating_counts["count"] / total * 100

    fig, ax = plt.subplots()
    ax.bar(
        rating_counts["rating"].astype(str),
        rating_counts["count"],
        color=COLORS_MIXED,
    )
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Distribution of Review Ratings")

    st.pyplot(fig)

    summary_lines = [
        f"- **{row['rating']:.0f}★**: {row['count']} reviews ({row['share']:.1f}%)"
        for _, row in rating_counts.iterrows()
    ]
    st.markdown(
        "**Key facts about rating distribution:**\n\n" + "\n".join(summary_lines)
    )


def plot_helpfulness_by_rating(df: pd.DataFrame):
    if "rating" not in df.columns or "helpful_ratio" not in df.columns:
        st.info("Helpfulness data not available.")
        return

    df_valid = df.dropna(subset=["rating", "helpful_ratio"])
    if df_valid.empty:
        st.info("No non-missing helpfulness values for this filter selection.")
        return

    fig, ax = plt.subplots()
    sns.boxplot(
        data=df_valid,
        x="rating",
        y="helpful_ratio",
        ax=ax,
        palette=COLORS_MIXED,
    )
    ax.set_xlabel("Rating")
    ax.set_ylabel("Helpful ratio (helpful_yes / total_votes)")
    ax.set_title("Helpfulness Ratio by Rating")

    st.pyplot(fig)

    st.markdown(
        """
**How to read this**

- Each box shows the spread of helpfulness scores for reviews with a given rating.  
- Taller boxes or many outliers mean more variability in how other users vote on those reviews.  
"""
    )


def plot_top_entities(df: pd.DataFrame):
    cols = st.columns(2)

    # Top products by review volume
    if "productId" in df.columns:
        product_counts = (
            df["productId"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        product_counts.columns = ["productId", "review_count"]

        with cols[0]:
            fig, ax = plt.subplots()
            ax.barh(
                product_counts["productId"],
                product_counts["review_count"],
                color=COLORS_MIXED[2],
            )
            ax.invert_yaxis()
            ax.set_xlabel("Number of reviews")
            ax.set_title("Top 10 Most Reviewed Products")
            st.pyplot(fig)

    # Top reviewers by volume
    if "userId" in df.columns:
        user_counts = (
            df["userId"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        user_counts.columns = ["userId", "review_count"]

        with cols[1]:
            fig, ax = plt.subplots()
            ax.barh(
                user_counts["userId"],
                user_counts["review_count"],
                color=COLORS_MIXED[3],
            )
            ax.invert_yaxis()
            ax.set_xlabel("Number of reviews")
            ax.set_title("Top 10 Most Active Reviewers")
            st.pyplot(fig)


def plot_helpful_votes_vs_total(df: pd.DataFrame):
    if "helpful_yes" not in df.columns or "total_votes" not in df.columns:
        st.info("Helpful vote information not available.")
        return

    df_valid = df.dropna(subset=["helpful_yes", "total_votes"])
    df_valid = df_valid[df_valid["total_votes"] > 0]

    if df_valid.empty:
        st.info("No valid helpful/total vote pairs in current selection.")
        return

    fig, ax = plt.subplots()
    ax.scatter(
        df_valid["total_votes"],
        df_valid["helpful_yes"],
        alpha=0.4,
        color=COLORS_MIXED[4],
    )
    ax.set_xlabel("Total votes")
    ax.set_ylabel("Helpful votes")
    ax.set_title("Helpful Votes vs Total Votes")

    st.pyplot(fig)

    st.markdown(
        """
**Insight**

Each point is a review.  
Reviews far from the diagonal line “helpful ≈ total” are polarizing — many users mark them as unhelpful even though they get a lot of attention.
"""
    )


# ---------------- MAIN APP -----------------


def main():
    st.title("Amazon Movie Review Dashboard")
    st.markdown(
        """
Interactive exploration of ratings, review activity, helpfulness, reviewers, and products
from the Amazon Movies & TV reviews dataset.  
Use the filters on the left to focus on specific time windows, rating bands, or review engagement levels.
"""
    )

    df_full = get_data()
    df = sidebar_filters(df_full)

    # --- High-level KPIs ---
    total_reviews = len(df)
    avg_rating = df["rating"].mean() if "rating" in df.columns else np.nan
    unique_products = df["productId"].nunique() if "productId" in_
