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

# In Streamlit Cloud, use the folder that contains app.py
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT  # sample_reviews.parquet lives here

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

    # Standardise column names
    df.columns = [c.strip() for c in df.columns]

    # Ensure numeric columns
    for col in ["rating", "helpful_ratio", "total_votes", "helpful_yes", "unixTime", "unixtime"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build date from unixTime where possible
    if "unixTime" in df.columns:
        df["review_date"] = pd.to_datetime(df["unixTime"], unit="s", errors="coerce")
    elif "unixtime" in df.columns:
        df["review_date"] = pd.to_datetime(df["unixtime"], unit="s", errors="coerce")
    else:
        df["review_date"] = pd.NaT

    # Derive or clean review_year
    if "review_year" not in df.columns:
        df["review_year"] = df["review_date"].dt.year
    else:
        df["review_year"] = pd.to_numeric(df["review_year"], errors="coerce")

    return df


# ---------------- SIDEBAR FILTERS -----------------


def sidebar_filters(df_full: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Global filters")

    # --- Sample size slider (performance control) ---
    max_rows = len(df_full)
    default_sample = min(50000, max_rows)

    sample_n = st.sidebar.slider(
        "Sample size (0 = full dataset)",
        min_value=0,
        max_value=max_rows,
        value=default_sample,
        step=max(1, max_rows // 20),
    )

    if sample_n and 0 < sample_n < max_rows:
        df = df_full.sample(sample_n, random_state=42)
    else:
        df = df_full.copy()

    # --- Year range slider ---
    year_range = None
    if "review_year" in df.columns:
        years = df["review_year"].dropna().astype(int)
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
                st.sidebar.markdown(f"Only one review year in data: **{min_year}**")
                year_range = (min_year, max_year)

    if year_range is not None:
        df = df[
            (df["review_year"] >= year_range[0])
            & (df["review_year"] <= year_range[1])
        ]

    # --- Date range slider (within the selected years) ---
    if "review_date" in df.columns and df["review_date"].notna().any():
        dates = df["review_date"].dropna()
        if not dates.empty:
            min_date = dates.min().date()
            max_date = dates.max().date()
            if min_date < max_date:
                date_range = st.sidebar.slider(
                    "Review date range",
                    min_value=min_date,
                    max_value=max_date,
                    value=(min_date, max_date),
                )
                start_dt = pd.to_datetime(date_range[0])
                end_dt = pd.to_datetime(date_range[1])
                df = df[
                    (df["review_date"] >= start_dt)
                    & (df["review_date"] <= end_dt)
                ]

    st.sidebar.markdown("---")

    # --- Rating filter ---
    if "rating" in df.columns:
        st.sidebar.subheader("Rating & sentiment")
        unique_ratings = sorted(df["rating"].dropna().unique())
        rating_options = st.sidebar.multiselect(
            "Ratings to include",
            options=unique_ratings,
            default=unique_ratings,
        )
        if rating_options:
            df = df[df["rating"].isin(rating_options)]

    # --- Sentiment filter ---
    if "sentiment" in df.columns:
        sentiments = sorted(df["sentiment"].dropna().unique())
        sentiment_sel = st.sidebar.multiselect(
            "Sentiment",
            options=sentiments,
            default=sentiments,
        )
        if sentiment_sel:
            df = df[df["sentiment"].isin(sentiment_sel)]

    st.sidebar.markdown("---")

    # --- Engagement filters ---
    if "total_votes" in df.columns:
        st.sidebar.subheader("Engagement")
        min_votes = int(df["total_votes"].fillna(0).min())
        max_votes = int(df["total_votes"].fillna(0).max())
        vote_thresh = st.sidebar.slider(
            "Minimum total votes",
            min_value=min_votes,
            max_value=max_votes,
            value=min_votes,
            step=max(1, (max_votes - min_votes) // 20 or 1),
        )
        df = df[df["total_votes"].fillna(0) >= vote_thresh]

    if "helpful_ratio" in df.columns:
        min_hr = float(df["helpful_ratio"].fillna(0).min())
        max_hr = float(df["helpful_ratio"].fillna(0).max())
        hr_range = st.sidebar.slider(
            "Helpful ratio range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, min(1.0, max_hr)),
            step=0.05,
        )
        df = df[df["helpful_ratio"].fillna(0).between(hr_range[0], hr_range[1])]

    # --- Toggle for advanced charts ---
    advanced = st.sidebar.checkbox("Show advanced reviewer/product analytics", value=True)

    return df, advanced


# ---------------- VISUAL HELPERS -----------------


def plot_reviews_over_time(df: pd.DataFrame, time_granularity: str = "Year"):
    if "review_date" not in df.columns or df["review_date"].dropna().empty:
        st.info("No date information available for time-based view.")
        return

    df_time = df.dropna(subset=["review_date"]).copy()

    if time_granularity == "Year":
        df_time["time_bucket"] = df_time["review_date"].dt.year
    elif time_granularity == "Quarter":
        df_time["time_bucket"] = df_time["review_date"].dt.to_period("Q").astype(str)
    else:  # Month
        df_time["time_bucket"] = df_time["review_date"].dt.to_period("M").astype(str)

    grouped = (
        df_time.groupby("time_bucket")
        .agg(
            review_count=("rating", "count"),
            avg_rating=("rating", "mean"),
        )
        .reset_index()
        .sort_values("time_bucket")
    )

    fig, ax1 = plt.subplots()
    ax1.bar(
        grouped["time_bucket"],
        grouped["review_count"],
        alpha=0.7,
        label="Review count",
        color=COLORS_MIXED[1],
    )
    ax1.set_xlabel(time_granularity)
    ax1.set_ylabel("Number of reviews")

    ax2 = ax1.twinx()
    ax2.plot(
        grouped["time_bucket"],
        grouped["avg_rating"],
        marker="o",
        linewidth=2,
        label="Average rating",
        color=COLORS_MIXED[0],
    )
    ax2.set_ylabel("Average rating (1–5)")

    ax1.set_title(f"Review Volume and Average Rating Over Time ({time_granularity})")
    ax1.tick_params(axis="x", rotation=45)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    st.pyplot(fig)

    st.markdown(
        """
**How to use this chart**

- Bars show how many reviews were written in each period.  
- The line shows how the average rating changed over the same periods.  
Look for:
- spikes in review volume (marketing campaigns, new releases), and  
- drops or peaks in the rating line (changes in audience sentiment).
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
        """
**What this tells you**

- Shows how reviews are spread across the 1–5★ scale.  
- Skewed to the right (more 4–5★) → generally satisfied customers.  
- Skewed to the left → more negative experience.

**Breakdown:**

""" + "\n".join(summary_lines)
    )


def plot_helpfulness_by_rating(df: pd.DataFrame):
    if "rating" not in df.columns or "helpful_ratio" not in df.columns:
        st.info("Helpfulness data not available.")
        return

    df_valid = df.dropna(subset=["rating", "helpful_ratio"])
    if df_valid.empty:
        st.info("No non-missing helpfulness values for this selection.")
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

- Each box summarises the helpfulness of reviews with a given rating.  
- Higher median helpfulness → other users tend to trust those reviews.  
- Lots of outliers → some reviews are very controversial for that rating band.
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

    st.markdown(
        """
**Why this matters**

- The left chart highlights “hero” products that dominate conversation.  
- The right chart surfaces power reviewers who could be targeted for engagement or loyalty programs.
"""
    )


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
Each point is a single review.

- Points close to the diagonal (helpful ≈ total) → most people agree the review is useful.  
- Points far below the diagonal → many users actively mark the review as **not** helpful, even if it gets a lot of attention.
"""
    )


# ---------------- MAIN APP -----------------


def main():
    st.title("Amazon Movie Review Dashboard")
    st.markdown(
        """
This dashboard lets you explore the **Amazon Movies & TV reviews** dataset from multiple angles:

- How review **volume and ratings** change over time  
- How ratings are **distributed** across the 1–5★ scale  
- How **helpful voters** interact with reviews  
- Which **products** and **reviewers** dominate the conversation  

Use the filters on the left to zoom in on specific time windows, rating bands, sentiment buckets,
and engagement levels.
"""
    )

    df_full = get_data()
    df, advanced = sidebar_filters(df_full)

    # --- High-level KPIs ---
    total_reviews = len(df)

    if "rating" in df.columns:
        avg_rating = float(df["rating"].mean())
    else:
        avg_rating = float("nan")

    if "productId" in df.columns:
        unique_products = int(df["productId"].nunique())
    else:
        unique_products = 0

    if "userId" in df.columns:
        unique_users = int(df["userId"].nunique())
    else:
        unique_users = 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews in view", f"{total_reviews:,}")
    c2.metric(
        "Average rating",
        f"{avg_rating:.2f}" if not np.isnan(avg_rating) else "N/A",
    )
    c3.metric("Unique products", f"{unique_products:,}")
    c4.metric("Unique reviewers", f"{unique_users:,}")

    with st.expander("What does this filter selection represent?", expanded=False):
        st.markdown(
            """
These KPIs respect all filters currently selected in the sidebar:

- **Reviews in view** – number of reviews after sampling + filters  
- **Average rating** – mean star rating for those reviews  
- **Unique products/reviewers** – breadth of catalogue and reviewer base within the filtered slice
"""
        )

    st.markdown("---")

    # --- Time-based insight ---
    st.subheader("1. Review Volume and Ratings Over Time")

    time_granularity = st.radio(
        "Time aggregation",
        options=["Year", "Quarter", "Month"],
        horizontal=True,
    )
    plot_reviews_over_time(df, time_granularity=time_granularity)

    st.markdown("---")

    # --- Rating distribution ---
    st.subheader("2. Rating Distribution")
    plot_rating_distribution(df)

    st.markdown("---")

    # --- Helpfulness analysis ---
    st.subheader("3. Helpfulness Patterns")
    cols_help = st.columns(2)

    with cols_help[0]:
        st.markdown("##### Helpfulness ratio by rating")
        plot_helpfulness_by_rating(df)

    with cols_help[1]:
        st.markdown("##### Helpful votes vs total votes")
        plot_helpful_votes_vs_total(df)

    st.markdown("---")

    # --- Products & reviewers (advanced toggle) ---
    if advanced:
        st.subheader("4. Top Products and Reviewers")
        plot_top_entities(df)
        st.markdown("---")

    # --- Optional raw data preview ---
    with st.expander("Peek at the underlying data (sample)", expanded=False):
        st.dataframe(df.head(200))

    st.caption(
        "Data source: Amazon Movies & TV reviews (sample). "
        "Dashboard prepared for ALY6110 deployment on Streamlit Cloud."
    )


if __name__ == "__main__":
    main()
