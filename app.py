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

PROJECT_ROOT = Path(__file__).parent

# sample parquet file (sample_reviews.parquet) 
DATA_DIR = PROJECT_ROOT

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "axes.titleweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

COLORS_MIXED = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#9B5DE5',
                '#F4A261', '#00BBF9', '#F72585', '#80ED99', '#264653']


# ------------- DATA LOADER ----------------

def load_parquet_folder(folder: Path) -> pd.DataFrame:
    parquet_files = list(folder.glob("*.parquet"))
    if parquet_files:
        dfs = []
        for pf in parquet_files:
            dfs.append(pd.read_parquet(pf))
        return pd.concat(dfs, ignore_index=True)

    csv_files = list(folder.glob("*.csv"))
    if csv_files:
        dfs = []
        for cf in csv_files:
            dfs.append(pd.read_csv(cf))
        return pd.concat(dfs, ignore_index=True)

    raise FileNotFoundError(f"No parquet or CSV files found in {folder}")


@st.cache_data(show_spinner=True)
def get_data() -> pd.DataFrame:
    df = load_parquet_folder(DATA_DIR)

    # basic typing cleanup
    if "rating" in df.columns:
        df["rating"] = df["rating"].astype(float)
    if "review_year" in df.columns:
        df["review_year"] = pd.to_numeric(df["review_year"], errors="coerce")
    if "helpful_ratio" in df.columns:
        df["helpful_ratio"] = pd.to_numeric(df["helpful_ratio"], errors="coerce")

    return df


# ------------- TIME-BASED HELPERS ----------------

def plot_review_volume_over_time(df: pd.DataFrame):
    if "review_year" not in df.columns:
        return None

    counts = (
        df.dropna(subset=["review_year"])
          .groupby("review_year")["productId"]
          .count()
          .reset_index(name="review_count")
          .sort_values("review_year")
    )
    if counts.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(counts["review_year"], counts["review_count"],
                  color=COLORS_MIXED[1], edgecolor="white", alpha=0.85)

    for bar in bars:
        height = bar.get_height()
        if height > counts["review_count"].max() * 0.15:
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f"{int(height / 1000)}K", ha="center", va="bottom", fontsize=9)

    total = counts["review_count"].sum()
    ax.set_title("Review Volume Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Reviews")
    ax.text(0.02, 0.95,
            f"Total reviews (filtered): {total:,}",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    return fig


def plot_avg_rating_over_time(df: pd.DataFrame):
    """Line chart of average rating by year."""
    if not {"review_year", "rating"}.issubset(df.columns):
        return None

    yearly = (
        df.dropna(subset=["review_year", "rating"])
          .groupby("review_year")["rating"]
          .mean()
          .reset_index(name="avg_rating")
          .sort_values("review_year")
    )
    if yearly.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        yearly["review_year"],
        yearly["avg_rating"],
        marker="o",
        linewidth=2.5,
        color=COLORS_MIXED[0],
        markerfacecolor="white",
        markeredgewidth=2
    )
    ax.set_title("Average Rating Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Rating (1–5)")
    ax.set_ylim(3.5, 5.1)

    # Annotate last point
    last_row = yearly.iloc[-1]
    ax.text(last_row["review_year"], last_row["avg_rating"],
            f"{last_row['avg_rating']:.2f}",
            ha="left", va="bottom", fontsize=9, fontweight="bold")
    return fig


def get_time_insights(df: pd.DataFrame) -> str:
    """Generate short narrative time-based insights."""
    if "review_year" not in df.columns:
        return "Time-based insights are not available because 'review_year' is missing."

    sub = df.dropna(subset=["review_year"]).copy()
    if sub.empty:
        return "No non-missing years in the filtered data, so time-based insights are not available."

    years = sub["review_year"]
    min_year = int(years.min())
    max_year = int(years.max())

    # Review volume trend
    vol = (
        sub.groupby("review_year")["productId"]
        .count()
        .reset_index(name="review_count")
        .sort_values("review_year")
    )
    total_reviews = vol["review_count"].sum()
    first_year = int(vol.iloc[0]["review_year"])
    first_count = int(vol.iloc[0]["review_count"])
    last_year = int(vol.iloc[-1]["review_year"])
    last_count = int(vol.iloc[-1]["review_count"])

    if first_count > 0:
        growth_factor = last_count / first_count
    else:
        growth_factor = np.nan

    # Top 3 busiest years
    top3 = vol.sort_values("review_count", ascending=False).head(3)

    # Average rating trend
    rating_trend_text = "Average rating is not available for this filtered selection."
    if "rating" in sub.columns and sub["rating"].notna().any():
        yearly_rating = (
            sub.dropna(subset=["rating"])
               .groupby("review_year")["rating"]
               .mean()
               .reset_index(name="avg_rating")
               .sort_values("review_year")
        )
        if not yearly_rating.empty:
            first_r = yearly_rating.iloc[0]["avg_rating"]
            last_r = yearly_rating.iloc[-1]["avg_rating"]
            diff = last_r - first_r
            if abs(diff) < 0.05:
                rating_trend_text = (
                    f"Average rating has been **fairly stable**, "
                    f"from about {first_r:.2f} to {last_r:.2f} over time."
                )
            elif diff > 0:
                rating_trend_text = (
                    f"Average rating shows a **mild upward trend**, "
                    f"rising from about {first_r:.2f} to {last_r:.2f}."
                )
            else:
                rating_trend_text = (
                    f"Average rating shows a **slight decline**, "
                    f"from about {first_r:.2f} down to {last_r:.2f}."
                )

    growth_text = ""
    if not np.isnan(growth_factor):
        if growth_factor > 1.2:
            growth_text = (
                f"Review volume has **grown** from {first_count:,} reviews in {first_year} "
                f"to {last_count:,} in {last_year} (about {growth_factor:.1f}×)."
            )
        elif growth_factor < 0.8:
            growth_text = (
                f"Review volume has **declined** from {first_count:,} reviews in {first_year} "
                f"to {last_count:,} in {last_year}."
            )
        else:
            growth_text = (
                f"Review volume in {first_year} vs {last_year} is **fairly similar** "
                f"({first_count:,} vs {last_count:,} reviews)."
            )

    top_years_text = ""
    if not top3.empty:
        parts = [
            f"{int(row['review_year'])} ({int(row['review_count']):,} reviews)"
            for _, row in top3.iterrows()
        ]
        top_years_text = (
            "The busiest years in the current selection are: "
            + ", ".join(parts) + "."
        )

    insight = (
        f"- Time span covered (after filters): **{min_year}–{max_year}**.\n"
        f"- Total reviews in this time window: **{total_reviews:,}**.\n"
    )
    if growth_text:
        insight += f"- {growth_text}\n"
    if top_years_text:
        insight += f"- {top_years_text}\n"
    insight += f"- {rating_trend_text}"

    return insight


# ------------- OTHER PLOT HELPERS ----------------

def plot_rating_distribution(df: pd.DataFrame):
    if "rating" not in df.columns:
        return None

    ratings = df["rating"].dropna()
    if ratings.empty:
        return None

    counts = ratings.value_counts().sort_index()
    total = counts.sum()
    percentages = (counts / total * 100).round(1)

    labels = counts.index.astype(int).tolist()
    values = counts.values
    bar_colors = [COLORS_MIXED[i] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=bar_colors,
                   edgecolor="white", linewidth=2, height=0.6)

    for bar, val, pct in zip(bars, values, percentages.values):
        ax.text(val + total * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,} ({pct:.1f}%)",
                va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r}★" for r in labels])
    ax.invert_yaxis()
    ax.set_xlabel("Number of Reviews")
    ax.set_title("Rating Distribution")

    avg_rating = ratings.mean()
    median_rating = ratings.median()
    positive_pct = percentages[percentages.index >= 4].sum()
    negative_pct = percentages[percentages.index <= 2].sum()
    neutral_pct = percentages.get(3, 0)

    key_facts = (
        f"Avg rating: {avg_rating:.2f}   |   "
        f"Median: {median_rating:.0f}★   |   "
        f"Positive (4–5★): {positive_pct:.1f}%   |   "
        f"Neutral (3★): {neutral_pct:.1f}%   |   "
        f"Negative (1–2★): {negative_pct:.1f}%"
    )
    ax.text(0.5, -0.18, key_facts, transform=ax.transAxes,
            ha="center", va="top", fontsize=10)

    fig.tight_layout()
    return fig


def plot_helpful_ratio_by_rating(df: pd.DataFrame, kind: str = "Violin"):
    if not {"rating", "helpful_ratio"}.issubset(df.columns):
        return None

    subset = df.dropna(subset=["rating", "helpful_ratio"]).copy()
    subset["helpful_ratio"] = subset["helpful_ratio"].clip(0, 1)
    if subset.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    if kind == "Violin":
        sns.violinplot(
            data=subset,
            x="rating",
            y="helpful_ratio",
            palette=COLORS_MIXED[:5],
            inner="quartile",
            cut=0,
            ax=ax
        )
    else:  # Box
        sns.boxplot(
            data=subset,
            x="rating",
            y="helpful_ratio",
            palette=COLORS_MIXED[:5],
            ax=ax
        )

    ax.set_title(f"Helpful Ratio by Rating ({kind} plot)")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Helpful Ratio (0–1)")
    return fig


def plot_helpful_vs_votes(df: pd.DataFrame):
    if not {"helpful_yes", "total_votes"}.issubset(df.columns):
        return None

    subset = df.dropna(subset=["helpful_yes", "total_votes"]).copy()
    subset = subset[(subset["total_votes"] > 0) & (subset["helpful_yes"] > 0)]
    if subset.empty:
        return None

    x = subset["total_votes"].values
    y = subset["helpful_yes"].values

    fig, ax = plt.subplots(figsize=(10, 6))
    hb = ax.hexbin(
        x, y,
        gridsize=50,
        cmap="YlOrRd",
        mincnt=1,
        xscale="log",
        yscale="log",
        linewidths=0.25
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Number of Reviews")

    max_val = max(x.max(), y.max())
    ax.plot([1, max_val], [1, max_val],
            "--", color=COLORS_MIXED[1], linewidth=2, alpha=0.8,
            label="100% Helpful Line")

    ax.set_title("Helpful Votes vs Total Votes (log–log Hexbin)")
    ax.set_xlabel("Total Votes (log)")
    ax.set_ylabel("Helpful Votes (log)")
    ax.legend(loc="lower right")
    return fig


def plot_top_reviewers(df: pd.DataFrame, top_n: int = 10):
    if "userId" not in df.columns:
        return None

    user_counts = (
        df.groupby("userId")["productId"]
          .count()
          .reset_index(name="review_count")
          .sort_values("review_count", ascending=False)
    )
    if user_counts.empty:
        return None

    top_users = user_counts.head(top_n)
    total_reviews = user_counts["review_count"].sum()
    top_total = top_users["review_count"].sum()
    top_share = top_total / total_reviews * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [COLORS_MIXED[i] for i in range(len(top_users))]
    bars = ax.barh(range(len(top_users)), top_users["review_count"], color=colors,
                   edgecolor="white", linewidth=1.5)

    labels = [f"#{i + 1}" for i in range(len(top_users))]
    ax.set_yticks(range(len(top_users)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Reviews")
    ax.set_title(f"Top {top_n} Most Active Reviewers")

    max_val = top_users["review_count"].max()
    for i, (bar, val) in enumerate(zip(bars, top_users["review_count"].values)):
        ax.text(val + max_val * 0.01, i,
                f"{val:,} reviews",
                va="center", fontsize=10, fontweight="bold")

    ax.text(0.98, 0.02,
            f"Top {top_n} share: {top_share:.2f}% of all reviews",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    fig.tight_layout()
    return fig


def plot_reviewer_heatmap(df: pd.DataFrame):
    if not {"userId", "rating"}.issubset(df.columns):
        return None

    stats = df.groupby("userId").agg(
        review_count=("productId", "count"),
        avg_rating=("rating", "mean")
    ).reset_index()

    if stats.empty:
        return None

    activity_bins = [0, 5, 20, 50, 100, float("inf")]
    activity_labels = ["1–5", "6–20", "21–50", "51–100", "100+"]
    rating_bins = [0, 2, 3, 4, 5.1]
    rating_labels = ["1–2", "2–3", "3–4", "4–5"]

    stats["activity_bin"] = pd.cut(stats["review_count"], bins=activity_bins,
                                   labels=activity_labels, right=False)
    stats["rating_bin"] = pd.cut(stats["avg_rating"], bins=rating_bins,
                                 labels=rating_labels, right=False)

    heat_data = stats.groupby(["activity_bin", "rating_bin"], observed=False).size().unstack(fill_value=0)
    heat_data = heat_data.reindex(index=activity_labels, columns=rating_labels, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        heat_data,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Number of Users"},
        ax=ax
    )
    ax.set_title("Reviewer Activity vs Average Rating Given")
    ax.set_xlabel("Average Rating")
    ax.set_ylabel("Number of Reviews (per user)")
    return fig


def plot_product_distribution(df: pd.DataFrame):
    if "productId" not in df.columns:
        return None

    product_counts = (
        df.groupby("productId")["userId"]
          .count()
          .reset_index(name="review_count")
          .sort_values("review_count", ascending=False)
    )
    if product_counts.empty:
        return None

    counts = product_counts["review_count"].values
    total_products = len(product_counts)
    total_reviews = counts.sum()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(counts, bins=50, color=COLORS_MIXED[3], ax=ax)
    ax.set_xscale("log")
    ax.set_title("Distribution of Reviews per Product (Log Scale)")
    ax.set_xlabel("Reviews per Product (log scale)")
    ax.set_ylabel("Number of Products")

    key_facts = (
        f"Products: {total_products:,}   |   "
        f"Total Reviews: {total_reviews:,}   |   "
        f"Median Reviews/Product: {np.median(counts):.0f}"
    )
    ax.text(0.5, -0.18, key_facts, transform=ax.transAxes,
            ha="center", va="top", fontsize=10)
    fig.tight_layout()
    return fig


def plot_top_products(df: pd.DataFrame, top_n: int = 10):
    if "productId" not in df.columns:
        return None

    product_counts = (
        df.groupby("productId")["userId"]
          .count()
          .reset_index(name="review_count")
          .sort_values("review_count", ascending=False)
    )
    if product_counts.empty:
        return None

    top_products = product_counts.head(top_n)
    total_reviews = product_counts["review_count"].sum()
    top_total = top_products["review_count"].sum()
    top_share = top_total / total_reviews * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(top_products))
    vals = top_products["review_count"].values
    max_val = vals[0]
    colors = [COLORS_MIXED[i] for i in range(len(top_products))]

    bars = ax.barh(y_pos, vals, color=colors, edgecolor="white", linewidth=2, height=0.7)
    labels = [f"Product #{i + 1}" for i in range(len(top_products))]

    for i, (bar, val) in enumerate(zip(bars, vals)):
        pct_of_max = val / max_val * 100
        ax.text(val + max_val * 0.01, i,
                f"{val:,} ({pct_of_max:.0f}% of #1)",
                va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title("Top Most Reviewed Products")
    ax.set_xlabel("Number of Reviews")

    ax.text(0.98, 0.02,
            f"Top {top_n} share: {top_share:.2f}% of all reviews",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    fig.tight_layout()
    return fig


# ------------- STREAMLIT APP ----------------

def main():
    st.set_page_config(
        page_title="Amazon Movie Review Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🎬 Amazon Movie Review Dashboard")
    st.caption(
        "Interactive exploration of ratings, review activity, helpfulness, reviewers, and products "
        "from the Amazon Movies & TV reviews dataset. Use the filters on the left to focus on "
        "specific time windows, rating bands, or review engagement levels."
    )

    with st.spinner("Loading data..."):
        df_full = get_data()

    # ---------- SIDEBAR FILTERS ----------
    st.sidebar.header("Filters")

    # Optional sampling (for performance)
    st.sidebar.subheader("Data size")
    max_rows = len(df_full)
    sample_n = st.sidebar.number_input(
        "Sample size (0 = full dataset)",
        min_value=0,
        max_value=max_rows,
        value=0,
        step=50000
    )
    if sample_n and sample_n > 0 and sample_n < max_rows:
        df = df_full.sample(sample_n, random_state=42)
    else:
        df = df_full.copy()

    # Year filter
    if "review_year" in df.columns and df["review_year"].notna().any():
        min_year = int(df["review_year"].min())
        max_year = int(df["review_year"].max())
        year_range = st.sidebar.slider(
            "Review year range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            step=1
        )
        df = df[(df["review_year"] >= year_range[0]) &
                (df["review_year"] <= year_range[1])]

    # Rating filter
    if "rating" in df.columns:
        unique_ratings = sorted(df["rating"].dropna().unique().tolist())
        st.sidebar.subheader("Ratings")
        selected_ratings = st.sidebar.multiselect(
            "Ratings to include",
            options=unique_ratings,
            default=unique_ratings
        )
        if selected_ratings:
            df = df[df["rating"].isin(selected_ratings)]

    # Min total votes filter
    if "total_votes" in df.columns:
        st.sidebar.subheader("Helpfulness")
        max_votes = int(df["total_votes"].max())
        min_votes = st.sidebar.number_input(
            "Minimum total votes (for helpfulness views)",
            min_value=0,
            max_value=max_votes,
            value=0,
            step=1
        )
        if min_votes > 0:
            df = df[df["total_votes"] >= min_votes]

    st.sidebar.markdown("---")
    st.sidebar.write(f"**Filtered rows:** {len(df):,}")

    # ---------- KPI CARDS ----------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Reviews (filtered)", f"{len(df):,}")
    with c2:
        if "rating" in df.columns and not df["rating"].dropna().empty:
            st.metric("Average Rating", f"{df['rating'].mean():.2f}")
        else:
            st.metric("Average Rating", "N/A")
    with c3:
        if "helpful_yes" in df.columns:
            st.metric("Total Helpful Votes", f"{int(df['helpful_yes'].sum()):,}")
        else:
            st.metric("Total Helpful Votes", "N/A")
    with c4:
        if "userId" in df.columns:
            st.metric("Distinct Reviewers", f"{df['userId'].nunique():,}")
        else:
            st.metric("Distinct Reviewers", "N/A")

    st.markdown("---")

    # ---------- TABS ----------
    tab_overview, tab_ratings, tab_helpful, tab_reviewers, tab_products = st.tabs(
        ["📊 Overview", "⭐ Ratings", "👍 Helpfulness", "👤 Reviewers", "📦 Products"]
    )

    # Overview
    with tab_overview:
        st.subheader("Time-based Overview")
        st.markdown(
            "This section focuses on how review activity and ratings have evolved over time "
            "within the current filter selection."
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Review Volume Over Time**")
            fig = plot_review_volume_over_time(df)
            if fig is not None:
                st.pyplot(fig)
            else:
                st.info("No 'review_year' data available for this view.")
        with col2:
            st.markdown("**Average Rating Over Time**")
            fig2 = plot_avg_rating_over_time(df)
            if fig2 is not None:
                st.pyplot(fig2)
            else:
                st.info("Need both 'review_year' and 'rating' for this chart.")

        st.markdown("**Time-based Insights**")
        st.markdown(get_time_insights(df))

        with st.expander("How to read these time-based charts"):
            st.markdown(
                "- The bar chart highlights **how many reviews** were written each year.\n"
                "- The line chart tracks **average rating** per year, showing sentiment stability or drift.\n"
                "- Use the **year slider** and **rating filter** in the sidebar to see how patterns change for "
                "specific periods or rating bands."
            )

    # Ratings
    with tab_ratings:
        st.subheader("Rating Structure and Balance")
        st.markdown(
            "This tab summarises how ratings are distributed in the filtered dataset. "
            "It helps answer: *Is the dataset dominated by 5★ praise, or is there a balanced mix of opinions?*"
        )

        fig = plot_rating_distribution(df)
        if fig is not None:
            st.pyplot(fig)
            with st.expander("Summary of rating behaviour"):
                ratings = df["rating"].dropna()
                counts = ratings.value_counts().sort_index()
                total = counts.sum()
                if total > 0:
                    pos = counts[counts.index >= 4].sum() / total * 100
                    neg = counts[counts.index <= 2].sum() / total * 100
                    st.markdown(
                        f"- There are **{total:,}** reviews in the current selection.\n"
                        f"- About **{pos:.1f}%** of reviews are **positive (4–5★)**.\n"
                        f"- About **{neg:.1f}%** are **negative (1–2★)**.\n"
                        f"- Use the rating filter in the sidebar to see how patterns change when you focus on "
                        f"only high or only low scores."
                    )
        else:
            st.info("No 'rating' data available for this view.")

    # Helpfulness
    with tab_helpful:
        st.subheader("Helpfulness Patterns")
        st.markdown(
            "This tab focuses on **engagement**, not just scores. It shows how often reviews "
            "are voted helpful and how those helpful votes relate to total votes."
        )

        chart_type = st.radio(
            "Choose chart type for helpful ratio by rating",
            options=["Violin", "Box"],
            horizontal=True
        )

        fig1 = plot_helpful_ratio_by_rating(df, kind=chart_type)
        if fig1 is not None:
            st.pyplot(fig1)
            with st.expander("How to interpret helpful ratio by rating"):
                st.markdown(
                    "- Each shape shows the **distribution of helpful ratios** (helpful_yes / total_votes) "
                    "for a rating.\n"
                    "- Wider sections indicate where most reviews sit; longer upper tails show a few very "
                    "highly regarded reviews.\n"
                    "- Use the **minimum total votes** filter in the sidebar to remove noisy reviews with only "
                    "1 or 2 votes."
                )
        else:
            st.info("Need 'rating' and 'helpful_ratio' columns for this chart.")

        st.markdown("---")
        st.subheader("Helpful Votes vs Total Votes")
        fig2 = plot_helpful_vs_votes(df)
        if fig2 is not None:
            st.pyplot(fig2)
            with st.expander("What this density plot tells you"):
                st.markdown(
                    "- Each hexagon bundles reviews with similar **total votes** and **helpful votes**.\n"
                    "- Darker hexagons mean many reviews in that region.\n"
                    "- The dashed line is the **100% helpful** frontier: points near this line were found helpful "
                    "by almost everyone who voted."
                )
        else:
            st.info("Need 'helpful_yes' and 'total_votes' columns for this chart.")

    # Reviewers
    with tab_reviewers:
        st.subheader("Reviewer Profiles")
        st.markdown(
            "This tab looks at **who** is writing reviews: a handful of power-users or a broad base of casual "
            "reviewers."
        )

        top_n_reviewers = st.slider(
            "Number of top reviewers to display",
            min_value=5,
            max_value=30,
            value=10,
            step=5
        )
        fig = plot_top_reviewers(df, top_n=top_n_reviewers)
        if fig is not None:
            st.pyplot(fig)
            with st.expander("Context for top reviewers"):
                st.markdown(
                    "- Bars represent individual reviewer accounts, ranked by number of reviews.\n"
                    "- The annotation at the bottom shows how much of the dataset is driven by this top group.\n"
                    "- A high share from a small group suggests **concentrated activity**."
                )
        else:
            st.info("Need 'userId' and 'productId' for this chart.")

        st.markdown("---")
        st.subheader("Reviewer Activity vs Average Rating")
        fig = plot_reviewer_heatmap(df)
        if fig is not None:
            st.pyplot(fig)
            with st.expander("What this heatmap shows"):
                st.markdown(
                    "- Rows group users by **how many reviews** they have written.\n"
                    "- Columns group users by **average rating** they give.\n"
                    "- Darker cells show where most reviewers sit: for example, many casual users who mostly give "
                    "4–5★ scores vs. a smaller cluster of critical high-activity reviewers."
                )
        else:
            st.info("Need 'userId', 'productId' and 'rating' for this chart.")

    # Products
    with tab_products:
        st.subheader("Product Coverage")
        st.markdown(
            "This tab explores how reviews are distributed across products: whether a few popular titles dominate, "
            "or if attention is more evenly spread."
        )

        fig = plot_product_distribution(df)
        if fig is not None:
            st.pyplot(fig)
            with st.expander("Interpreting the product distribution"):
                st.markdown(
                    "- The x-axis (log scale) shows **how many reviews** each product has.\n"
                    "- The y-axis shows **how many products** fall into each bucket.\n"
                    "- A long right tail means a small set of products attract a very large number of reviews."
                )
        else:
            st.info("Need 'productId' for this chart.")

        st.markdown("---")
        st.subheader("Top Reviewed Products")

        top_n_products = st.slider(
            "Number of top products to display",
            min_value=5,
            max_value=30,
            value=10,
            step=5
        )
        fig = plot_top_products(df, top_n=top_n_products)
        if fig is not None:
            st.pyplot(fig)
            with st.expander("Context for top products"):
                st.markdown(
                    "- Each bar represents a single movie/TV title (anonymised here as Product #1, #2, etc.).\n"
                    "- The percentage next to each bar shows how its review count compares to the **most reviewed** product.\n"
                    "- The annotation at the bottom summarises how dominant the top product set is."
                )
        else:
            st.info("Need 'productId' and 'userId' for this chart.")


if __name__ == "__main__":
    main()
