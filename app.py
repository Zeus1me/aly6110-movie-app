# app.py
# ================================================
# Amazon Movie Review Dashboard – Premium Interactive Edition
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from io import BytesIO
from datetime import datetime

# ---------------- CONFIG -----------------

st.set_page_config(
    page_title="Amazon Movie Review Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional dark corporate styling
st.markdown("""
<style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Global styling */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main background */
    .stApp {
        background-color: #0a0a0a;
    }
    
    /* Gradient header - Corporate dark theme */
    .main-header {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 50%, #1a1a1a 100%);
        border: 2px solid #ffa500;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(255, 165, 0, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #ffa500, #ff8c00, #ffa500);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .main-header h1 {
        color: #ffa500;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 0 20px rgba(255, 165, 0, 0.3);
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #d0d0d0;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Metric cards - Orange/Yellow theme */
    .metric-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        border: 1px solid #ffa500;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #ffa500, #ff8c00);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(255, 165, 0, 0.3);
        border-color: #ff8c00;
    }
    
    .metric-card h3 {
        color: #888;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-card h1 {
        color: #ffa500;
        font-weight: 800;
        font-size: 2.2rem;
        margin: 0.5rem 0;
    }
    
    .metric-card p {
        color: #999;
        font-size: 0.85rem;
    }
    
    /* Section headers */
    .section-header {
        color: #ffa500 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        margin: 2rem 0 1rem 0 !important;
        padding-left: 1rem !important;
        border-left: 4px solid #ffa500 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(255, 165, 0, 0.05);
        border-left: 4px solid #ffa500;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #d0d0d0;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #ffa500 0%, #ff8c00 100%) !important;
        color: #000 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(255, 165, 0, 0.3) !important;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #ff8c00 0%, #ffa500 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 165, 0, 0.5) !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: #1a1a1a;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        color: #888 !important;
        border: 1px solid transparent !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 165, 0, 0.1) !important;
        color: #ffa500 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ffa500 0%, #ff8c00 100%) !important;
        color: #000 !important;
        border-color: #ffa500 !important;
        font-weight: 700 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 165, 0, 0.1) !important;
        border: 1px solid #ffa500 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        color: #ffa500 !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 165, 0, 0.2) !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a1a1a 100%) !important;
        border-right: 2px solid #ffa500 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #d0d0d0 !important;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #ffa500 !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #888 !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        font-size: 0.8rem !important;
        letter-spacing: 1px !important;
    }
    
    /* Success/Info messages */
    .stSuccess {
        background-color: rgba(255, 165, 0, 0.1) !important;
        border-left: 4px solid #ffa500 !important;
        color: #ffa500 !important;
    }
    
    .stInfo {
        background-color: rgba(255, 165, 0, 0.05) !important;
        border-left: 4px solid #ff8c00 !important;
        color: #d0d0d0 !important;
    }
    
    /* Download button */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #ffa500 0%, #ff8c00 100%) !important;
        color: #000 !important;
        border: none !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# In Streamlit Cloud, use the folder that contains app.py
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT  # sample_reviews.parquet lives here

# Modern color palettes - Professional Corporate Orange Theme
COLORS_VIBRANT = ['#ffa500', '#ff8c00', '#ff7b00', '#ff6b00', '#ff5500', '#ff4500']
COLORS_SUNSET = ['#ffb84d', '#ffa500', '#ff8c00', '#ff7b00', '#ff6b00', '#ff5500']
COLORS_OCEAN = ['#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#a93226']

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
def load_sample_raw() -> pd.DataFrame:
    """Load the bundled sample dataset from the repo folder."""
    return load_parquet_folder(DATA_DIR)


@st.cache_data(show_spinner=True)
def load_uploaded_raw(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Load an uploaded CSV or Parquet file."""
    buffer = BytesIO(file_bytes)
    name_lower = file_name.lower()

    if name_lower.endswith(".parquet"):
        df = pd.read_parquet(buffer)
    elif name_lower.endswith(".csv"):
        df = pd.read_csv(buffer)
    else:
        raise ValueError("Unsupported file type. Please upload .csv or .parquet.")
    return df


def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Standard cleaning so visuals work for both sample + uploaded datasets."""
    df.columns = [c.strip() for c in df.columns]

    for col in ["rating", "helpful_ratio", "total_votes", "helpful_yes", "unixTime", "unixtime"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "unixTime" in df.columns:
        df["review_date"] = pd.to_datetime(df["unixTime"], unit="s", errors="coerce")
    elif "unixtime" in df.columns:
        df["review_date"] = pd.to_datetime(df["unixtime"], unit="s", errors="coerce")
    elif "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    else:
        df["review_date"] = pd.NaT

    if "review_year" not in df.columns:
        df["review_year"] = df["review_date"].dt.year
    else:
        df["review_year"] = pd.to_numeric(df["review_year"], errors="coerce")

    return df


# ---------------- SIDEBAR FILTERS -----------------


def sidebar_filters(df_full: pd.DataFrame):
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); border: 2px solid #ffa500; border-radius: 10px; margin-bottom: 2rem;'>
        <h2 style='color: #ffa500; margin: 0; font-weight: 800; text-transform: uppercase; letter-spacing: 2px;'>⚙️ CONTROLS</h2>
    </div>
    """, unsafe_allow_html=True)

    # --- Sample size slider ---
    max_rows = len(df_full)
    default_sample = min(50000, max_rows)

    st.sidebar.markdown("### 📊 Data Sample")
    sample_n = st.sidebar.slider(
        "Sample size (0 = full dataset)",
        min_value=0,
        max_value=max_rows,
        value=default_sample,
        step=max(1, max_rows // 20) if max_rows > 0 else 1,
        help="Adjust for performance. Larger samples = more accurate but slower."
    )

    if sample_n and 0 < sample_n < max_rows:
        df = df_full.sample(sample_n, random_state=42)
    else:
        df = df_full.copy()

    st.sidebar.markdown("---")

    # --- Year range ---
    year_range = None
    if "review_year" in df.columns:
        st.sidebar.markdown("### 📅 Time Period")
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
                st.sidebar.info(f"Only one year: **{min_year}**")
                year_range = (min_year, max_year)

    if year_range is not None:
        df = df[
            (df["review_year"] >= year_range[0])
            & (df["review_year"] <= year_range[1])
        ]

    # --- Date range ---
    if "review_date" in df.columns and df["review_date"].notna().any():
        dates = df["review_date"].dropna()
        if not dates.empty:
            min_date = dates.min().date()
            max_date = dates.max().date()
            if min_date < max_date:
                date_range = st.sidebar.date_input(
                    "Precise date range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                )
                if len(date_range) == 2:
                    start_dt = pd.to_datetime(date_range[0])
                    end_dt = pd.to_datetime(date_range[1])
                    df = df[
                        (df["review_date"] >= start_dt)
                        & (df["review_date"] <= end_dt)
                    ]

    st.sidebar.markdown("---")

    # --- Rating filter ---
    if "rating" in df.columns:
        st.sidebar.markdown("### ⭐ Ratings")
        unique_ratings = sorted(df["rating"].dropna().unique())
        rating_options = st.sidebar.multiselect(
            "Include ratings",
            options=unique_ratings,
            default=unique_ratings,
            help="Filter reviews by star rating"
        )
        if rating_options:
            df = df[df["rating"].isin(rating_options)]

    st.sidebar.markdown("---")

    # --- Engagement filters ---
    st.sidebar.markdown("### 🔥 Engagement")
    
    if "total_votes" in df.columns:
        min_votes = int(df["total_votes"].fillna(0).min())
        max_votes = int(df["total_votes"].fillna(0).max())
        vote_thresh = st.sidebar.slider(
            "Minimum total votes",
            min_value=min_votes,
            max_value=max_votes,
            value=min_votes,
            step=max(1, (max_votes - min_votes) // 20 or 1),
            help="Filter highly engaged reviews"
        )
        df = df[df["total_votes"].fillna(0) >= vote_thresh]

    if "helpful_ratio" in df.columns:
        hr_range = st.sidebar.slider(
            "Helpful ratio range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.05,
            help="Filter by review helpfulness"
        )
        df = df[df["helpful_ratio"].fillna(0).between(hr_range[0], hr_range[1])]

    st.sidebar.markdown("---")

    # --- Advanced analytics toggle ---
    st.sidebar.markdown("### 🔬 Analytics Depth")
    advanced = st.sidebar.checkbox("Advanced analytics", value=True, help="Show detailed product and reviewer analysis")

    return df, advanced


# ---------------- INTERACTIVE VISUALIZATIONS -----------------


def plot_reviews_over_time_interactive(df: pd.DataFrame, time_granularity: str = "Year"):
    """Interactive time series with dual-axis - properly rendered."""
    if "review_date" not in df.columns or df["review_date"].dropna().empty:
        st.info("📅 No date information available for time-based view.")
        return

    df_time = df.dropna(subset=["review_date"]).copy()

    # Create proper time buckets
    if time_granularity == "Year":
        df_time["time_bucket"] = df_time["review_date"].dt.year
        grouped = (
            df_time.groupby("time_bucket")
            .agg(
                review_count=("rating", "count"),
                avg_rating=("rating", "mean"),
            )
            .reset_index()
            .sort_values("time_bucket")
        )
        grouped["time_display"] = grouped["time_bucket"].astype(int).astype(str)
        
    elif time_granularity == "Quarter":
        df_time["time_bucket"] = df_time["review_date"].dt.to_period("Q")
        grouped = (
            df_time.groupby("time_bucket")
            .agg(
                review_count=("rating", "count"),
                avg_rating=("rating", "mean"),
            )
            .reset_index()
        )
        grouped = grouped.sort_values("time_bucket")
        grouped["time_display"] = grouped["time_bucket"].astype(str)
        
    else:  # Month
        df_time["time_bucket"] = df_time["review_date"].dt.to_period("M")
        grouped = (
            df_time.groupby("time_bucket")
            .agg(
                review_count=("rating", "count"),
                avg_rating=("rating", "mean"),
            )
            .reset_index()
        )
        grouped = grouped.sort_values("time_bucket")
        grouped["time_display"] = grouped["time_bucket"].astype(str)

    if grouped.empty:
        st.info("No data available for the selected time range.")
        return

    # Calculate appropriate bar width based on number of bars
    num_bars = len(grouped)
    if num_bars == 1:
        bar_width = 0.4
    elif num_bars <= 5:
        bar_width = 0.6
    elif num_bars <= 10:
        bar_width = 0.7
    else:
        bar_width = 0.8

    # Create figure with secondary y-axis
    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
        row_heights=[1.0]
    )

    # Add bar trace for review count
    fig.add_trace(
        go.Bar(
            x=grouped["time_display"],
            y=grouped["review_count"],
            name="Review Count",
            marker=dict(
                color=grouped["review_count"],
                colorscale=[
                    [0, '#ff8c00'],
                    [0.5, '#ffa500'],
                    [1, '#ffb84d']
                ],
                line=dict(color='rgba(255,165,0,0.3)', width=1),
                showscale=False
            ),
            text=[f"{val:,}" for val in grouped["review_count"]],
            textposition='outside',
            textfont=dict(size=10, color='white'),
            hovertemplate='<b>%{x}</b><br>Reviews: %{y:,}<extra></extra>',
            opacity=0.9,
            width=bar_width
        ),
        secondary_y=False
    )

    # Add line trace for average rating
    fig.add_trace(
        go.Scatter(
            x=grouped["time_display"],
            y=grouped["avg_rating"],
            name="Avg Rating",
            mode='lines+markers',
            line=dict(color='#00bcd4', width=4),
            marker=dict(
                size=10,
                color='#00bcd4',
                symbol='circle',
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>%{x}</b><br>Rating: %{y:.2f}★<extra></extra>'
        ),
        secondary_y=True
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"📈 Review Volume & Ratings Trend ({time_granularity})",
            font=dict(size=22, color='#ffa500', family='Arial Black'),
            x=0.05,
            y=0.98
        ),
        hovermode='x unified',
        plot_bgcolor='rgba(15, 23, 42, 0.3)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=550,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
            bgcolor='rgba(30, 60, 114, 0.6)',
            bordercolor='rgba(255,255,255,0.2)',
            borderwidth=1,
            font=dict(color='white', size=12)
        ),
        margin=dict(l=80, r=80, t=120, b=100),
        bargap=0.15
    )

    # X-axis
    fig.update_xaxes(
        title_text=time_granularity,
        title_font=dict(size=14, color='white'),
        showgrid=True,
        gridcolor='rgba(128,128,128,0.15)',
        tickangle=-45 if num_bars > 10 else 0,
        tickfont=dict(size=11, color='white'),
        showline=True,
        linewidth=1,
        linecolor='rgba(255,255,255,0.2)',
        type='category'  # Force categorical to ensure all values show
    )

    # Primary y-axis (Review Count)
    fig.update_yaxes(
        title_text="Number of Reviews",
        title_font=dict(color='#4a9eff', size=14),
        secondary_y=False,
        showgrid=True,
        gridcolor='rgba(128,128,128,0.15)',
        tickfont=dict(color='#4a9eff', size=11),
        showline=True,
        linewidth=1,
        linecolor='rgba(255,255,255,0.2)'
    )

    # Secondary y-axis (Average Rating)
    fig.update_yaxes(
        title_text="Average Rating (1-5★)",
        title_font=dict(color='#00bcd4', size=14),
        secondary_y=True,
        showgrid=False,
        range=[0, 5.5],
        tickfont=dict(color='#00bcd4', size=11),
        showline=True,
        linewidth=1,
        linecolor='rgba(255,255,255,0.2)'
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})

    # Key insights
    with st.expander("💡 What This Chart Tells You", expanded=False):
        total_reviews = int(grouped["review_count"].sum())
        peak_period = grouped.loc[grouped["review_count"].idxmax(), "time_display"]
        peak_count = int(grouped["review_count"].max())
        avg_rating_overall = float(grouped["avg_rating"].mean())
        min_rating = float(grouped["avg_rating"].min())
        max_rating = float(grouped["avg_rating"].max())
        
        st.markdown(f"""
        **Key Insights:**
        
        - 📊 **Total Reviews:** {total_reviews:,} reviews in the selected period
        - 🚀 **Peak Activity:** {peak_period} with {peak_count:,} reviews
        - ⭐ **Average Rating:** {avg_rating_overall:.2f} stars across all periods
        - 📉 **Rating Range:** {min_rating:.2f} to {max_rating:.2f} stars
        - 📈 **Trend Analysis:** Look for spikes (marketing campaigns, releases) and rating shifts (sentiment changes)
        """)


def plot_rating_distribution_interactive(df: pd.DataFrame):
    """Beautiful interactive rating distribution with percentages."""
    if "rating" not in df.columns:
        st.info("⭐ No rating column available.")
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
    rating_counts["percentage"] = (rating_counts["count"] / total * 100).round(1)

    if rating_counts.empty:
        st.info("No ratings available for the current selection.")
        return

    # Create colorful bar chart with cold colors
    fig = go.Figure()

    colors = ['#0d47a1', '#1565c0', '#1976d2', '#1e88e5', '#2196f3']
    
    fig.add_trace(go.Bar(
        x=rating_counts["rating"].astype(str) + "★",
        y=rating_counts["count"],
        text=rating_counts.apply(lambda x: f"{x['count']:,}<br>({x['percentage']:.1f}%)", axis=1),
        textposition='outside',
        textfont=dict(size=12, color='white'),
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.4)', width=2)
        ),
        hovertemplate='<b>%{x}</b><br>Reviews: %{y:,}<br>Share: %{text}<extra></extra>',
        showlegend=False,
        opacity=0.9
    ))

    fig.update_layout(
        title=dict(
            text="⭐ Rating Distribution Breakdown",
            font=dict(size=22, color='#2a5298', family='Arial Black'),
            x=0.05
        ),
        xaxis_title="Rating",
        yaxis_title="Number of Reviews",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=500,
        hovermode='x',
        margin=dict(l=60, r=60, t=100, b=80)
    )

    fig.update_xaxes(showgrid=False, tickfont=dict(size=13))
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')

    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics in columns
    col1, col2, col3 = st.columns(3)
    
    positive = rating_counts[rating_counts["rating"] >= 4]["count"].sum()
    neutral = rating_counts[rating_counts["rating"] == 3]["count"].sum()
    negative = rating_counts[rating_counts["rating"] <= 2]["count"].sum()
    
    with col1:
        st.metric("😊 Positive (4-5★)", f"{positive:,}", f"{positive/total*100:.1f}%")
    with col2:
        st.metric("😐 Neutral (3★)", f"{neutral:,}", f"{neutral/total*100:.1f}%")
    with col3:
        st.metric("😞 Negative (1-2★)", f"{negative:,}", f"{negative/total*100:.1f}%")


def plot_helpfulness_analysis_interactive(df: pd.DataFrame):
    """Interactive helpfulness analysis with multiple views."""
    if "rating" not in df.columns or "helpful_ratio" not in df.columns:
        st.info("💡 Helpfulness data not available.")
        return

    df_valid = df.dropna(subset=["rating", "helpful_ratio"])
    if df_valid.empty:
        st.info("No non-missing helpfulness values.")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Violin plot for helpfulness by rating
        fig = go.Figure()

        colors_cold = ['#0d47a1', '#1565c0', '#1976d2', '#1e88e5', '#2196f3', '#42a5f5']
        for i, rating in enumerate(sorted(df_valid["rating"].unique())):
            rating_data = df_valid[df_valid["rating"] == rating]["helpful_ratio"]
            fig.add_trace(go.Violin(
                y=rating_data,
                name=f"{int(rating)}★",
                box_visible=True,
                meanline_visible=True,
                fillcolor=colors_cold[i % len(colors_cold)],
                opacity=0.8,
                line=dict(color=colors_cold[i % len(colors_cold)]),
                hovertemplate='Rating: %{fullData.name}<br>Helpfulness: %{y:.3f}<extra></extra>'
            ))

        fig.update_layout(
            title=dict(
                text="🎻 Helpfulness Distribution by Rating",
                font=dict(size=18, color='#2a5298', family='Arial Black'),
                x=0.05
            ),
            yaxis_title="Helpful Ratio",
            showlegend=True,
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor='center'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=450,
            margin=dict(l=60, r=40, t=80, b=80)
        )

        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', range=[0, 1])

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Average by rating
        avg_helpfulness = df_valid.groupby("rating")["helpful_ratio"].mean().reset_index()
        
        fig = go.Figure(go.Bar(
            x=avg_helpfulness["rating"].astype(str) + "★",
            y=avg_helpfulness["helpful_ratio"],
            marker=dict(
                color=avg_helpfulness["helpful_ratio"],
                colorscale=[[0, '#0d47a1'], [0.5, '#1976d2'], [1, '#42a5f5']],
                showscale=True,
                colorbar=dict(title="Avg<br>Ratio", len=0.5),
                line=dict(color='rgba(255,255,255,0.3)', width=2)
            ),
            text=avg_helpfulness["helpful_ratio"].round(3),
            textposition='outside',
            textfont=dict(size=12, color='white'),
            hovertemplate='<b>%{x}</b><br>Avg Helpfulness: %{y:.3f}<extra></extra>',
            opacity=0.9
        ))

        fig.update_layout(
            title=dict(
                text="📊 Average Helpfulness by Rating",
                font=dict(size=18, color='#2a5298', family='Arial Black'),
                x=0.05
            ),
            xaxis_title="Rating",
            yaxis_title="Average Helpful Ratio",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=450,
            margin=dict(l=60, r=40, t=80, b=80)
        )

        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', range=[0, 1])

        st.plotly_chart(fig, use_container_width=True)


def plot_scatter_helpful_votes_interactive(df: pd.DataFrame):
    """Beautiful scatter plot with trend line."""
    if "helpful_yes" not in df.columns or "total_votes" not in df.columns:
        st.info("📊 Vote information not available.")
        return

    df_valid = df.dropna(subset=["helpful_yes", "total_votes"])
    df_valid = df_valid[df_valid["total_votes"] > 0].copy()

    if df_valid.empty:
        st.info("No valid helpful/total vote pairs.")
        return

    # Sample for performance if too large
    if len(df_valid) > 5000:
        df_valid = df_valid.sample(5000, random_state=42)

    # Add rating color if available
    if "rating" in df_valid.columns:
        df_valid["rating_str"] = df_valid["rating"].astype(str) + "★"
        
        # Cold color scale for ratings
        color_map = {
            "1★": "#0d47a1",
            "2★": "#1565c0", 
            "3★": "#1976d2",
            "4★": "#1e88e5",
            "5★": "#2196f3"
        }
        
        fig = px.scatter(
            df_valid,
            x="total_votes",
            y="helpful_yes",
            color="rating_str",
            color_discrete_map=color_map,
            opacity=0.6,
            title="🎯 Helpful Votes vs Total Votes (Color = Rating)",
            labels={"total_votes": "Total Votes", "helpful_yes": "Helpful Votes", "rating_str": "Rating"},
            hover_data={"rating_str": True, "total_votes": ":,", "helpful_yes": ":,"}
        )
    else:
        fig = px.scatter(
            df_valid,
            x="total_votes",
            y="helpful_yes",
            opacity=0.5,
            title="🎯 Helpful Votes vs Total Votes",
            labels={"total_votes": "Total Votes", "helpful_yes": "Helpful Votes"},
            color_discrete_sequence=['#1976d2']
        )

    # Add diagonal reference line
    max_val = max(df_valid["total_votes"].max(), df_valid["helpful_yes"].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(color='#00bcd4', dash='dash', width=3),
        name='100% Helpful Line',
        hovertemplate='Perfect Agreement Line<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text="🎯 Helpful Votes vs Total Votes (Color = Rating)",
            font=dict(size=20, color='#2a5298', family='Arial Black'),
            x=0.05
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=550,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            bgcolor='rgba(0,0,0,0.5)',
            font=dict(color='white')
        ),
        margin=dict(l=60, r=60, t=100, b=80)
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("💡 How to Read This Chart"):
        st.markdown("""
        - **Points near the cyan line:** Most users found the review helpful
        - **Points below the line:** Mixed reception - many voted "not helpful"
        - **Color indicates rating:** See if certain ratings get more agreement
        - **Clusters:** Common patterns in user engagement
        """)


def plot_top_entities_interactive(df: pd.DataFrame):
    """Interactive top products and reviewers."""
    col1, col2 = st.columns(2)

    # Top products
    if "productId" in df.columns:
        product_counts = df["productId"].value_counts().head(15).reset_index()
        product_counts.columns = ["productId", "review_count"]

        if not product_counts.empty:
            with col1:
                fig = go.Figure(go.Bar(
                    x=product_counts["review_count"],
                    y=product_counts["productId"],
                    orientation='h',
                    marker=dict(
                        color=product_counts["review_count"],
                        colorscale=[[0, '#0d47a1'], [0.5, '#1976d2'], [1, '#42a5f5']],
                        showscale=False,
                        line=dict(color='rgba(255,255,255,0.3)', width=1)
                    ),
                    text=product_counts["review_count"],
                    textposition='outside',
                    textfont=dict(size=11, color='white'),
                    hovertemplate='<b>%{y}</b><br>Reviews: %{x:,}<extra></extra>'
                ))

                fig.update_layout(
                    title=dict(
                        text="🏆 Top 15 Most Reviewed Products",
                        font=dict(size=18, color='#2a5298', family='Arial Black'),
                        x=0.05
                    ),
                    xaxis_title="Number of Reviews",
                    yaxis_title="Product ID",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=550,
                    yaxis={'categoryorder': 'total ascending'},
                    margin=dict(l=120, r=60, t=80, b=60)
                )

                fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
                fig.update_yaxes(tickfont=dict(size=10))

                st.plotly_chart(fig, use_container_width=True)

    # Top reviewers
    if "userId" in df.columns:
        user_counts = df["userId"].value_counts().head(15).reset_index()
        user_counts.columns = ["userId", "review_count"]

        if not user_counts.empty:
            with col2:
                fig = go.Figure(go.Bar(
                    x=user_counts["review_count"],
                    y=user_counts["userId"],
                    orientation='h',
                    marker=dict(
                        color=user_counts["review_count"],
                        colorscale=[[0, '#006064'], [0.5, '#00838f'], [1, '#26c6da']],
                        showscale=False,
                        line=dict(color='rgba(255,255,255,0.3)', width=1)
                    ),
                    text=user_counts["review_count"],
                    textposition='outside',
                    textfont=dict(size=11, color='white'),
                    hovertemplate='<b>%{y}</b><br>Reviews: %{x:,}<extra></extra>'
                ))

                fig.update_layout(
                    title=dict(
                        text="👥 Top 15 Most Active Reviewers",
                        font=dict(size=18, color='#2a5298', family='Arial Black'),
                        x=0.05
                    ),
                    xaxis_title="Number of Reviews",
                    yaxis_title="User ID",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=550,
                    yaxis={'categoryorder': 'total ascending'},
                    margin=dict(l=120, r=60, t=80, b=60)
                )

                fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
                fig.update_yaxes(tickfont=dict(size=10))

                st.plotly_chart(fig, use_container_width=True)


def plot_engagement_heatmap(df: pd.DataFrame):
    """Heatmap of engagement patterns - properly rendered."""
    
    # Check if required columns exist
    if "review_year" not in df.columns:
        st.warning("⚠️ No 'review_year' column found in data")
        return
    
    if "rating" not in df.columns:
        st.warning("⚠️ No 'rating' column found in data")
        return
        
    if "helpful_ratio" not in df.columns:
        st.warning("⚠️ No 'helpful_ratio' column found in data")
        return

    # Filter valid data
    df_valid = df.dropna(subset=["review_year", "rating", "helpful_ratio"]).copy()
    
    if len(df_valid) < 5:
        st.info(f"💡 Not enough data for engagement heatmap. Found {len(df_valid)} rows, need at least 5 reviews with year, rating, and helpfulness data.")
        return
    
    # Convert to proper types
    try:
        df_valid["review_year"] = df_valid["review_year"].astype(int)
        df_valid["rating"] = df_valid["rating"].astype(int)
    except Exception as e:
        st.error(f"Error converting data types: {str(e)}")
        return
    
    # Check for variety
    unique_years = df_valid["review_year"].nunique()
    unique_ratings = df_valid["rating"].nunique()
    
    if unique_years < 1 or unique_ratings < 1:
        st.info(f"💡 Need at least 1 year and 1 rating. Found {unique_years} years and {unique_ratings} ratings.")
        return

    # Create pivot table
    try:
        pivot = df_valid.pivot_table(
            values="helpful_ratio",
            index="rating",
            columns="review_year",
            aggfunc="mean"
        )
        
        # Sort by rating (descending) and year (ascending)
        pivot = pivot.sort_index(ascending=False)
        pivot = pivot[sorted(pivot.columns)]
        
    except Exception as e:
        st.error(f"Error creating pivot table: {str(e)}")
        return

    if pivot.empty:
        st.info("💡 Pivot table is empty - no data to display")
        return

    # Create the heatmap
    try:
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[str(int(col)) for col in pivot.columns],
            y=[f"{int(r)}★" for r in pivot.index],
            colorscale=[
                [0.0, '#1a0b2e'],    # Deep purple (very low)
                [0.2, '#3d1e6d'],    # Dark purple
                [0.4, '#7e2e84'],    # Medium purple
                [0.6, '#e8345f'],    # Hot pink
                [0.8, '#ff6b35'],    # Vibrant orange
                [1.0, '#f7b731']     # Bright yellow-orange (very high)
            ],
            hoverongaps=False,
            hovertemplate='<b>Year: %{x}</b><br>Rating: %{y}<br>Avg Helpfulness: %{z:.3f}<extra></extra>',
            colorbar=dict(
                title=dict(
                    text="Avg Helpful Ratio",
                    font=dict(size=12, color='white')
                ),
                tickmode="linear",
                tick0=0,
                dtick=0.1,
                len=0.6,
                thickness=15,
                tickfont=dict(size=11, color='white')
            ),
            zmin=0,
            zmax=1
        ))

        fig.update_layout(
            title=dict(
                text="🔥 Engagement Heatmap: Helpfulness by Year & Rating",
                font=dict(size=20, color='#2a5298', family='Arial Black'),
                x=0.05,
                y=0.98
            ),
            xaxis=dict(
                title="Year",
                title_font=dict(size=14, color='white'),
                tickfont=dict(size=11, color='white'),
                showgrid=False,
                side='bottom'
            ),
            yaxis=dict(
                title="Rating",
                title_font=dict(size=14, color='white'),
                tickfont=dict(size=12, color='white'),
                showgrid=False
            ),
            plot_bgcolor='rgba(15, 23, 42, 0.3)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=450,
            margin=dict(l=80, r=120, t=100, b=80)
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        
        # Remove debug messages after successful render
        st.success(f"✅ Heatmap displayed: {unique_years} year(s) × {unique_ratings} rating(s)")
        
        # Add interpretation guide
        with st.expander("💡 Reading the Heatmap"):
            avg_helpfulness = float(pivot.values.mean())
            min_helpfulness = float(pivot.values.min())
            max_helpfulness = float(pivot.values.max())
            
            if unique_years == 1:
                year_msg = f"**Note:** This heatmap shows data from only one year ({df_valid['review_year'].iloc[0]}). Upload data with multiple years to see temporal trends."
            else:
                year_msg = f"This heatmap covers {unique_years} years of review data."
            
            st.markdown(f"""
            **How to interpret:**
            
            - **Yellow-Orange cells:** High helpfulness ratio (up to {max_helpfulness:.3f}) - users found these reviews very useful
            - **Purple cells:** Lower helpfulness ratio (down to {min_helpfulness:.3f}) - reviews were less valued
            - **Pink cells:** Medium helpfulness
            - **Average:** {avg_helpfulness:.3f} across all year-rating combinations
            
            {year_msg}
            
            **Use Cases:**
            
            - Identify which rating levels consistently produce helpful content
            - Spot temporal trends in review quality (when multiple years available)
            - Target improvement efforts on low-helpfulness segments
            """)
            
    except Exception as e:
        st.error(f"Error creating heatmap visualization: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return


# ---------------- MAIN APP -----------------


def main():
    # Animated header
    st.markdown("""
    <div class='main-header'>
        <h1>🎬 AMAZON MOVIE REVIEW ANALYTICS</h1>
        <p>Big Data Analytics Dashboard | ALY6110 - Cloud & Big Data Management | Final Project</p>
    </div>
    """, unsafe_allow_html=True)

    # --------- DATA SOURCE ---------
    st.sidebar.markdown("### 📁 Data Source")
    uploaded = st.sidebar.file_uploader(
        "Upload dataset (.parquet or .csv)",
        type=["parquet", "csv"],
        help="Optional: Upload full dataset. Otherwise, sample data will be used.",
    )

    with st.spinner("🔄 Loading data..."):
        if uploaded is not None:
            st.sidebar.success("✅ Using uploaded file")
            raw_df = load_uploaded_raw(uploaded.getvalue(), uploaded.name)
        else:
            st.sidebar.info("📦 Using sample dataset")
            raw_df = load_sample_raw()

        df_full = preprocess_reviews(raw_df)

    # --------- FILTERS ---------
    df, advanced = sidebar_filters(df_full)

    # --------- KPI METRICS ---------
    total_reviews = len(df)
    avg_rating = float(df["rating"].mean()) if "rating" in df.columns else float("nan")
    unique_products = int(df["productId"].nunique()) if "productId" in df.columns else 0
    unique_users = int(df["userId"].nunique()) if "userId" in df.columns else 0

    # Beautiful metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color: white; margin: 0;'>📝 Reviews</h3>
            <h1 style='color: white; margin: 0.5rem 0;'>{total_reviews:,}</h1>
            <p style='color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem;'>Total in view</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        rating_display = f"{avg_rating:.2f}★" if not np.isnan(avg_rating) else "N/A"
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color: white; margin: 0;'>⭐ Rating</h3>
            <h1 style='color: white; margin: 0.5rem 0;'>{rating_display}</h1>
            <p style='color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem;'>Average score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color: white; margin: 0;'>🎬 Products</h3>
            <h1 style='color: white; margin: 0.5rem 0;'>{unique_products:,}</h1>
            <p style='color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem;'>Unique items</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color: white; margin: 0;'>👥 Reviewers</h3>
            <h1 style='color: white; margin: 0.5rem 0;'>{unique_users:,}</h1>
            <p style='color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem;'>Active users</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------- MAIN VISUALIZATIONS ---------
    
    # Tab-based navigation for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "⭐ Ratings", "💡 Helpfulness", "🏆 Leaders"])

    with tab1:
        st.markdown("<h2 class='section-header'>Temporal Analysis</h2>", unsafe_allow_html=True)
        
        col_a, col_b = st.columns([3, 1])
        with col_b:
            time_granularity = st.selectbox(
                "Time granularity",
                options=["Year", "Quarter", "Month"],
                index=0,
                help="Choose how to aggregate the time series"
            )
        
        plot_reviews_over_time_interactive(df, time_granularity=time_granularity)
        
        if advanced:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h2 class='section-header'>Engagement Patterns</h2>", unsafe_allow_html=True)
            plot_engagement_heatmap(df)

    with tab2:
        st.markdown("<h2 class='section-header'>Rating Distribution</h2>", unsafe_allow_html=True)
        plot_rating_distribution_interactive(df)

    with tab3:
        st.markdown("<h2 class='section-header'>Helpfulness Insights</h2>", unsafe_allow_html=True)
        plot_helpfulness_analysis_interactive(df)
        st.markdown("<br>", unsafe_allow_html=True)
        plot_scatter_helpful_votes_interactive(df)

    with tab4:
        if advanced:
            st.markdown("<h2 class='section-header'>Top Performers</h2>", unsafe_allow_html=True)
            plot_top_entities_interactive(df)
            
            with st.expander("💡 Why This Matters"):
                st.markdown("""
                **Strategic Insights:**
                
                - **Top Products:** Identify your "hero" SKUs that drive conversation and engagement
                - **Power Reviewers:** Recognize influencers who could be leveraged for marketing or loyalty programs
                - **Content Strategy:** Focus resources on high-traffic products and cultivate relationships with active reviewers
                """)
        else:
            st.info("💡 Enable 'Advanced analytics' in the sidebar to see top products and reviewers")

    # --------- DATA PREVIEW ---------
    st.markdown("<br><br>", unsafe_allow_html=True)
    with st.expander("🔍 Explore Raw Data", expanded=False):
        st.markdown("### Sample of Current Filtered View")
        
        # Add search functionality
        search_col = st.selectbox("Search in column", options=["None"] + list(df.columns))
        search_term = st.text_input("Search term", "")
        
        display_df = df.copy()
        if search_col != "None" and search_term:
            display_df = display_df[display_df[search_col].astype(str).str.contains(search_term, case=False, na=False)]
        
        st.dataframe(
            display_df.head(500),
            use_container_width=True,
            height=400
        )
        
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=display_df.to_csv(index=False).encode('utf-8'),
            file_name=f'amazon_reviews_filtered_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv',
        )

    # --------- FOOTER ---------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 2rem; color: rgba(255,255,255,0.6);'>
        <p>📊 Dashboard built with Streamlit • 🎨 Enhanced with Plotly</p>
        <p>Data: Amazon Movies & TV Reviews • Created for ALY6110</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
