import collections
import re
import time
from google_play_scraper import Sort, reviews
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="App Intelligence Dashboard", layout="wide")
st.title("⚡ Play Store Intelligence (10,000 Scale)")

# Sidebar Controls
st.sidebar.header("Configuration")
review_limit = st.sidebar.select_slider(
    "Number of Reviews to Ingest:",
    options=[500, 1000, 2500, 5000, 7500, 10000],
    value=2500,
)
market_country = st.sidebar.selectbox(
    "App Store Region:", ["us", "in", "gb", "ca", "au"], index=0
)

url_input = st.text_input(
    "Google Play Store URL:",
    "https://play.google.com/store/apps/details?id=com.spotify.music",
)

STOPWORDS = {
    "the",
    "and",
    "to",
    "a",
    "of",
    "in",
    "it",
    "is",
    "i",
    "that",
    "this",
    "for",
    "you",
    "my",
    "with",
    "on",
    "have",
    "app",
    "are",
    "so",
    "but",
    "be",
    "at",
    "can",
    "was",
    "not",
    "as",
    "or",
    "very",
    "just",
    "they",
    "like",
    "good",
    "bad",
    "all",
    "from",
    "an",
    "too",
    "really",
    "please",
    "when",
    "even",
    "more",
    "get",
    "will",
    "would",
    "after",
}


def extract_app_id(link: str):
  match = re.search(r"id=([a-zA-Z0-9._]+)", link)
  return match.group(1) if match else None


def clean_text_fast(text: str) -> str:
  text = re.sub(r"http\S+|www\S+", "", str(text))
  text = re.sub(r"[^\x00-\x7F]+", " ", text)
  return text.strip()


# Cached Ingestion Engine (Keeps 10k items in memory without re-fetching)
@st.cache_data(show_spinner=False)
def fetch_reviews_batch(app_id: str, target_count: int, country: str):
  collected_records = []
  continuation_token = None
  batch_size = 1000  # Safe Play Store pagination chunk

  while len(collected_records) < target_count:
    count_to_fetch = min(batch_size, target_count - len(collected_records))
    data, token = reviews(
        app_id,
        lang="en",
        country=country,
        sort=Sort.NEWEST,
        count=count_to_fetch,
        continuation_token=continuation_token,
    )
    if not data:
      break
    collected_records.extend(data)
    continuation_token = token
    if not continuation_token:
      break
    time.sleep(0.15)  # Micro-sleep to avoid Google rate limit blocks

  return collected_records


def extract_phrases(texts, n=2):
  phrases = []
  for t in texts:
    words = [
        w
        for w in re.findall(r"\b[a-z]{3,15}\b", str(t).lower())
        if w not in STOPWORDS
    ]
    if len(words) >= n:
      for i in range(len(words) - n + 1):
        phrases.append(" ".join(words[i : i + n]))
  return collections.Counter(phrases).most_common(8)


# Execution Flow
if st.button("Run Deep Analysis"):
  target_id = extract_app_id(url_input)

  if not target_id:
    st.error("Invalid URL format. Make sure it contains '?id=...'")
  else:
    progress_box = st.empty()
    progress_box.info(
        f"Ingesting up to {review_limit:,} reviews for '{target_id}'..."
    )

    try:
      raw_reviews = fetch_reviews_batch(
          target_id, review_limit, market_country
      )
      if not raw_reviews:
        progress_box.error("No reviews could be retrieved.")
        st.stop()

      df = pd.DataFrame(raw_reviews)
      progress_box.empty()

      # Fast Data Transformation
      df["cleaned_text"] = df["content"].apply(clean_text_fast)
      df["at"] = pd.to_datetime(df["at"])
      df["char_length"] = df["cleaned_text"].str.len()

      # Single-Instance VADER Engine
      analyzer = SentimentIntensityAnalyzer()
      df["compound"] = [
          analyzer.polarity_scores(x)["compound"] for x in df["cleaned_text"]
      ]

      conditions = [
          (df["compound"] >= 0.05),
          (df["compound"] <= -0.05),
      ]
      choices = ["Positive", "Negative"]
      df["Sentiment"] = np.select(conditions, choices, default="Neutral")

      # Sarcasm / Mismatch Flags
      df["Discrepancy"] = (
          (df["score"] >= 4) & (df["Sentiment"] == "Negative")
      ) | ((df["score"] <= 2) & (df["Sentiment"] == "Positive"))

      # Top Level KPI Calculations
      total = len(df)
      pos_count = (df["Sentiment"] == "Positive").sum()
      neg_count = (df["Sentiment"] == "Negative").sum()
      neu_count = (df["Sentiment"] == "Neutral").sum()
      nss = ((pos_count - neg_count) / total) * 100

      st.markdown("### Core Product Performance Metrics")
      m1, m2, m3, m4, m5 = st.columns(5)
      m1.metric("Analyzed", f"{total:,}")
      m2.metric("Avg Rating", f"{df['score'].mean():.2f} ⭐")
      m3.metric(
          "Net Sentiment (NSS)",
          f"{nss:+.1f}",
          help="Ranges from -100 to +100",
      )
      m4.metric("Negative %", f"{(neg_count/total)*100:.1f}%")
      m5.metric(
          "Flagged Reviews",
          f"{df['Discrepancy'].sum():,}",
          help="Rating vs Sentiment Mismatch",
      )

      st.markdown("---")

      # Visual Row 1: Distribution & Version Health
      c1, c2 = st.columns(2)
      with c1:
        st.subheader("Rating vs Sentiment Distribution")
        fig_hist = px.histogram(
            df,
            x="score",
            color="Sentiment",
            barmode="group",
            labels={"score": "Star Rating"},
            color_discrete_map={
                "Positive": "#2ecc71",
                "Negative": "#e74c3c",
                "Neutral": "#95a5a6",
            },
        )
        st.plotly_chart(fig_hist, use_container_width=True)

      with c2:
        st.subheader("Version Health Check")
        version_df = (
            df[df["appVersion"].notna()]
            .groupby("appVersion")
            .agg(avg_compound=("compound", "mean"), count=("score", "count"))
            .loc[lambda x: x["count"] >= 15]
            .sort_values(by="avg_compound", ascending=False)
            .head(7)
            .reset_index()
        )

        if not version_df.empty:
          fig_ver = px.bar(
              version_df,
              x="appVersion",
              y="avg_compound",
              labels={
                  "avg_compound": "Sentiment Score",
                  "appVersion": "Build Version",
              },
              color="avg_compound",
              color_continuous_scale="RdYlGn",
          )
          st.plotly_chart(fig_ver, use_container_width=True)
        else:
          st.info(
              "App versions not disclosed by this developer in review payloads."
          )

      # Visual Row 2: Deep Feedback Phrases
      st.markdown("---")
      st.subheader("Actionable Issue Detection (Key 2-Word Phrases)")
      k1, k2 = st.columns(2)

      with k1:
        st.write("🚨 **Critical Frustration Drivers**")
        neg_phrases = extract_phrases(
            df[df["Sentiment"] == "Negative"]["cleaned_text"]
        )
        if neg_phrases:
          neg_p_df = pd.DataFrame(
              neg_phrases, columns=["Complaint Theme", "Mentions"]
          )
          st.plotly_chart(
              px.bar(
                  neg_p_df,
                  x="Mentions",
                  y="Complaint Theme",
                  orientation="h",
                  color_discrete_sequence=["#e74c3c"],
              ).update_layout(yaxis={"autorange": "reversed"}),
              use_container_width=True,
          )

      with k2:
        st.write("🏆 **Product Strengths & Loved Features**")
        pos_phrases = extract_phrases(
            df[df["Sentiment"] == "Positive"]["cleaned_text"]
        )
        if pos_phrases:
          pos_p_df = pd.DataFrame(
              pos_phrases, columns=["Loved Feature", "Mentions"]
          )
          st.plotly_chart(
              px.bar(
                  pos_p_df,
                  x="Mentions",
                  y="Loved Feature",
                  orientation="h",
                  color_discrete_sequence=["#2ecc71"],
              ).update_layout(yaxis={"autorange": "reversed"}),
              use_container_width=True,
          )

      # Row 3: Behavioral Insight & Raw Data Export
      st.markdown("---")
      exp1, exp2 = st.columns([1, 2])
      with exp1:
        st.subheader("Effort vs Sentiment")
        char_lens = (
            df.groupby("Sentiment")["char_length"].mean().round(1).reset_index()
        )
        fig_len = px.bar(
            char_lens,
            x="Sentiment",
            y="char_length",
            labels={"char_length": "Avg Characters per Review"},
            color="Sentiment",
            color_discrete_map={
                "Positive": "#2ecc71",
                "Negative": "#e74c3c",
                "Neutral": "#95a5a6",
            },
        )
        st.plotly_chart(fig_len, use_container_width=True)

      with exp2:
        st.subheader("Targeted Data Investigation")
        inspect_filter = st.radio(
            "Show Rows:",
            [
                "Flagged Discrepancies (Sarcasm/Misratings)",
                "Top Helpful Reviews",
                "All",
            ],
            horizontal=True,
        )

        if inspect_filter == "Flagged Discrepancies (Sarcasm/Misratings)":
          inspect_df = df[df["Discrepancy"]]
        elif inspect_filter == "Top Helpful Reviews":
          inspect_df = df.sort_values(by="thumbsUpCount", ascending=False)
        else:
          inspect_df = df

        st.dataframe(
            inspect_df[[
                "score",
                "Sentiment",
                "compound",
                "thumbsUpCount",
                "cleaned_text",
            ]].head(40),
            use_container_width=True,
        )

      # CSV Export
      csv_buffer = df[[
          "userName",
          "score",
          "at",
          "Sentiment",
          "compound",
          "thumbsUpCount",
          "appVersion",
          "cleaned_text",
      ]].to_csv(index=False)

      st.download_button(
          label=f"📥 Download Processed Dataset ({len(df):,} Rows)",
          data=csv_buffer,
          file_name=f"{target_id}_intelligence_10k.csv",
          mime="text/csv",
      )

    except Exception as exc:
      progress_box.empty()
      st.error(f"Processing Encountered an Issue: {exc}")
