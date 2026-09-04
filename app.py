import collections
import re
import time
from google_play_scraper import Sort, reviews
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Multi-Phase App Intelligence", layout="wide")
st.title("🛡️ Scaled Multi-Phase Review Intelligence")

# Sidebar Controls
st.sidebar.header("Extraction Phases")
total_target = st.sidebar.select_slider(
    "Total Reviews to Ingest across Phases:",
    options=[5000, 10000, 20000, 30000],
    value=10000,
)
market_country = st.sidebar.selectbox(
    "Country Store:", ["us", "in", "gb", "ca", "au"], index=0
)

url_input = st.text_input(
    "Play Store URL:",
    "https://play.google.com/store/apps/details?id=com.spotify.music",
)

# Text Processing Vocabularies
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

THEME_TAXONOMY = {
    "Crashes & Stability": [
        "crash",
        "freeze",
        "bug",
        "lag",
        "glitch",
        "close",
        "black",
        "stop",
    ],
    "Billing, Ads & Subs": [
        "ad",
        "ads",
        "money",
        "pay",
        "price",
        "premium",
        "subscription",
        "cost",
        "refund",
    ],
    "Account & Auth": [
        "login",
        "log",
        "password",
        "account",
        "otp",
        "verify",
        "sign",
        "email",
    ],
    "Speed & Battery": [
        "battery",
        "drain",
        "slow",
        "heat",
        "ram",
        "loading",
        "data",
        "storage",
    ],
    "UI & Experience": [
        "update",
        "interface",
        "design",
        "layout",
        "font",
        "dark",
        "ugly",
        "confusing",
    ],
}


def extract_app_id(link: str):
  match = re.search(r"id=([a-zA-Z0-9._]+)", link)
  return match.group(1) if match else None


def clean_text(text: str) -> str:
  text = re.sub(r"http\S+|www\S+", "", str(text))
  text = re.sub(r"[^\x00-\x7F]+", " ", text)
  return text.strip()


def classify_review_themes(text: str):
  found_themes = []
  lowered = text.lower()
  for theme, keywords in THEME_TAXONOMY.items():
    if any(re.search(r"\b" + kw + r"\b", lowered) for kw in keywords):
      found_themes.append(theme)
  return found_themes if found_themes else ["General Feedback"]


def extract_key_phrases(texts, n=2):
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


# Phased Multi-batch Extractor
def phased_review_collector(app_id, target, country):
  all_data = []
  token = None
  phase_size = 2000  # Pull in 2k chunks to keep memory flat
  progress_bar = st.progress(0)
  status_feed = st.empty()

  while len(all_data) < target:
    pull_target = min(phase_size, target - len(all_data))
    status_feed.info(
        f"Phase {len(all_data)//phase_size + 1}: Extracted {len(all_data):,} of"
        f" {target:,} reviews..."
    )

    batch_res, token = reviews(
        app_id,
        lang="en",
        country=country,
        sort=Sort.NEWEST,
        count=pull_target,
        continuation_token=token,
    )

    if not batch_res:
      break

    all_data.extend(batch_res)
    progress_bar.progress(min(len(all_data) / target, 1.0))

    if not token:
      break
    time.sleep(0.3)  # Cooldown between phases

  progress_bar.empty()
  status_feed.empty()
  return all_data


# Main App Flow
if st.button("Start Phased Ingestion & Analysis"):
  target_id = extract_app_id(url_input)

  if not target_id:
    st.error("Invalid Play Store URL. Please include '?id=...'")
  else:
    with st.spinner("Connecting to Google Play Store..."):
      raw_records = phased_review_collector(
          target_id, total_target, market_country
      )

    if not raw_records:
      st.error("No reviews could be extracted. Please check the App ID/Region.")
      st.stop()

    df = pd.DataFrame(raw_records)
    st.success(f"Successfully processed {len(df):,} total reviews!")

    # 1. Text & Sentiment Engine
    df["cleaned_text"] = df["content"].apply(clean_text)
    df["at"] = pd.to_datetime(df["at"])

    analyzer = SentimentIntensityAnalyzer()
    df["compound"] = [
        analyzer.polarity_scores(x)["compound"] for x in df["cleaned_text"]
    ]

    df["Sentiment"] = np.select(
        [df["compound"] >= 0.05, df["compound"] <= -0.05],
        ["Positive", "Negative"],
        default="Neutral",
    )

    # 2. Extract Thematic Clusters
    df["Themes"] = df["cleaned_text"].apply(classify_review_themes)
    exploded_df = df.explode("Themes")

    # High-Level Metrics
    total = len(df)
    pos_c = (df["Sentiment"] == "Positive").sum()
    neg_c = (df["Sentiment"] == "Negative").sum()
    nss = ((pos_c - neg_c) / total) * 100

    st.markdown("### 📈 Executive Performance Scorecard")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Ingested", f"{total:,}")
    kpi2.metric("Average Rating", f"{df['score'].mean():.2f} ⭐")
    kpi3.metric("Net Sentiment Score", f"{nss:+.1f}")
    kpi4.metric("Negative Ratio", f"{(neg_c/total)*100:.1f}%")

    st.markdown("---")

    # Thematic Diagnosis
    st.subheader("🔍 Thematic Issue Clustering")
    theme_counts = exploded_df[exploded_df["Sentiment"] == "Negative"][
        "Themes"
    ].value_counts()
    theme_df = theme_counts.reset_index()
    theme_df.columns = ["Root Cause Theme", "Negative Mentions"]

    fig_themes = px.bar(
        theme_df,
        x="Negative Mentions",
        y="Root Cause Theme",
        orientation="h",
        color="Negative Mentions",
        color_continuous_scale="Reds",
    )
    fig_themes.update_layout(yaxis={"autorange": "reversed"})
    st.plotly_chart(fig_themes, use_container_width=True)

    # Version Health Check
    st.markdown("---")
    st.subheader("📦 Update & Version Stability Breakdown")
    version_summary = (
        df[df["appVersion"].notna()]
        .groupby("appVersion")
        .agg(
            Volume=("score", "count"),
            Avg_Rating=("score", "mean"),
            Sentiment_Polarity=("compound", "mean"),
        )
        .loc[lambda x: x["Volume"] >= 25]
        .sort_values(by="Volume", ascending=False)
        .head(10)
        .reset_index()
    )

    if not version_summary.empty:
      fig_version = px.scatter(
          version_summary,
          x="Avg_Rating",
          y="Sentiment_Polarity",
          size="Volume",
          color="Sentiment_Polarity",
          text="appVersion",
          labels={
              "Avg_Rating": "Star Rating (1-5)",
              "Sentiment_Polarity": "Sentiment (-1 to 1)",
          },
          color_continuous_scale="RdYlGn",
      )
      fig_version.update_traces(textposition="top center")
      st.plotly_chart(fig_version, use_container_width=True)
    else:
      st.info(
          "Build version numbers were not provided in these review entries."
      )

    # Recurring Phrase Identification
    st.markdown("---")
    st.subheader("🗣️ Recurring Reviewer Keywords")
    col_p1, col_p2 = st.columns(2)

    with col_p1:
      st.write("**Top Recurring Complaint Phrases**")
      neg_phrases = extract_key_phrases(
          df[df["Sentiment"] == "Negative"]["cleaned_text"]
      )
      if neg_phrases:
        p_df = pd.DataFrame(neg_phrases, columns=["Phrase", "Count"])
        st.plotly_chart(
            px.bar(
                p_df,
                x="Count",
                y="Phrase",
                orientation="h",
                color_discrete_sequence=["#e74c3c"],
            ).update_layout(yaxis={"autorange": "reversed"}),
            use_container_width=True,
        )

    with col_p2:
      st.write("**Top Recurring Praise Phrases**")
      pos_phrases = extract_key_phrases(
          df[df["Sentiment"] == "Positive"]["cleaned_text"]
      )
      if pos_phrases:
        p_df_pos = pd.DataFrame(pos_phrases, columns=["Phrase", "Count"])
        st.plotly_chart(
            px.bar(
                p_df_pos,
                x="Count",
                y="Phrase",
                orientation="h",
                color_discrete_sequence=["#2ecc71"],
            ).update_layout(yaxis={"autorange": "reversed"}),
            use_container_width=True,
        )

    # Download Dataset
    st.markdown("---")
    csv_out = df[[
        "userName",
        "score",
        "at",
        "Sentiment",
        "compound",
        "appVersion",
        "cleaned_text",
    ]].to_csv(index=False)
    st.download_button(
        f"📥 Download All Cleaned Records ({len(df):,} Rows)",
        data=csv_out,
        file_name=f"{target_id}_deep_analysis.csv",
        mime="text/csv",
    )
