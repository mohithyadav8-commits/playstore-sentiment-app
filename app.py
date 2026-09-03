import re
import pandas as pd
import plotly.express as px
import streamlit as st
from google_play_scraper import Sort, reviews
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Play Store Review Analyzer", layout="wide")
st.title("Play Store Sentiment Analyzer")

url = st.text_input(
    "Paste Google Play Store App Link:",
    "https://play.google.com/store/apps/details?id=com.spotify.music",
)

review_count = st.slider(
    "Number of reviews to analyze:",
    min_value=50,
    max_value=1000,
    value=200,
    step=50,
)


def extract_app_id(link):
  match = re.search(r"id=([a-zA-Z0-9._]+)", link)
  return match.group(1) if match else None


def clean_text(text):
  text = re.sub(r"http\S+|www\S+", "", str(text))
  text = re.sub(r"[^\x00-\x7F]+", " ", text)
  return text.strip()


if st.button("Extract and Analyze Reviews"):
  app_id = extract_app_id(url)

  if not app_id:
    st.error(
        "Could not detect App ID. Please make sure the link includes '?id=...'"
    )
  else:
    with st.spinner(f"Fetching reviews for {app_id}..."):
      try:
        data, _ = reviews(
            app_id,
            lang="en",
            country="us",
            sort=Sort.NEWEST,
            count=review_count,
        )

        if not data:
          st.warning("No reviews found for this app.")
          st.stop()

        df = pd.DataFrame(data)
        df["cleaned_text"] = df["content"].apply(clean_text)

        analyzer = SentimentIntensityAnalyzer()

        def analyze(text):
          score = analyzer.polarity_scores(text)["compound"]
          if score >= 0.05:
            return "Positive"
          elif score <= -0.05:
            return "Negative"
          return "Neutral"

        df["Sentiment"] = df["cleaned_text"].apply(analyze)

        pos = (df["Sentiment"] == "Positive").sum()
        neg = (df["Sentiment"] == "Negative").sum()
        neu = (df["Sentiment"] == "Neutral").sum()
        total = len(df)

        col1, col2, col3 = st.columns(3)
        col1.metric("Positive", f"{pos} ({pos/total*100:.1f}%)")
        col2.metric("Negative", f"{neg} ({neg/total*100:.1f}%)")
        col3.metric("Neutral", f"{neu} ({neu/total*100:.1f}%)")

        fig = px.pie(
            df,
            names="Sentiment",
            title="Sentiment Breakdown",
            color="Sentiment",
            color_discrete_map={
                "Positive": "#2ecc71",
                "Negative": "#e74c3c",
                "Neutral": "#95a5a6",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Sample Reviews")
        st.dataframe(
            df[["userName", "score", "Sentiment", "cleaned_text"]].head(25)
        )

      except Exception as err:
        st.error(f"Error fetching data: {err}")
