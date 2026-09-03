import collections
import re
from google_play_scraper import Sort, reviews, reviews_all
import pandas as pd
import plotly.express as px
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="Deep Play Store Review Intelligence", layout="wide"
)
st.title("📊 Deep Play Store Review Intelligence")

# Sidebar configurations
st.sidebar.header("Extraction Settings")
mode = st.sidebar.radio(
    "Scrape Mode",
    [
        "Targeted Sample (Fast, 50 - 5,000)",
        "Scrape ALL Reviews (Slow / Small Apps)",
    ],
)

if mode == "Targeted Sample (Fast, 50 - 5,000)":
  review_count = st.sidebar.slider(
      "Sample Size:", min_value=100, max_value=5000, value=500, step=100
  )
else:
  review_count = None
  st.sidebar.warning(
      "Note: For apps with millions of reviews, this may take a long time or"
      " hit Google rate limits."
  )

url = st.text_input(
    "Google Play Store App URL:",
    "https://play.google.com/store/apps/details?id=com.spotify.music",
)

STOPWORDS = set([
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
])


def extract_app_id(link):
  match = re.search(r"id=([a-zA-Z0-9._]+)", link)
  return match.group(1) if match else None


def clean_text(text):
  text = re.sub(r"http\S+|www\S+", "", str(text))
  text = re.sub(r"[^\x00-\x7F]+", " ", text)
  return text.strip()


def extract_keywords(texts):
  words = []
  for t in texts:
    tokens = re.findall(r"\b[a-z]{3,15}\b", str(t).lower())
    words.extend([w for w in tokens if w not in STOPWORDS])
  return collections.Counter(words).most_common(12)


if st.button("Run Full Intelligence Analysis"):
  app_id = extract_app_id(url)

  if not app_id:
    st.error("Invalid URL: Please make sure the link includes '?id=...'.")
  else:
    status = st.empty()
    status.info(f"Connecting to Play Store for '{app_id}'...")

    try:
      if mode == "Scrape ALL Reviews (Slow / Small Apps)":
        status.info(
            "Extracting all reviews via pagination... Please wait."
        )  #
        data = reviews_all(
            app_id, lang="en", country="us", sleep_milliseconds=100
        )  #
      else:
        status.info(f"Extracting {review_count} newest reviews...")
        data, _ = reviews(
            app_id,
            lang="en",
            country="us",
            sort=Sort.NEWEST,
            count=review_count,
        )

      if not data:
        status.warning("No reviews found for this app.")
        st.stop()

      df = pd.DataFrame(data)
      status.success(f"Successfully collected {len(df):,} reviews!")

      # Process Data
      df["cleaned_text"] = df["content"].apply(clean_text)
      df["at"] = pd.to_datetime(df["at"])

      analyzer = SentimentIntensityAnalyzer()

      def score_sentiment(text):
        return analyzer.polarity_scores(text)["compound"]

      df["polarity"] = df["cleaned_text"].apply(score_sentiment)

      def categorize(score):
        if score >= 0.05:
          return "Positive"
        elif score <= -0.05:
          return "Negative"
        return "Neutral"

      df["Sentiment"] = df["polarity"].apply(categorize)

      # Section 1: Overview KPIs
      total_reviews = len(df)
      avg_rating = df["score"].mean()
      pos_pct = (df["Sentiment"] == "Positive").mean() * 100
      neg_pct = (df["Sentiment"] == "Negative").mean() * 100

      st.markdown("---")
      col1, col2, col3, col4 = st.columns(4)
      col1.metric("Total Reviews Analyzed", f"{total_reviews:,}")
      col2.metric("Average Star Rating", f"{avg_rating:.2f} ⭐")
      col3.metric("Positive Sentiment", f"{pos_pct:.1f}%")
      col4.metric("Negative Sentiment", f"{neg_pct:.1f}%")

      # Section 2: Charts (Rating vs Sentiment & Timeline)
      col_chart1, col_chart2 = st.columns(2)

      with col_chart1:
        st.subheader("Star Rating vs Sentiment")
        fig_bar = px.histogram(
            df,
            x="score",
            color="Sentiment",
            barmode="group",
            title="Distribution of Sentiment Across Star Ratings",
            labels={"score": "Star Rating (1-5)"},
            color_discrete_map={
                "Positive": "#2ecc71",
                "Negative": "#e74c3c",
                "Neutral": "#95a5a6",
            },
        )
        st.plotly_chart(fig_bar, use_container_width=True)

      with col_chart2:
        st.subheader("Sentiment Timeline Trend")
        df_time = (
            df.set_index("at")
            .resample("W")["polarity"]
            .mean()
            .reset_index()
            .dropna()
        )
        fig_line = px.line(
            df_time,
            x="at",
            y="polarity",
            title="Average Weekly Sentiment Polarity",
            labels={"at": "Date", "polarity": "Sentiment (-1 to +1)"},
        )
        st.plotly_chart(fig_line, use_container_width=True)

      # Section 3: Complaint Drivers vs Praise Drivers
      st.markdown("---")
      st.subheader("What Users Are Actually Talking About")
      col_k1, col_k2 = st.columns(2)

      neg_reviews = df[df["Sentiment"] == "Negative"]["cleaned_text"]
      pos_reviews = df[df["Sentiment"] == "Positive"]["cleaned_text"]

      with col_k1:
        st.write("🔥 **Top Negative Themes (Complaints/Bugs)**")
        neg_kw = extract_keywords(neg_reviews)
        if neg_kw:
          kw_df = pd.DataFrame(neg_kw, columns=["Keyword", "Frequency"])
          fig_neg = px.bar(
              kw_df,
              x="Frequency",
              y="Keyword",
              orientation="h",
              color_discrete_sequence=["#e74c3c"],
          )
          fig_neg.update_layout(yaxis={"autorange": "reversed"})
          st.plotly_chart(fig_neg, use_container_width=True)
        else:
          st.write("No recurring complaint themes found.")

      with col_k2:
        st.write("⭐ **Top Positive Themes (Praise/Features)**")
        pos_kw = extract_keywords(pos_reviews)
        if pos_kw:
          kw_df_pos = pd.DataFrame(pos_kw, columns=["Keyword", "Frequency"])
          fig_pos = px.bar(
              kw_df_pos,
              x="Frequency",
              y="Keyword",
              orientation="h",
              color_discrete_sequence=["#2ecc71"],
          )
          fig_pos.update_layout(yaxis={"autorange": "reversed"})
          st.plotly_chart(fig_pos, use_container_width=True)
        else:
          st.write("No recurring positive themes found.")

      # Section 4: Export Raw Data
      st.markdown("---")
      st.subheader("Export Cleaned Dataset")
      csv_data = df[[
          "userName",
          "score",
          "at",
          "Sentiment",
          "polarity",
          "cleaned_text",
      ]].to_csv(index=False)
      st.download_button(
          label="📥 Download Cleaned Reviews as CSV",
          data=csv_data,
          file_name=f"{app_id}_sentiment_analysis.csv",
          mime="text/csv",
      )

      # Section 5: Filterable Table
      st.subheader("Review Explorer")
      filter_choice = st.selectbox(
          "Filter by Sentiment:", ["All", "Negative", "Positive", "Neutral"]
      )
      if filter_choice != "All":
        filtered_df = df[df["Sentiment"] == filter_choice]
      else:
        filtered_df = df

      st.dataframe(
          filtered_df[[
              "score",
              "Sentiment",
              "at",
              "cleaned_text",
              "thumbsUpCount",
          ]].head(50),
          use_container_width=True,
      )

    except Exception as err:
      status.empty()
      st.error(f"Execution Error: {err}")
