import collections
import re
import time
from google_play_scraper import Sort, reviews
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Enterprise App Review Intelligence", layout="wide")
st.title("🛡️ Enterprise Play Store Review & Competitive Intelligence")

# Sidebar Controls
st.sidebar.header("Extraction Settings")
analysis_mode = st.sidebar.radio("Operating Mode:", ["Single App Deep Dive", "Competitor Head-to-Head"])

total_target = st.sidebar.select_slider(
    "Reviews to Ingest (per app):",
    options=[2000, 5000, 10000, 20000],
    value=5000
)
market_country = st.sidebar.selectbox(
    "Country Store:",
    ["us", "in", "gb", "ca", "au", "de", "fr"],
    index=0
)

# Text Processing Constants
STOPWORDS = {
    "the", "and", "to", "a", "of", "in", "it", "is", "i", "that", "this",
    "for", "you", "my", "with", "on", "have", "app", "are", "so", "but",
    "be", "at", "can", "was", "not", "as", "or", "very", "just", "they",
    "like", "good", "bad", "all", "from", "an", "too", "really", "please",
    "when", "even", "more", "get", "will", "would", "after"
}

THEME_TAXONOMY = {
    "Crashes & Stability": ["crash", "freeze", "bug", "lag", "glitch", "close", "black", "stop"],
    "Billing, Ads & Subs": ["ad", "ads", "money", "pay", "price", "premium", "subscription", "cost", "refund"],
    "Account & Auth": ["login", "log", "password", "account", "otp", "verify", "sign", "email"],
    "Speed & Battery": ["battery", "drain", "slow", "heat", "ram", "loading", "data", "storage"],
    "UI & Experience": ["update", "interface", "design", "layout", "font", "dark", "ugly", "confusing"]
}

WISHLIST_PATTERNS = [
    r"please add ([\w\s]{4,35})",
    r"wish (?:there was|it had|you would add) ([\w\s]{4,35})",
    r"bring back ([\w\s]{4,35})",
    r"would be (?:better|great|awesome) if ([\w\s]{4,35})",
    r"need (?:an?|to) ([\w\s]{4,35})"
]

CHURN_PATTERNS = [
    r"\b(uninstalled|uninstalling|un-install)\b",
    r"\b(switching to|moving to|going to)\b",
    r"\b(deleted the app|deleting)\b",
    r"\b(cancelled|canceling|cancel) (subscription|premium|membership)\b",
    r"\b(ruined|worst) (update|app)\b"
]

def extract_app_id(link: str):
    match = re.search(r"id=([a-zA-Z0-9._]+)", link)
    return match.group(1) if match else link.strip()

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", str(text))
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()

def classify_review_themes(text: str):
    found = []
    lowered = text.lower()
    for theme, keywords in THEME_TAXONOMY.items():
        if any(re.search(r"\b" + kw + r"\b", lowered) for kw in keywords):
            found.append(theme)
    return found if found else ["General Feedback"]

def extract_key_phrases(texts, n=2):
    phrases = []
    for t in texts:
        words = [w for w in re.findall(r"\b[a-z]{3,15}\b", str(t).lower()) if w not in STOPWORDS]
        if len(words) >= n:
            for i in range(len(words) - n + 1):
                phrases.append(" ".join(words[i:i + n]))
    return collections.Counter(phrases).most_common(6)

def mine_wishlist(texts):
    requests = []
    for t in texts:
        t_low = str(t).lower()
        for p in WISHLIST_PATTERNS:
            matches = re.findall(p, t_low)
            for m in matches:
                clean_m = re.sub(r"[^\w\s]", "", m).strip()
                if len(clean_m.split()) >= 2:
                    requests.append(clean_m)
    return collections.Counter(requests).most_common(8)

def detect_churn(text: str) -> bool:
    t_low = str(text).lower()
    return any(re.search(p, t_low) for p in CHURN_PATTERNS)

def phased_review_collector(app_id, target, country):
    all_data = []
    token = None
    phase_size = 2000
    pbar = st.progress(0)
    stat = st.empty()

    while len(all_data) < target:
        pull_target = min(phase_size, target - len(all_data))
        stat.info(f"[{app_id}] Fetching {len(all_data):,} of {target:,} reviews...")
        try:
            batch, token = reviews(
                app_id,
                lang="en",
                country=country,
                sort=Sort.NEWEST,
                count=pull_target,
                continuation_token=token
            )
        except Exception:
            break

        if not batch:
            break

        all_data.extend(batch)
        pbar.progress(min(len(all_data) / target, 1.0))

        if not token:
            break
        time.sleep(0.25)

    pbar.empty()
    stat.empty()
    return all_data

def process_dataframe(data):
    df = pd.DataFrame(data)
    df["cleaned_text"] = df["content"].apply(clean_text)
    df["at"] = pd.to_datetime(df["at"])
    df["day_of_week"] = df["at"].dt.day_name()
    df["hour"] = df["at"].dt.hour

    analyzer = SentimentIntensityAnalyzer()
    df["compound"] = [analyzer.polarity_scores(x)["compound"] for x in df["cleaned_text"]]

    df["Sentiment"] = np.select(
        [df["compound"] >= 0.05, df["compound"] <= -0.05],
        ["Positive", "Negative"],
        default="Neutral"
    )

    df["Themes"] = df["cleaned_text"].apply(classify_review_themes)
    df["Churn_Risk"] = df["cleaned_text"].apply(detect_churn)
    
    # Customer Support Gap detection
    df["has_reply"] = df["replyContent"].notna()
    if "repliedAt" in df.columns:
        df["repliedAt"] = pd.to_datetime(df["repliedAt"])
        df["reply_delay_days"] = (df["repliedAt"] - df["at"]).dt.total_seconds() / 86400.0
    else:
        df["reply_delay_days"] = np.nan

    return df

# =========================================================
# MODE 1: SINGLE APP DEEP DIVE
# =========================================================
if analysis_mode == "Single App Deep Dive":
    url_input = st.text_input("Play Store URL:", "https://play.google.com/store/apps/details?id=com.spotify.music")

    if st.button("Run Comprehensive Intelligence Pipeline"):
        app_id = extract_app_id(url_input)
        raw = phased_review_collector(app_id, total_target, market_country)

        if not raw:
            st.error("No reviews retrieved. Check the App ID and region.")
            st.stop()

        df = process_dataframe(raw)
        st.success(f"Ingested & analyzed {len(df):,} reviews for '{app_id}'!")

        total = len(df)
        pos_c = (df["Sentiment"] == "Positive").sum()
        neg_c = (df["Sentiment"] == "Negative").sum()
        nss = ((pos_c - neg_c) / total) * 100
        churn_c = df["Churn_Risk"].sum()

        # Scorecard
        st.markdown("### 📈 Executive Performance Scorecard")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Ingested", f"{total:,}")
        k2.metric("Average Rating", f"{df['score'].mean():.2f} ⭐")
        k3.metric("Net Sentiment (NSS)", f"{nss:+.1f}")
        k4.metric("Negative Ratio", f"{(neg_c/total)*100:.1f}%")
        k5.metric("Churn Threats", f"{churn_c:,} ({(churn_c/total)*100:.1f}%)")

        st.markdown("---")

        # Visuals: Thematic Clustering & Version Health
        st.subheader("🔍 Thematic Issue Clustering & Build Stability")
        c_th1, c_th2 = st.columns(2)

        exploded_df = df.explode("Themes")
        theme_counts = exploded_df[exploded_df["Sentiment"] == "Negative"]["Themes"].value_counts()
        theme_df = theme_counts.reset_index()
        theme_df.columns = ["Root Cause Theme", "Negative Mentions"]

        with c_th1:
            fig_themes = px.bar(
                theme_df,
                x="Negative Mentions",
                y="Root Cause Theme",
                orientation="h",
                color="Negative Mentions",
                color_continuous_scale="Reds"
            ).update_layout(yaxis={'autorange': 'reversed'})
            st.plotly_chart(fig_themes, use_container_width=True)

        with c_th2:
            version_summary = (
                df[df["appVersion"].notna()]
                .groupby("appVersion")
                .agg(Volume=("score", "count"), Avg_Rating=("score", "mean"), Sentiment_Polarity=("compound", "mean"))
                .loc[lambda x: x["Volume"] >= 20]
                .sort_values(by="Volume", ascending=False)
                .head(8)
                .reset_index()
            )
            if not version_summary.empty:
                fig_ver = px.scatter(
                    version_summary,
                    x="Avg_Rating",
                    y="Sentiment_Polarity",
                    size="Volume",
                    color="Sentiment_Polarity",
                    text="appVersion",
                    labels={"Avg_Rating": "Star Rating", "Sentiment_Polarity": "Sentiment Polarity"},
                    color_continuous_scale="RdYlGn"
                ).update_traces(textposition='top center')
                st.plotly_chart(fig_ver, use_container_width=True)
            else:
                st.info("App versions not disclosed by this developer in review payloads.")

        # Keywords Row
        st.markdown("---")
        st.subheader("🗣️ Recurring Reviewer Keywords")
        col_p1, col_p2 = st.columns(2)
        neg_phrases = extract_key_phrases(df[df["Sentiment"] == "Negative"]["cleaned_text"])
        pos_phrases = extract_key_phrases(df[df["Sentiment"] == "Positive"]["cleaned_text"])

        with col_p1:
            st.write("**Top Complaint Drivers**")
            if neg_phrases:
                p_df = pd.DataFrame(neg_phrases, columns=["Phrase", "Count"])
                st.plotly_chart(px.bar(p_df, x="Count", y="Phrase", orientation="h", color_discrete_sequence=["#e74c3c"]).update_layout(yaxis={'autorange': 'reversed'}), use_container_width=True)

        with col_p2:
            st.write("**Top Praise Drivers**")
            if pos_phrases:
                p_df_pos = pd.DataFrame(pos_phrases, columns=["Phrase", "Count"])
                st.plotly_chart(px.bar(p_df_pos, x="Count", y="Phrase", orientation="h", color_discrete_sequence=["#2ecc71"]).update_layout(yaxis={'autorange': 'reversed'}), use_container_width=True)

        # Advanced Modules: Wishlist & Support Gap Analysis
        st.markdown("---")
        st.subheader("💡 Feature Wishlist & Support Operations Audit")
        col_w, col_s = st.columns(2)

        with col_w:
            st.write("🎯 **Top Community Feature Requests (Auto-Mined)**")
            wishes = mine_wishlist(df["cleaned_text"])
            if wishes:
                wish_df = pd.DataFrame(wishes, columns=["Feature Request Phrase", "Frequency"])
                st.dataframe(wish_df, use_container_width=True)
            else:
                st.write("No explicit pattern matches found for direct feature requests.")

        with col_s:
            st.write("🎧 **Customer Support Responsiveness**")
            crit_reviews = df[df["score"] <= 2]
            crit_total = len(crit_reviews)
            crit_replied = crit_reviews["has_reply"].sum()
            crit_rate = (crit_replied / crit_total * 100) if crit_total > 0 else 0.0
            avg_reply_time = crit_reviews["reply_delay_days"].dropna().median()

            st.metric("Critical Review Reply Rate (1-2 Stars)", f"{crit_rate:.1f}%")
            if not np.isnan(avg_reply_time):
                st.metric("Median Support Response Time", f"{avg_reply_time:.1f} Days")
            else:
                st.metric("Median Support Response Time", "N/A (No replies logged)")

        # Temporal Infrastructure Peak Analysis
        st.markdown("---")
        st.subheader("⏰ Incident Timing: Peak Complaint Periods")
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_summary = df[df["Sentiment"] == "Negative"]["day_of_week"].value_counts().reindex(dow_order).fillna(0).reset_index()
        dow_summary.columns = ["Day of Week", "Negative Review Spike"]
        st.plotly_chart(px.bar(dow_summary, x="Day of Week", y="Negative Review Spike", color_discrete_sequence=["#e67e22"]), use_container_width=True)

        # Executive Readout Section
        st.markdown("---")
        st.header("📋 Automated Executive Readout & Action Plan")
        if nss >= 35:
            health_color = "green"
            health_status = "Healthy / High User Retention"
            health_summary = "Product sentiment is resilient. Negative volume remains contained within acceptable baseline bounds."
        elif nss >= 0:
            health_color = "orange"
            health_status = "Moderate / Churn Warning"
            health_summary = "Substantial friction points exist. Issues are offsetting positive product advocacy."
        else:
            health_color = "red"
            health_status = "Critical / Escalated Friction"
            health_summary = "Negative sentiment dominates incoming reviews. High risk of immediate uninstalls."

        st.markdown(f"#### Health Diagnosis: :{health_color}[{health_status}]")
        st.write(health_summary)

        st.subheader("🎯 Primary Engineering & Product Priorities")
        non_gen = theme_df[theme_df["Root Cause Theme"] != "General Feedback"]
        if not non_gen.empty:
            st.markdown(f"* **Primary Friction Area:** Focus development sprints on `{non_gen.iloc[0]['Root Cause Theme']}` which accounted for **{non_gen.iloc[0]['Negative Mentions']:,} negative complaints**.")
        if wishes:
            st.markdown(f"* **Highest-Demand Feature Request:** Users frequently requested: *\"{wishes[0][0]}\"* ({wishes[0][1]} times).")
        if churn_c > 0:
            st.markdown(f"* **Churn Threat Containment:** Detected **{churn_c:,} users explicitly claiming to have uninstalled or cancelled**.")

        st.download_button(
            "📥 Export Full Processed Dataset",
            data=df.to_csv(index=False),
            file_name=f"{app_id}_full_intelligence.csv",
            mime="text/csv"
        )

# =========================================================
# MODE 2: COMPETITOR HEAD-TO-HEAD
# =========================================================
else:
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        app1_url = st.text_input("App A URL:", "https://play.google.com/store/apps/details?id=com.spotify.music")
    with col_c2:
        app2_url = st.text_input("App B URL:", "https://play.google.com/store/apps/details?id=com.apple.android.music")

    if st.button("Run Head-to-Head Benchmark"):
        id_a = extract_app_id(app1_url)
        id_b = extract_app_id(app2_url)

        with st.spinner("Extracting App A reviews..."):
            raw_a = phased_review_collector(id_a, total_target, market_country)
        with st.spinner("Extracting App B reviews..."):
            raw_b = phased_review_collector(id_b, total_target, market_country)

        if not raw_a or not raw_b:
            st.error("Failed to fetch reviews for one or both apps. Check identifiers.")
            st.stop()

        df_a = process_dataframe(raw_a)
        df_b = process_dataframe(raw_b)

        df_a["App"] = id_a
        df_b["App"] = id_b
        combined = pd.concat([df_a, df_b], ignore_index=True)

        nss_a = (((df_a["Sentiment"] == "Positive").sum() - (df_a["Sentiment"] == "Negative").sum()) / len(df_a)) * 100
        nss_b = (((df_b["Sentiment"] == "Positive").sum() - (df_b["Sentiment"] == "Negative").sum()) / len(df_b)) * 100

        st.markdown("### 🥊 Head-to-Head Scorecard")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{id_a} Rating", f"{df_a['score'].mean():.2f} ⭐")
        m2.metric(f"{id_b} Rating", f"{df_b['score'].mean():.2f} ⭐")
        m3.metric(f"{id_a} Net Sentiment", f"{nss_a:+.1f}")
        m4.metric(f"{id_b} Net Sentiment", f"{nss_b:+.1f}")

        st.markdown("---")
        st.subheader("Theme Comparison: Complaint Breakdown")
        comp_exploded = combined.explode("Themes")
        comp_neg = comp_exploded[comp_exploded["Sentiment"] == "Negative"].groupby(["Themes", "App"]).size().reset_index(name="Negative Mentions")

        st.plotly_chart(
            px.bar(comp_neg, x="Themes", y="Negative Mentions", color="App", barmode="group", color_discrete_sequence=["#2ecc71", "#e74c3c"]),
            use_container_width=True
        )
