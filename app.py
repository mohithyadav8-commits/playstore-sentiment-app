import collections
import re
import time
from google_play_scraper import Sort, reviews
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="Enterprise App Review Intelligence",
    page_icon="⚡",
    layout="wide"
)

# =========================================================
# NEUMORPHIC (SOFT UI) DESIGN SYSTEM INJECTION
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #e8ecf2 !important;
        color: #2d3748 !important;
    }

    /* Neumorphic elevated surfaces */
    .neu-card {
        background: #e8ecf2;
        border-radius: 20px;
        box-shadow: 8px 8px 18px #c5cad3, -8px -8px 18px #ffffff;
        padding: 24px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.4);
    }

    .neu-card-inset {
        background: #e8ecf2;
        border-radius: 16px;
        box-shadow: inset 5px 5px 10px #c5cad3, inset -5px -5px 10px #ffffff;
        padding: 18px;
        margin-bottom: 16px;
    }

    /* Sidebar neumorphic panel */
    section[data-testid="stSidebar"] {
        background-color: #e8ecf2 !important;
        border-right: 1px solid #d4d9e2;
        box-shadow: 4px 0px 12px rgba(163, 177, 198, 0.35);
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #e8ecf2 !important;
        border-radius: 18px !important;
        box-shadow: 6px 6px 14px #c5cad3, -6px -6px 14px #ffffff !important;
        padding: 18px 22px !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }

    /* Inputs: text input, select boxes */
    .stTextInput > div > div > input, .stSelectbox > div > div {
        background: #e8ecf2 !important;
        border-radius: 14px !important;
        border: none !important;
        box-shadow: inset 4px 4px 8px #c5cad3, inset -4px -4px 8px #ffffff !important;
        color: #1a202c !important;
        padding: 12px 16px !important;
    }

    /* Neumorphic Buttons */
    .stButton > button {
        background: #e8ecf2 !important;
        color: #2b6cb0 !important;
        font-weight: 700 !important;
        border-radius: 14px !important;
        border: none !important;
        padding: 12px 28px !important;
        box-shadow: 6px 6px 12px #c5cad3, -6px -6px 12px #ffffff !important;
        transition: all 0.2s ease-in-out !important;
    }

    .stButton > button:hover {
        color: #1a365d !important;
        box-shadow: 3px 3px 6px #c5cad3, -3px -3px 6px #ffffff !important;
        transform: translateY(1px);
    }

    .stButton > button:active {
        box-shadow: inset 4px 4px 8px #c5cad3, inset -4px -4px 8px #ffffff !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: #e8ecf2 !important;
        color: #2e7d32 !important;
        font-weight: 600 !important;
        border-radius: 14px !important;
        border: none !important;
        box-shadow: 6px 6px 12px #c5cad3, -6px -6px 12px #ffffff !important;
    }

    /* Headers */
    h1, h2, h3, h4 {
        color: #1a202c !important;
        font-weight: 700 !important;
    }

    /* Badges */
    .badge-critical {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 12px;
        background: #ffdede;
        color: #c53030;
        font-weight: 700;
        font-size: 0.85rem;
        box-shadow: inset 2px 2px 5px #e2b6b6, inset -2px -2px 5px #ffffff;
    }

    .badge-warning {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 12px;
        background: #feebc8;
        color: #c05621;
        font-weight: 700;
        font-size: 0.85rem;
        box-shadow: inset 2px 2px 5px #e2cbb0, inset -2px -2px 5px #ffffff;
    }

    .badge-healthy {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 12px;
        background: #c6f6d5;
        color: #22543d;
        font-weight: 700;
        font-size: 0.85rem;
        box-shadow: inset 2px 2px 5px #a6d7b5, inset -2px -2px 5px #ffffff;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Enterprise Play Store Review Intelligence")
st.markdown("<p style='color:#718096; font-size:1.05rem; margin-top:-10px; margin-bottom:25px;'>Neumorphic Cognitive Analytics & Strategic Executive Advisory</p>", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("Navigation & Scope")
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

# Text Processing Vocabularies
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
                if len(clean_m.split()) >= 2 and clean_m not in ["this", "that", "app", "something"]:
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

        total = len(df)
        pos_c = (df["Sentiment"] == "Positive").sum()
        neg_c = (df["Sentiment"] == "Negative").sum()
        neu_c = (df["Sentiment"] == "Neutral").sum()
        nss = ((pos_c - neg_c) / total) * 100
        churn_c = df["Churn_Risk"].sum()

        # Scorecard
        st.markdown("### 📈 Executive Performance Scorecard")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Ingested Reviews", f"{total:,}")
        k2.metric("Average Rating", f"{df['score'].mean():.2f} ⭐")
        k3.metric("Net Sentiment (NSS)", f"{nss:+.1f}")
        k4.metric("Negative Ratio", f"{(neg_c/total)*100:.1f}%")
        k5.metric("Churn Threats", f"{churn_c:,}")

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

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
            ).update_layout(
                yaxis={'autorange': 'reversed'},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
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
                ).update_traces(textposition='top center').update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_ver, use_container_width=True)
            else:
                st.info("App versions not disclosed by this developer in review payloads.")

        # Keywords Row
        st.subheader("🗣️ Recurring Reviewer Keywords")
        col_p1, col_p2 = st.columns(2)
        neg_phrases = extract_key_phrases(df[df["Sentiment"] == "Negative"]["cleaned_text"])
        pos_phrases = extract_key_phrases(df[df["Sentiment"] == "Positive"]["cleaned_text"])

        with col_p1:
            st.write("**Top Complaint Drivers**")
            if neg_phrases:
                p_df = pd.DataFrame(neg_phrases, columns=["Phrase", "Count"])
                st.plotly_chart(
                    px.bar(p_df, x="Count", y="Phrase", orientation="h", color_discrete_sequence=["#e74c3c"])
                    .update_layout(yaxis={'autorange': 'reversed'}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'),
                    use_container_width=True
                )

        with col_p2:
            st.write("**Top Praise Drivers**")
            if pos_phrases:
                p_df_pos = pd.DataFrame(pos_phrases, columns=["Phrase", "Count"])
                st.plotly_chart(
                    px.bar(p_df_pos, x="Count", y="Phrase", orientation="h", color_discrete_sequence=["#2ecc71"])
                    .update_layout(yaxis={'autorange': 'reversed'}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'),
                    use_container_width=True
                )

        # Advanced Modules: Wishlist & Support Gap
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

        # Temporal Timing
        st.subheader("⏰ Incident Timing: Peak Complaint Periods")
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_summary = df[df["Sentiment"] == "Negative"]["day_of_week"].value_counts().reindex(dow_order).fillna(0).reset_index()
        dow_summary.columns = ["Day of Week", "Negative Review Spike"]
        st.plotly_chart(
            px.bar(dow_summary, x="Day of Week", y="Negative Review Spike", color_discrete_sequence=["#e67e22"])
            .update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'),
            use_container_width=True
        )

        # =========================================================================
        # HIGH-DEPTH NEUMORPHIC AUTOMATED EXECUTIVE READOUT & ACTION PLAN
        # =========================================================================
        st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
        st.markdown("## 📋 Automated Executive Readout & Strategic Action Plan")

        # Diagnosis Level
        if nss >= 35:
            badge_class = "badge-healthy"
            status_title = "HEALTHY / STRONG PRODUCT-MARKET FIT"
            status_desc = "Organic customer satisfaction is significantly outweighing friction points. The core user loop remains intact."
            risk_level = "Low (Proactive Maintenance)"
        elif nss >= 0:
            badge_class = "badge-warning"
            status_title = "MODERATE / CHURN WARNING"
            status_desc = "Noticeable degradation in user perception. Customer friction points are neutralizing positive retention drivers."
            risk_level = "Elevated (Immediate Product Intervention Needed)"
        else:
            badge_class = "badge-critical"
            status_title = "CRITICAL / SEVERE ATTRITION THREAT"
            status_desc = "Negative sentiment dominates the feedback loop. Attrition rates and negative word-of-mouth are actively degrading brand equity."
            risk_level = "High / Emergency Level"

        # Card 1: Strategic Health Matrix
        st.markdown(f"""
        <div class="neu-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3 style="margin: 0;">1. Strategic Product Health Matrix</h3>
                <span class="{badge_class}">{status_title}</span>
            </div>
            <p style="color: #4a5568; font-size: 1.05rem; line-height: 1.6;">{status_desc}</p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin-top: 20px;">
                <div class="neu-card-inset">
                    <span style="font-size: 0.8rem; color: #718096; font-weight: 600; text-transform: uppercase;">Net Sentiment Index</span>
                    <h2 style="margin: 5px 0 0 0; color: #2d3748;">{nss:+.1f} <span style="font-size:0.85rem; color:#a0aec0;">/ 100</span></h2>
                </div>
                <div class="neu-card-inset">
                    <span style="font-size: 0.8rem; color: #718096; font-weight: 600; text-transform: uppercase;">Estimated Attrition Exposure</span>
                    <h2 style="margin: 5px 0 0 0; color: #e53e3e;">{(churn_c/total)*100:.1f}% <span style="font-size:0.85rem; color:#a0aec0;">({churn_c:,} users)</span></h2>
                </div>
                <div class="neu-card-inset">
                    <span style="font-size: 0.8rem; color: #718096; font-weight: 600; text-transform: uppercase;">Negative Friction Share</span>
                    <h2 style="margin: 5px 0 0 0; color: #dd6b20;">{(neg_c/total)*100:.1f}%</h2>
                </div>
                <div class="neu-card-inset">
                    <span style="font-size: 0.8rem; color: #718096; font-weight: 600; text-transform: uppercase;">Systemic Risk Rating</span>
                    <h3 style="margin: 5px 0 0 0; color: #2d3748; font-size: 1.1rem;">{risk_level}</h3>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Dynamic Extraction for Engineering Actions
        non_gen_themes = theme_df[theme_df["Root Cause Theme"] != "General Feedback"]
        top_complaint = non_gen_themes.iloc[0]["Root Cause Theme"] if not non_gen_themes.empty else "System Stability"
        top_complaint_vol = non_gen_themes.iloc[0]["Negative Mentions"] if not non_gen_themes.empty else 0
        second_complaint = non_gen_themes.iloc[1]["Root Cause Theme"] if len(non_gen_themes) > 1 else "UI Navigation"
        second_complaint_vol = non_gen_themes.iloc[1]["Negative Mentions"] if len(non_gen_themes) > 1 else 0

        # Card 2: Engineering Sprint Backlog
        st.markdown(f"""
        <div class="neu-card">
            <h3 style="margin-top: 0;">2. Engineering & Technical Sprint Priorities</h3>
            <p style="color: #718096; margin-bottom: 20px;">Data-driven Jira/Linear sprint ticket recommendations derived from reviewer pain clusters.</p>
            
            <div class="neu-card-inset" style="border-left: 4px solid #e53e3e;">
                <strong style="color: #c53030;">[P0 - Immediate Hotfix] Mitigate {top_complaint} Friction</strong>
                <p style="margin: 6px 0 0 0; color: #4a5568;">
                    Accounting for <strong>{top_complaint_vol:,} verified complaints</strong>, this is the single highest volume issue. Engineering must immediately open an incident audit covering timeout limits, gateway fallbacks, or crash stack traces associated with this domain.
                </p>
            </div>
            
            <div class="neu-card-inset" style="border-left: 4px solid #dd6b20;">
                <strong style="color: #c05621;">[P1 - Next Sprint Scope] Optimize {second_complaint}</strong>
                <p style="margin: 6px 0 0 0; color: #4a5568;">
                    Logged <strong>{second_complaint_vol:,} friction reports</strong>. Assign QA and frontend leads to perform audit passes specifically addressing this vector in the upcoming release cycle.
                </p>
            </div>
            
            <div class="neu-card-inset" style="border-left: 4px solid #3182ce;">
                <strong style="color: #2b6cb0;">[P2 - Keyword Remediation] Search Logs for High-Frequency Trigger Terms</strong>
                <p style="margin: 6px 0 0 0; color: #4a5568;">
                    Automated text parsing detected peak recurrence for: {', '.join([f'<code>{p[0]}</code>' for p in neg_phrases[:3]]) if neg_phrases else 'general error terms'}. Cross-reference Sentry / Datadog telemetry with these strings.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Card 3: VoC Feature Roadmap
        top_wishes_html = ""
        if wishes:
            for idx, (wish, freq) in enumerate(wishes[:5], start=1):
                top_wishes_html += f"""
                <tr>
                    <td style="padding: 10px 12px; border-bottom: 1px solid #d4d9e2;"><strong>#{idx}</strong></td>
                    <td style="padding: 10px 12px; border-bottom: 1px solid #d4d9e2;"><code>"{wish}"</code></td>
                    <td style="padding: 10px 12px; border-bottom: 1px solid #d4d9e2; text-align: center;"><span style="background:#edf2f7; padding:4px 10px; border-radius:8px; font-weight:600;">{freq} requests</span></td>
                    <td style="padding: 10px 12px; border-bottom: 1px solid #d4d9e2; color: #2e7d32; font-weight: 600;">{'🔥 High Priority' if freq >= 5 else '⚡ Roadmap Candidate'}</td>
                </tr>
                """
        else:
            top_wishes_html = "<tr><td colspan='4' style='padding:12px; text-align:center;'>No high-confidence structured wishlist items extracted.</td></tr>"

        st.markdown(f"""
        <div class="neu-card">
            <h3 style="margin-top: 0;">3. Voice of the Customer (VoC): Feature Demand Roadmap</h3>
            <p style="color: #718096; margin-bottom: 15px;">Mined feature demand patterns expressing explicit user appetite.</p>
            <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
                <thead>
                    <tr style="text-align: left; color: #718096; font-size: 0.85rem; text-transform: uppercase;">
                        <th style="padding: 8px 12px; border-bottom: 2px solid #cbd5e0;">Rank</th>
                        <th style="padding: 8px 12px; border-bottom: 2px solid #cbd5e0;">Feature Request Expression</th>
                        <th style="padding: 8px 12px; border-bottom: 2px solid #cbd5e0; text-align: center;">Frequency</th>
                        <th style="padding: 8px 12px; border-bottom: 2px solid #cbd5e0;">Strategy Recommendation</th>
                    </tr>
                </thead>
                <tbody>
                    {top_wishes_html}
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # Card 4: Retention & Support SLA Protocol
        st.markdown(f"""
        <div class="neu-card">
            <h3 style="margin-top: 0;">4. Churn Interception & Support Operations Audit</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div class="neu-card-inset">
                    <strong style="color: #c53030;">🚨 Churn Interception Playbook</strong>
                    <p style="color: #4a5568; font-size: 0.95rem; margin-top: 8px; line-height: 1.5;">
                        Found <strong>{churn_c:,} reviews containing explicit departure intent</strong> (e.g., "uninstalled", "moving to competitor").
                    </p>
                    <ul style="color: #4a5568; font-size: 0.9rem; padding-left: 20px; margin-top: 5px;">
                        <li>Deploy automated Google Play Console reply templates to 1-star reviews offering resolution paths.</li>
                        <li>Verify if recent monetization or subscription wall updates altered the free-tier experience.</li>
                    </ul>
                </div>
                <div class="neu-card-inset">
                    <strong style="color: #2b6cb0;">🎧 Support SLA Performance</strong>
                    <p style="color: #4a5568; font-size: 0.95rem; margin-top: 8px; line-height: 1.5;">
                        Support response coverage on 1-2 star reviews stands at <strong>{crit_rate:.1f}%</strong>.
                    </p>
                    <ul style="color: #4a5568; font-size: 0.9rem; padding-left: 20px; margin-top: 5px;">
                        <li>Target an industry benchmark of <strong>&gt;40% reply coverage</strong> on critical ratings.</li>
                        <li>Maintain resolution response latency within <strong>&lt;48 hours</strong> to maximize re-rating potential.</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Export Button
        st.download_button(
            "📥 Export Full Dataset (CSV)",
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

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.subheader("Theme Comparison: Complaint Breakdown")
        comp_exploded = combined.explode("Themes")
        comp_neg = comp_exploded[comp_exploded["Sentiment"] == "Negative"].groupby(["Themes", "App"]).size().reset_index(name="Negative Mentions")

        st.plotly_chart(
            px.bar(comp_neg, x="Themes", y="Negative Mentions", color="App", barmode="group", color_discrete_sequence=["#3182ce", "#e53e3e"])
            .update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'),
            use_container_width=True
        )
