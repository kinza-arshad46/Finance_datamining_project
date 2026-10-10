import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score,
    roc_auc_score, roc_curve
)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Finance Credit Risk | Data Mining",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS (Premium Finance Theme) ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(160deg, #0a0e17 0%, #0f172a 40%, #1e1b4b 100%);
        font-family: 'Inter', sans-serif;
        color: #e2e8f0;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }
    
    [data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    
    /* Premium Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, rgba(30,41,59,0.9), rgba(15,23,42,0.95));
        border-radius: 18px;
        padding: 22px 18px;
        border: 1px solid rgba(56,189,248,0.25);
        box-shadow: 0 10px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05);
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 12px;
    }
    .metric-card:hover {
        border-color: #38bdf8;
        transform: translateY(-3px);
        box-shadow: 0 15px 45px rgba(56,189,248,0.15);
    }
    .metric-card .label {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }
    .metric-card .value {
        color: #38bdf8;
        font-size: 26px;
        font-weight: 700;
        margin: 0;
    }
    .metric-card .value.danger { color: #f43f5e; }
    .metric-card .value.success { color: #34d399; }
    .metric-card .value.warning { color: #fbbf24; }
    
    /* Login Card */
    .login-box {
        max-width: 440px;
        margin: 60px auto;
        padding: 48px 40px;
        background: linear-gradient(145deg, rgba(30,41,59,0.95), rgba(15,23,42,0.98));
        border-radius: 24px;
        border: 1px solid rgba(56,189,248,0.3);
        box-shadow: 0 25px 80px rgba(0,0,0,0.5), 0 0 60px rgba(56,189,248,0.08);
        text-align: center;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 20px rgba(14,165,233,0.35) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(14,165,233,0.5) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(30,41,59,0.7);
        border-radius: 10px 10px 0 0;
        color: #94a3b8;
        padding: 10px 22px;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
        color: white !important;
        border: none !important;
    }
    
    /* Inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        background: rgba(15,23,42,0.8) !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }
    
    /* Info / Alert boxes */
    .custom-info {
        background: rgba(14,165,233,0.1);
        border-left: 4px solid #0ea5e9;
        padding: 16px 20px;
        border-radius: 0 12px 12px 0;
        margin: 16px 0;
        color: #bae6fd;
    }
    .custom-success {
        background: rgba(52,211,153,0.1);
        border-left: 4px solid #34d399;
        padding: 16px 20px;
        border-radius: 0 12px 12px 0;
        margin: 16px 0;
    }
    .custom-danger {
        background: rgba(244,63,94,0.1);
        border-left: 4px solid #f43f5e;
        padding: 16px 20px;
        border-radius: 0 12px 12px 0;
        margin: 16px 0;
    }
    
    /* Section header */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
    }
    
    /* Footer */
    .app-footer {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        margin-top: 50px;
        padding: 24px;
        border-top: 1px solid #1e293b;
    }
    
    /* Hide default streamlit branding a bit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ==================== CONSTANTS ====================
VALID_USER = "admin"
VALID_PASS = "finance123"

CAT_COLS = ["education", "home_ownership", "loan_purpose"]
NUM_COLS_RAW = ["age", "annual_income", "credit_score", "loan_amount",
                "loan_term_months", "employment_years", "debt_to_income_ratio",
                "num_open_accounts", "num_late_payments"]
TARGET = "loan_default"

# ==================== SESSION STATE ====================
defaults = {
    "logged_in": False,
    "df_raw": None,
    "df_processed": None,
    "model": None,
    "scaler": None,
    "label_encoders": {},
    "feature_names": None,
    "X_test": None,
    "y_test": None,
    "y_pred": None,
    "y_proba": None,
    "trained_model_name": None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==================== DATA FUNCTIONS ====================
@st.cache_data
def load_raw_data():
    try:
        return pd.read_csv("data/loan_default.csv")
    except Exception:
        # Fallback
        np.random.seed(42)
        n = 800
        df = pd.DataFrame({
            "age": np.random.randint(21, 70, n),
            "annual_income": np.round(np.random.lognormal(10.8, 0.55, n), -2),
            "credit_score": np.clip(np.random.normal(680, 80, n).astype(int), 300, 850),
            "loan_amount": np.round(np.random.uniform(5000, 80000, n), -2),
            "loan_term_months": np.random.choice([12, 24, 36, 48, 60], n),
            "employment_years": np.random.randint(0, 35, n),
            "debt_to_income_ratio": np.round(np.random.uniform(0.05, 0.65, n), 3),
            "num_open_accounts": np.random.randint(1, 15, n),
            "num_late_payments": np.random.poisson(1.2, n),
            "education": np.random.choice(["High School", "Bachelor", "Master", "PhD"], n),
            "home_ownership": np.random.choice(["Rent", "Own", "Mortgage"], n),
            "loan_purpose": np.random.choice(["Personal", "Home", "Auto", "Education", "Business"], n),
            "loan_default": np.random.randint(0, 2, n)
        })
        return df

def feature_engineering(df):
    """Create powerful new features for finance domain"""
    df = df.copy()
    
    # Ratios
    df["income_to_loan_ratio"] = df["annual_income"] / (df["loan_amount"] + 1)
    df["loan_to_income_ratio"] = df["loan_amount"] / (df["annual_income"] + 1)
    df["credit_utilization_proxy"] = df["num_open_accounts"] / (df["credit_score"] / 100 + 1)
    
    # Risk indicators
    df["high_dti"] = (df["debt_to_income_ratio"] > 0.4).astype(int)
    df["low_credit"] = (df["credit_score"] < 620).astype(int)
    df["young_borrower"] = (df["age"] < 28).astype(int)
    df["unstable_job"] = (df["employment_years"] < 2).astype(int)
    df["multiple_lates"] = (df["num_late_payments"] >= 3).astype(int)
    
    # Composite risk score (domain knowledge)
    df["engineered_risk_score"] = (
        df["low_credit"] * 2.0 +
        df["high_dti"] * 1.8 +
        df["multiple_lates"] * 2.2 +
        df["unstable_job"] * 1.3 +
        df["young_borrower"] * 0.7 +
        (df["loan_to_income_ratio"] > 0.5).astype(int) * 1.5
    )
    
    # Log transforms for skewed financial vars
    df["log_income"] = np.log1p(df["annual_income"])
    df["log_loan"] = np.log1p(df["loan_amount"])
    
    # Age groups
    df["age_group"] = pd.cut(df["age"], bins=[0, 30, 45, 60, 100],
                             labels=["Young", "Mid", "Senior", "Elder"])
    
    return df

def full_preprocess(df, fit_encoders=True):
    """Complete pipeline: impute → FE → encode → return clean DF + feature list"""
    df = df.copy()
    
    # 1. Impute numeric
    num_impute_cols = [c for c in NUM_COLS_RAW if c in df.columns]
    imputer = SimpleImputer(strategy="median")
    df[num_impute_cols] = imputer.fit_transform(df[num_impute_cols])
    
    # 2. Impute categoricals with mode
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # 3. Feature Engineering
    df = feature_engineering(df)
    
    # 4. Encode categoricals
    encoders = {}
    for col in CAT_COLS + ["age_group"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    # 5. Outlier capping (IQR) on key numeric
    cap_cols = ["annual_income", "loan_amount", "debt_to_income_ratio",
                "credit_score", "num_late_payments"]
    for col in cap_cols:
        if col in df.columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    
    # Final feature list (exclude target and intermediate)
    exclude = [TARGET, "age_group"]  # age_group already encoded but we keep engineered ones
    feature_cols = [c for c in df.columns if c not in exclude]
    
    return df, feature_cols, encoders

def train_models(df, feature_cols, model_name):
    X = df[feature_cols]
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    if model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            random_state=42, n_jobs=-1, class_weight="balanced"
        )
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            random_state=42
        )
    elif model_name == "Hist Gradient Boosting":
        model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=8, learning_rate=0.08,
            random_state=42
        )
    else:
        model = LogisticRegression(
            max_iter=1500, C=0.8, class_weight="balanced", random_state=42
        )
    
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    
    return model, scaler, X_test_s, y_test, y_pred, y_proba, feature_cols

# ==================== LOGIN ====================
def login_page():
    st.markdown("""
    <div class="login-box">
        <div style="font-size:48px; margin-bottom:8px;">💳</div>
        <h1 style="margin:0; font-size:28px; background:linear-gradient(90deg,#38bdf8,#818cf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            Credit Risk Analytics
        </h1>
        <p style="color:#94a3b8; margin-top:8px; font-size:14px;">
            Finance Data Mining Platform • Assignment 2
        </p>
        <hr style="border:none; height:1px; background:linear-gradient(90deg,transparent,#334155,transparent); margin:28px 0;">
    </div>
    """, unsafe_allow_html=True)
    
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        with st.form("login_form", clear_on_submit=False):
            st.markdown("##### 🔐 Secure Login")
            user = st.text_input("Username", placeholder="Enter username")
            pwd = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Sign In →", use_container_width=True)
            
            if submitted:
                if user == VALID_USER and pwd == VALID_PASS:
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
        
        st.markdown("""
        <div class="custom-info" style="margin-top:24px; font-size:13px;">
            <b>Demo Access</b><br>
            Username: <code style="color:#38bdf8;">admin</code><br>
            Password: <code style="color:#38bdf8;">finance123</code>
        </div>
        """, unsafe_allow_html=True)

# ==================== SIDEBAR ====================
def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:16px 0 8px;">
            <div style="font-size:32px;">💳</div>
            <h3 style="margin:4px 0; color:#38bdf8 !important;">CreditRisk AI</h3>
            <p style="font-size:11px; color:#64748b !important;">Finance Data Mining</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        page = st.radio(
            "Navigate",
            [
                "🏠 Executive Dashboard",
                "📂 Data & Feature Engineering",
                "📊 Advanced EDA + 3D Visuals",
                "🧠 Model Training & Mining",
                "🔮 Live Risk Prediction",
                "📄 Documentation"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align:center; font-size:12px; color:#64748b;">
            Signed in as <span style="color:#38bdf8; font-weight:600;">admin</span>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Logout", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        
        st.markdown("""
        <div class="app-footer">
            Data Mining Lab • Assignment #2<br>
            CLO-2 & CLO-3 • Finance Domain
        </div>
        """, unsafe_allow_html=True)
    
    return page

# ==================== DASHBOARD ====================
def page_dashboard(df):
    st.markdown("## 🏠 Executive Dashboard")
    st.caption("Real-time overview of the credit portfolio and default risk landscape")
    
    total = len(df)
    defaults = int(df[TARGET].sum())
    default_rate = defaults / total * 100
    avg_income = df["annual_income"].median()
    avg_score = df["credit_score"].median()
    
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "Total Applications", f"{total:,}", ""),
        (c2, "Defaults", f"{defaults:,}", "danger"),
        (c3, "Default Rate", f"{default_rate:.1f}%", "warning"),
        (c4, "Median Income", f"${avg_income:,.0f}", "success"),
        (c5, "Median Credit Score", f"{avg_score:.0f}", ""),
    ]
    for col, label, value, cls in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value {cls}">{value}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Default Distribution")
        fig = px.pie(
            values=[total - defaults, defaults],
            names=["No Default", "Default"],
            color_discrete_sequence=["#34d399", "#f43f5e"],
            hole=0.55
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            height=340,
            margin=dict(t=30, b=20, l=20, r=20),
            showlegend=True,
            legend=dict(orientation="h", y=-0.1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Credit Score vs Default")
        fig2 = px.histogram(
            df, x="credit_score", color=TARGET,
            barmode="overlay", nbins=30, opacity=0.75,
            color_discrete_map={0: "#34d399", 1: "#f43f5e"},
            labels={TARGET: "Default"}
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            height=340,
            margin=dict(t=30, b=20, l=20, r=20),
            legend_title_text=""
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # 3D teaser
    st.markdown("#### 3D Risk Landscape (Age × Income × Credit Score)")
    sample = df.sample(min(400, len(df)), random_state=42)
    fig3d = px.scatter_3d(
        sample, x="age", y="annual_income", z="credit_score",
        color=TARGET,
        color_discrete_map={0: "#34d399", 1: "#f43f5e"},
        opacity=0.7,
        size_max=8,
        labels={TARGET: "Default Status"}
    )
    fig3d.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        height=520,
        margin=dict(t=20, b=0, l=0, r=0),
        scene=dict(
            xaxis_title="Age",
            yaxis_title="Annual Income",
            zaxis_title="Credit Score",
            bgcolor="rgba(15,23,42,0.6)",
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#334155"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#334155"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#334155"),
        )
    )
    st.plotly_chart(fig3d, use_container_width=True)
    
    st.markdown("""
    <div class="custom-info">
        <b>Domain:</b> Finance / Credit Risk &nbsp;•&nbsp;
        <b>Task:</b> Binary Classification (Loan Default Prediction) &nbsp;•&nbsp;
        <b>Pipeline:</b> Load → Clean → Feature Engineering → EDA → Model → Predict
    </div>
    """, unsafe_allow_html=True)

# ==================== DATA + FEATURE ENGINEERING ====================
def page_data_fe(df_raw):
    st.markdown("## 📂 Data Loading, Preprocessing & Feature Engineering")
    
    tabs = st.tabs([
        "📋 Raw Dataset", "🔍 Missing & Outliers", "⚙️ Feature Engineering", "✨ Final Features"
    ])
    
    with tabs[0]:
        st.markdown("#### Original Loan Application Dataset")
        st.dataframe(df_raw.head(20), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", len(df_raw))
        c2.metric("Columns", df_raw.shape[1])
        c3.metric("Default Rate", f"{df_raw[TARGET].mean()*100:.1f}%")
        
        st.markdown("#### Statistical Summary")
        st.dataframe(df_raw.describe(include="all").T, use_container_width=True)
    
    with tabs[1]:
        st.markdown("#### Missing Value Profile")
        miss = df_raw.isnull().sum()
        miss = miss[miss > 0]
        if len(miss) > 0:
            miss_df = pd.DataFrame({
                "Feature": miss.index,
                "Missing Count": miss.values,
                "Missing %": (miss.values / len(df_raw) * 100).round(2)
            })
            fig = px.bar(miss_df, x="Feature", y="Missing Count",
                         color="Missing %", color_continuous_scale="Teal",
                         text="Missing %")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", height=380
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(miss_df, use_container_width=True)
        else:
            st.success("No missing values detected.")
        
        st.markdown("#### Outlier Inspection (Boxplots)")
        num_show = st.multiselect(
            "Select numeric features",
            ["annual_income", "credit_score", "loan_amount", "debt_to_income_ratio", "num_late_payments"],
            default=["annual_income", "credit_score", "loan_amount"]
        )
        if num_show:
            fig = make_subplots(rows=1, cols=len(num_show), subplot_titles=num_show)
            for i, col in enumerate(num_show):
                fig.add_trace(
                    go.Box(y=df_raw[col].dropna(), name=col,
                           marker_color="#38bdf8", boxmean=True),
                    row=1, col=i+1
                )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", height=380, showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="custom-info">
            <b>Strategy:</b> Median imputation for numeric • Mode for categorical • IQR capping for outliers
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("#### Feature Engineering Pipeline")
        st.markdown("""
        New features created using domain knowledge in credit risk:
        
        | New Feature | Formula / Logic | Why it helps |
        |-------------|-----------------|--------------|
        | `income_to_loan_ratio` | income / loan_amount | Capacity to repay |
        | `loan_to_income_ratio` | loan / income | Burden indicator |
        | `credit_utilization_proxy` | open_accounts / (credit_score/100) | Credit stress |
        | `high_dti` | debt_to_income > 0.4 | Binary risk flag |
        | `low_credit` | credit_score < 620 | Subprime flag |
        | `multiple_lates` | late_payments ≥ 3 | Payment behavior |
        | `unstable_job` | employment_years < 2 | Job stability |
        | `engineered_risk_score` | Weighted combination of flags | Composite risk |
        | `log_income`, `log_loan` | log1p transforms | Reduce skewness |
        """)
        
        if st.button("🚀 Run Full Preprocessing + Feature Engineering", type="primary"):
            with st.spinner("Running pipeline..."):
                df_proc, feats, encoders = full_preprocess(df_raw)
                st.session_state.df_processed = df_proc
                st.session_state.feature_names = feats
                st.session_state.label_encoders = encoders
                st.success(f"✅ Done! {len(feats)} features ready for modeling.")
    
    with tabs[3]:
        if st.session_state.df_processed is not None:
            dfp = st.session_state.df_processed
            st.markdown("#### Processed Dataset Preview")
            st.dataframe(dfp.head(15), use_container_width=True)
            
            st.markdown("#### Newly Engineered Features (sample)")
            eng_cols = [c for c in dfp.columns if c not in list(df_raw.columns) + [TARGET]]
            st.dataframe(dfp[eng_cols].head(10), use_container_width=True)
            
            c1, c2 = st.columns(2)
            c1.metric("Final Feature Count", len(st.session_state.feature_names))
            c2.metric("Missing after clean", int(dfp.isnull().sum().sum()))
            
            st.markdown("#### Correlation of Engineered Risk Score with Target")
            if "engineered_risk_score" in dfp.columns:
                corr_val = dfp["engineered_risk_score"].corr(dfp[TARGET])
                st.metric("Correlation with Default", f"{corr_val:.3f}")
        else:
            st.warning("Please run Feature Engineering first (previous tab).")

# ==================== ADVANCED EDA + 3D ====================
def page_eda(df):
    st.markdown("## 📊 Advanced Exploratory Data Analysis + 3D Visuals")
    
    if df is None:
        st.warning("Run Feature Engineering first for best results. Using raw data with basic cleaning.")
        df, _, _ = full_preprocess(st.session_state.df_raw)
    
    viz = st.selectbox(
        "Choose Visualization",
        [
            "Histogram",
            "Boxplot",
            "Scatterplot (2D)",
            "3D Scatter Plot",
            "Correlation Heatmap",
            "3D Surface / Density"
        ]
    )
    
    st.markdown("---")
    
    numeric_feats = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET]
    
    if viz == "Histogram":
        col = st.selectbox("Feature", numeric_feats, index=numeric_feats.index("credit_score") if "credit_score" in numeric_feats else 0)
        bins = st.slider("Bins", 15, 60, 30)
        fig = px.histogram(
            df, x=col, color=TARGET, nbins=bins, barmode="overlay", opacity=0.75,
            color_discrete_map={0: "#34d399", 1: "#f43f5e"},
            title=f"Distribution of {col}"
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", height=480)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz == "Boxplot":
        col = st.selectbox("Feature", numeric_feats)
        fig = px.box(
            df, x=TARGET, y=col, color=TARGET,
            color_discrete_map={0: "#34d399", 1: "#f43f5e"},
            title=f"{col} by Default Status", points="outliers"
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", height=480)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz == "Scatterplot (2D)":
        c1, c2 = st.columns(2)
        with c1:
            x = st.selectbox("X-axis", numeric_feats, index=0)
        with c2:
            y = st.selectbox("Y-axis", numeric_feats, index=min(1, len(numeric_feats)-1))
        fig = px.scatter(
            df.sample(min(500, len(df))), x=x, y=y, color=TARGET,
            color_discrete_map={0: "#34d399", 1: "#f43f5e"},
            opacity=0.65, title=f"{x} vs {y}",
            trendline="ols" if st.checkbox("Show trendline") else None
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", height=520)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz == "3D Scatter Plot":
        st.markdown("#### Interactive 3D Scatter")
        c1, c2, c3 = st.columns(3)
        with c1:
            x = st.selectbox("X", numeric_feats, index=0, key="3dx")
        with c2:
            y = st.selectbox("Y", numeric_feats, index=1, key="3dy")
        with c3:
            z = st.selectbox("Z", numeric_feats, index=2, key="3dz")
        
        sample = df.sample(min(450, len(df)), random_state=42)
        fig = px.scatter_3d(
            sample, x=x, y=y, z=z, color=TARGET,
            color_discrete_map={0: "#34d399", 1: "#f43f5e"},
            opacity=0.75, size_max=9,
            title=f"3D: {x} × {y} × {z}"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            height=600,
            scene=dict(
                bgcolor="rgba(15,23,42,0.7)",
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#334155", title=x),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#334155", title=y),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#334155", title=z),
            ),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Drag to rotate • Scroll to zoom • Hover for details")
    
    elif viz == "Correlation Heatmap":
        st.markdown("#### Feature Correlation Matrix")
        cols_for_corr = [c for c in numeric_feats + [TARGET] if c in df.columns][:18]
        corr = df[cols_for_corr].corr()
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap"
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=650
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Top Absolute Correlations with Default")
        target_corr = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False).head(12)
        fig2 = px.bar(
            x=target_corr.values, y=target_corr.index, orientation="h",
            color=target_corr.values, color_continuous_scale="Teal",
            labels={"x": "|Correlation|", "y": "Feature"}
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", height=420, yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    elif viz == "3D Surface / Density":
        st.markdown("#### 3D Density / Surface Style (Credit Score × DTI × Default Density)")
        # Create a grid density
        from scipy.stats import gaussian_kde
        try:
            sample = df.dropna(subset=["credit_score", "debt_to_income_ratio"]).sample(min(600, len(df)))
            x = sample["credit_score"]
            y = sample["debt_to_income_ratio"]
            z = sample[TARGET]
            
            fig = go.Figure(data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                marker=dict(
                    size=4,
                    color=z,
                    colorscale=[[0, "#34d399"], [1, "#f43f5e"]],
                    opacity=0.7
                )
            )])
            fig.update_layout(
                title="Credit Score × Debt-to-Income × Default",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                height=580,
                scene=dict(
                    xaxis_title="Credit Score",
                    yaxis_title="Debt-to-Income",
                    zaxis_title="Default (0/1)",
                    bgcolor="rgba(15,23,42,0.7)",
                    xaxis=dict(gridcolor="#334155"),
                    yaxis=dict(gridcolor="#334155"),
                    zaxis=dict(gridcolor="#334155"),
                ),
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info("3D density view requires scipy. Showing alternative 3D scatter.")
            fig = px.scatter_3d(
                df.sample(400), x="credit_score", y="debt_to_income_ratio",
                z="annual_income", color=TARGET,
                color_discrete_map={0: "#34d399", 1: "#f43f5e"}
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=560)
            st.plotly_chart(fig, use_container_width=True)

# ==================== MODEL TRAINING ====================
def page_mining(df):
    st.markdown("## 🧠 Model Training & Data Mining")
    
    if df is None or st.session_state.feature_names is None:
        st.warning("Please complete Feature Engineering first.")
        if st.button("Auto-run Feature Engineering"):
            df_proc, feats, enc = full_preprocess(st.session_state.df_raw)
            st.session_state.df_processed = df_proc
            st.session_state.feature_names = feats
            st.session_state.label_encoders = enc
            st.rerun()
        return
    
    technique = st.radio(
        "Technique",
        ["Supervised Classification", "Unsupervised Clustering"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if technique == "Supervised Classification":
        model_name = st.selectbox(
            "Select Algorithm",
            ["Random Forest", "Gradient Boosting", "Hist Gradient Boosting", "Logistic Regression"]
        )
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner(f"Training {model_name} with engineered features..."):
                model, scaler, X_test, y_test, y_pred, y_proba, feats = train_models(
                    df, st.session_state.feature_names, model_name
                )
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred
                st.session_state.y_proba = y_proba
                st.session_state.trained_model_name = model_name
                st.success(f"✅ {model_name} trained successfully!")
        
        if st.session_state.model is not None:
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred
            y_proba = st.session_state.y_proba
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            
            st.markdown(f"#### Performance — {st.session_state.trained_model_name}")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", f"{acc:.3f}")
            m2.metric("Precision", f"{prec:.3f}")
            m3.metric("Recall", f"{rec:.3f}")
            m4.metric("F1-Score", f"{f1:.3f}")
            m5.metric("ROC-AUC", f"{auc:.3f}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(
                    cm, text_auto=True,
                    labels=dict(x="Predicted", y="Actual"),
                    x=["No Default", "Default"],
                    y=["No Default", "Default"],
                    color_continuous_scale="Blues",
                    aspect="auto"
                )
                fig_cm.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=380
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with col2:
                st.markdown("#### ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines", name=f"AUC = {auc:.3f}",
                    line=dict(color="#38bdf8", width=3)
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines", name="Random",
                    line=dict(color="#64748b", dash="dash")
                ))
                fig_roc.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0", height=380,
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    legend=dict(y=0.1)
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            
            # Feature Importance
            if hasattr(st.session_state.model, "feature_importances_"):
                st.markdown("#### Feature Importance (Top 15)")
                imp = pd.DataFrame({
                    "Feature": st.session_state.feature_names,
                    "Importance": st.session_state.model.feature_importances_
                }).sort_values("Importance", ascending=True).tail(15)
                
                fig_imp = px.bar(
                    imp, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale="Tealgrn"
                )
                fig_imp.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0", height=480
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            
            st.markdown("#### Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}"), use_container_width=True)
            
            st.markdown("""
            <div class="custom-success">
                Model is ready. Go to <b>Live Risk Prediction</b> to score new applicants.
            </div>
            """, unsafe_allow_html=True)
    
    else:  # Clustering
        st.markdown("### K-Means Customer Segmentation")
        k = st.slider("Number of Clusters", 2, 6, 3)
        
        if st.button("Run Clustering", type="primary"):
            with st.spinner("Clustering..."):
                X = df[st.session_state.feature_names]
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                km = KMeans(n_clusters=k, random_state=42, n_init=15)
                labels = km.fit_predict(Xs)
                sil = silhouette_score(Xs, labels)
                
                df_c = df.copy()
                df_c["Cluster"] = labels
                
                st.success(f"Silhouette Score: **{sil:.3f}**")
                
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.pie(
                        values=pd.Series(labels).value_counts().sort_index(),
                        names=[f"Cluster {i}" for i in range(k)],
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        title="Cluster Sizes"
                    )
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=380)
                    st.plotly_chart(fig, use_container_width=True)
                
                with c2:
                    fig2 = px.scatter_3d(
                        df_c.sample(min(400, len(df_c))),
                        x="credit_score", y="annual_income", z="debt_to_income_ratio",
                        color="Cluster", opacity=0.75,
                        title="3D Cluster View"
                    )
                    fig2.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=420,
                        scene=dict(bgcolor="rgba(15,23,42,0.6)")
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                
                st.markdown("#### Cluster vs Actual Default")
                st.dataframe(pd.crosstab(df_c["Cluster"], df_c[TARGET], margins=True), use_container_width=True)

# ==================== PREDICTION ====================
def page_predict():
    st.markdown("## 🔮 Live Credit Risk Prediction")
    
    if st.session_state.model is None:
        st.warning("Train a model first in the **Model Training** page.")
        return
    
    st.markdown(f"Active model: **{st.session_state.trained_model_name}**")
    st.markdown("Enter applicant details to get default risk probability.")
    
    with st.form("predict_form"):
        st.markdown("### Applicant Profile")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            age = st.number_input("Age", 18, 80, 35)
            annual_income = st.number_input("Annual Income ($)", 10000, 500000, 55000, step=1000)
            credit_score = st.number_input("Credit Score", 300, 850, 680)
            loan_amount = st.number_input("Loan Amount ($)", 1000, 150000, 25000, step=500)
        
        with c2:
            loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60], index=2)
            employment_years = st.number_input("Employment Years", 0, 40, 5)
            debt_to_income = st.slider("Debt-to-Income Ratio", 0.05, 0.80, 0.32, 0.01)
            num_open = st.number_input("Open Credit Accounts", 0, 20, 4)
        
        with c3:
            num_lates = st.number_input("Late Payments (last 2 yrs)", 0, 15, 1)
            education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
            home = st.selectbox("Home Ownership", ["Rent", "Own", "Mortgage"])
            purpose = st.selectbox("Loan Purpose", ["Personal", "Home", "Auto", "Education", "Business"])
        
        submitted = st.form_submit_button("⚡ Calculate Default Risk", type="primary", use_container_width=True)
        
        if submitted:
            # Build single-row dataframe matching training pipeline
            row = pd.DataFrame([{
                "age": age,
                "annual_income": annual_income,
                "credit_score": credit_score,
                "loan_amount": loan_amount,
                "loan_term_months": loan_term,
                "employment_years": employment_years,
                "debt_to_income_ratio": debt_to_income,
                "num_open_accounts": num_open,
                "num_late_payments": num_lates,
                "education": education,
                "home_ownership": home,
                "loan_purpose": purpose,
                "loan_default": 0  # placeholder
            }])
            
            # Apply same FE
            row_fe = feature_engineering(row)
            
            # Encode using saved encoders
            for col, le in st.session_state.label_encoders.items():
                if col in row_fe.columns:
                    try:
                        row_fe[col] = le.transform(row_fe[col].astype(str))
                    except Exception:
                        row_fe[col] = 0
            
            # Align columns
            feats = st.session_state.feature_names
            for f in feats:
                if f not in row_fe.columns:
                    row_fe[f] = 0
            X_new = row_fe[feats]
            
            X_scaled = st.session_state.scaler.transform(X_new)
            pred = st.session_state.model.predict(X_scaled)[0]
            proba = st.session_state.model.predict_proba(X_scaled)[0]
            
            st.markdown("---")
            st.markdown("### Risk Assessment Result")
            
            risk_pct = proba[1] * 100
            
            if pred == 1:
                st.markdown(f"""
                <div class="custom-danger">
                    <h3 style="margin:0; color:#f43f5e;">⚠️ HIGH RISK — Likely to Default</h3>
                    <p style="font-size:28px; font-weight:700; margin:8px 0; color:#f43f5e;">{risk_pct:.1f}% probability of default</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="custom-success">
                    <h3 style="margin:0; color:#34d399;">✅ LOW RISK — Likely to Repay</h3>
                    <p style="font-size:28px; font-weight:700; margin:8px 0; color:#34d399;">{proba[0]*100:.1f}% probability of no default</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability bars
            fig = go.Figure(go.Bar(
                x=["No Default", "Default"],
                y=[proba[0]*100, proba[1]*100],
                marker_color=["#34d399", "#f43f5e"],
                text=[f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"],
                textposition="auto",
                textfont=dict(size=16, color="white")
            ))
            fig.update_layout(
                title="Default Probability Breakdown",
                yaxis_title="Probability (%)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", height=340,
                yaxis=dict(range=[0, 100])
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk factors
            st.markdown("#### Key Risk Drivers for this Applicant")
            factors = []
            if credit_score < 620:
                factors.append("Low credit score (< 620)")
            if debt_to_income > 0.4:
                factors.append("High debt-to-income ratio (> 40%)")
            if num_lates >= 3:
                factors.append("Multiple late payments")
            if employment_years < 2:
                factors.append("Short employment history")
            if loan_amount / (annual_income + 1) > 0.5:
                factors.append("High loan-to-income ratio")
            if age < 28:
                factors.append("Young borrower profile")
            
            if factors:
                for f in factors:
                    st.markdown(f"- 🔴 {f}")
            else:
                st.markdown("- 🟢 No major high-risk flags detected")
            
            st.caption("Educational model only — not for real lending decisions.")

# ==================== DOCUMENTATION ====================
def page_docs():
    st.markdown("## 📄 Documentation & Analysis")
    
    st.markdown("""
    ### Finance Data Mining Application — Credit Risk Prediction
    
    **Assignment #2 • Data Mining Lab • CLO-2 & CLO-3**
    
    ---
    
    #### 1. Domain & Problem
    - **Domain:** Finance / Banking (Credit Risk)
    - **Problem:** Predict whether a loan applicant will **default** (binary classification)
    - **Business Value:** Helps lenders reduce non-performing loans and price risk correctly
    
    #### 2. Dataset
    - 800 synthetic but realistic loan applications
    - Features: demographics, income, credit score, loan details, payment history, education, housing
    - Target: `loan_default` (0 = repaid, 1 = defaulted)
    - Missing values intentionally introduced (~4%) on key columns
    
    #### 3. Preprocessing Pipeline
    1. Median imputation (numeric) + Mode imputation (categorical)
    2. IQR-based outlier capping
    3. Label encoding for categorical variables
    4. StandardScaler before modeling
    
    #### 4. Feature Engineering (Key Strength)
    Domain-driven features created:
    - `income_to_loan_ratio`, `loan_to_income_ratio`
    - `credit_utilization_proxy`
    - Binary risk flags: `high_dti`, `low_credit`, `multiple_lates`, `unstable_job`, `young_borrower`
    - `engineered_risk_score` (weighted composite)
    - Log transforms for skewed financial variables
    
    These features significantly improve model discrimination.
    
    #### 5. EDA Capabilities
    - Interactive Histogram, Boxplot, 2D Scatter
    - **3D Scatter plots** (fully rotatable)
    - Correlation heatmap + ranked feature correlations
    - 3D density-style views
    
    #### 6. Modeling
    - **Algorithms:** Random Forest, Gradient Boosting, HistGradientBoosting, Logistic Regression
    - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
    - Confusion Matrix + ROC Curve
    - Feature Importance ranking
    - Optional K-Means clustering for customer segmentation
    
    #### 7. Prediction Interface
    User enters 12 applicant attributes → model returns:
    - Binary decision (High / Low risk)
    - Probability scores
    - List of triggered risk factors
    
    #### 8. Observations
    - Engineered risk flags and ratios are among the top important features
    - Tree-based models (RF / GB) generally outperform linear models on this data
    - Credit score, late payments, and DTI remain strong traditional predictors
    - 3D visualizations help reveal non-linear separation between default and non-default groups
    
    ---
    
    **Login:** `admin` / `finance123`  
    **Disclaimer:** Academic / educational use only. Not for production lending decisions.
    """)

# ==================== MAIN ====================
def main():
    if not st.session_state.logged_in:
        login_page()
        return
    
    if st.session_state.df_raw is None:
        st.session_state.df_raw = load_raw_data()
    
    page = sidebar()
    
    df_raw = st.session_state.df_raw
    df_proc = st.session_state.df_processed
    
    if page == "🏠 Executive Dashboard":
        page_dashboard(df_raw)
    elif page == "📂 Data & Feature Engineering":
        page_data_fe(df_raw)
    elif page == "📊 Advanced EDA + 3D Visuals":
        page_eda(df_proc)
    elif page == "🧠 Model Training & Mining":
        page_mining(df_proc)
    elif page == "🔮 Live Risk Prediction":
        page_predict()
    elif page == "📄 Documentation":
        page_docs()

if __name__ == "__main__":
    main()
