# COM6003 Data Science — Liverpool EPC Analysis
# Streamlit Dashboard — Buckinghamshire New University 2025-26

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               RandomForestRegressor, GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                              mean_absolute_error, mean_squared_error, r2_score)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Liverpool EPC — COM6003",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Professional GitHub-inspired dark theme ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

.stApp { background: #0d1117; }

[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* Top header strip */
.page-header {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.8rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.page-header h1 {
    color: #e6edf3;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.3px;
}
.page-header p {
    color: #8b949e;
    font-size: 0.8rem;
    margin: 0.2rem 0 0 0;
}
.badge {
    background: #1f6feb22;
    border: 1px solid #1f6feb;
    color: #58a6ff;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* Metric strip */
.kpi-row { display: flex; gap: 0.85rem; margin-bottom: 1.6rem; flex-wrap: wrap; }
.kpi {
    flex: 1;
    min-width: 130px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    position: relative;
    overflow: hidden;
}
.kpi::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #58a6ff, #3fb950);
}
.kpi-val { font-size: 1.75rem; font-weight: 700; color: #e6edf3; line-height: 1.1; }
.kpi-label { font-size: 0.7rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.9px; margin-top: 0.3rem; }

/* Section title */
.sec-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #e6edf3;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 1.6rem 0 0.9rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #30363d;
}

/* Insight cards — three variants */
.card {
    border-radius: 7px;
    padding: 0.85rem 1.1rem;
    margin: 0.45rem 0;
    font-size: 0.875rem;
    line-height: 1.65;
    color: #c9d1d9;
}
.card-blue  { background: #0c1e35; border-left: 3px solid #58a6ff; }
.card-green { background: #0c1f14; border-left: 3px solid #3fb950; }
.card-amber { background: #1f1a0c; border-left: 3px solid #d29922; }
.card-red   { background: #200d0d; border-left: 3px solid #f85149; }
.card b     { color: #e6edf3; }

/* Prediction result box */
.pred-result {
    border-radius: 10px;
    padding: 1.8rem;
    text-align: center;
    border: 1px solid #30363d;
}
.pred-grade {
    font-size: 5.5rem;
    font-weight: 800;
    line-height: 1;
    font-family: 'JetBrains Mono', monospace;
}

/* Buttons */
.stButton > button {
    background: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 7px;
    padding: 0.55rem 1.5rem;
    font-weight: 500;
    font-size: 0.9rem;
    width: 100%;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: #30363d;
    border-color: #58a6ff;
    color: #58a6ff;
}

/* Tables */
.stDataFrame { border-radius: 8px; overflow: hidden; }
[data-testid="stDataFrame"] th {
    background: #161b22 !important;
    color: #8b949e !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 7px;
    padding: 3px;
    gap: 3px;
}
.stTabs [data-baseweb="tab"] {
    color: #8b949e !important;
    font-size: 0.83rem;
    font-weight: 500;
    border-radius: 5px;
    padding: 0.4rem 1rem;
}
.stTabs [aria-selected="true"] {
    background: #21262d !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
}

/* Selectbox / slider labels */
.stSelectbox label, .stSlider label, .stNumberInput label {
    font-size: 0.8rem !important;
    color: #8b949e !important;
    font-weight: 500 !important;
}

hr.divider { border: none; border-top: 1px solid #30363d; margin: 1.4rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Chart defaults ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':   '#161b22',
    'axes.facecolor':     '#161b22',
    'axes.edgecolor':     '#30363d',
    'axes.labelcolor':    '#8b949e',
    'axes.titlecolor':    '#e6edf3',
    'axes.titlesize':     12,
    'axes.titleweight':   'bold',
    'axes.titlepad':      14,
    'axes.labelsize':     10,
    'text.color':         '#c9d1d9',
    'xtick.color':        '#8b949e',
    'ytick.color':        '#8b949e',
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'grid.color':         '#21262d',
    'grid.alpha':         1.0,
    'legend.facecolor':   '#161b22',
    'legend.edgecolor':   '#30363d',
    'legend.labelcolor':  '#c9d1d9',
    'legend.fontsize':    9,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.spines.left':   True,
    'axes.spines.bottom': True,
})

# ── Constants ─────────────────────────────────────────────────────────────────
AGE_ORDER  = ['Pre-1900','1900-1949','1950-1975','1976-1990','1991-2002','2003-2021','Post-2021']
EFF_ORDER  = ['Very Good','Good','Average','Poor','Very Poor']
TYPE_COLORS = {
    'House':      '#58a6ff',
    'Flat':       '#3fb950',
    'Bungalow':   '#d29922',
    'Maisonette': '#f78166',
}
RATING_COLORS = {'A':'#3fb950','B':'#26a641','C':'#d29922','D':'#e16060','E':'#f85149'}
PALETTE = ['#58a6ff','#3fb950','#d29922','#f78166','#a371f7','#79c0ff','#56d364']
CLUSTER_COLORS = ['#f85149','#d29922','#3fb950','#58a6ff']
CLUSTER_ORDER  = ['Low Efficiency','Below Average','Above Average','High Efficiency']


def _card(text, variant='blue'):
    st.markdown(f'<div class="card card-{variant}">{text}</div>', unsafe_allow_html=True)


def _sec(title):
    st.markdown(f'<p class="sec-title">{title}</p>', unsafe_allow_html=True)


def _divider():
    st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('liverpool_epc_cleaned.csv', low_memory=False)
    if 'TENURE_CLEAN' not in df.columns:
        df['TENURE_CLEAN'] = df['TENURE'].map({
            'Rented (social)':  'Social Rented',
            'Owner-occupied':   'Owner-occupied',
            'Rented (private)': 'Private Rented'
        }).fillna('Other')
    if 'PROPERTY_AGE_GROUP' not in df.columns:
        def _age(v):
            if pd.isnull(v): return 'Unknown'
            v = str(v)
            if 'before 1900' in v:                        return 'Pre-1900'
            elif '1900-1929' in v or '1930-1949' in v:   return '1900-1949'
            elif '1950-1966' in v or '1967-1975' in v:   return '1950-1975'
            elif '1976-1982' in v or '1983-1990' in v:   return '1976-1990'
            elif '1991-1995' in v or '1996-2002' in v:   return '1991-2002'
            elif any(x in v for x in ['2003','2007','2012','2020']): return '2003-2021'
            else:                                         return 'Post-2021'
        df['PROPERTY_AGE_GROUP'] = df['CONSTRUCTION_AGE_BAND'].apply(_age)
    return df


def _feature_matrix(df):
    fcols = [
        'TOTAL_FLOOR_AREA','FLOOR_HEIGHT','NUMBER_HABITABLE_ROOMS','NUMBER_HEATED_ROOMS',
        'MULTI_GLAZE_PROPORTION','LOW_ENERGY_LIGHTING','EXTENSION_COUNT',
        'WIND_TURBINE_COUNT','PHOTO_SUPPLY',
        'PROPERTY_TYPE','BUILT_FORM','PROPERTY_AGE_GROUP','TENURE_CLEAN',
        'MAINS_GAS_FLAG','SOLAR_WATER_HEATING_FLAG','MAIN_FUEL','ENERGY_TARIFF',
        'WALLS_ENERGY_EFF','ROOF_ENERGY_EFF','WINDOWS_ENERGY_EFF',
        'MAINHEAT_ENERGY_EFF','MAINHEATC_ENERGY_EFF',
        'HOT_WATER_ENERGY_EFF','LIGHTING_ENERGY_EFF','INSPECTION_YEAR',
    ]
    fcols = [c for c in fcols if c in df.columns]
    enc   = df[fcols].copy()
    led   = {}
    for col in enc.select_dtypes(include='object').columns:
        le = LabelEncoder()
        enc[col] = le.fit_transform(enc[col].astype(str))
        led[col] = le
    return enc.values, fcols, led


@st.cache_resource
def train_models(_df):
    X, fcols, le_dict = _feature_matrix(_df)
    y_raw = _df['CURRENT_ENERGY_RATING'].values
    le_tgt = LabelEncoder()
    y = le_tgt.fit_transform(y_raw)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)

    lr  = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tr_s, y_tr); lr_p = lr.predict(X_te_s)

    rf  = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr);  rf_p = rf.predict(X_te)

    gb  = GradientBoostingClassifier(n_estimators=200, random_state=42)
    gb.fit(X_tr, y_tr);  gb_p = gb.predict(X_te)

    cv_rf = cross_val_score(rf, X_tr, y_tr, cv=5, scoring='accuracy')
    cv_gb = cross_val_score(gb, X_tr, y_tr, cv=5, scoring='accuracy')

    # Per-class accuracy breakdown for property type analysis
    X_te_df = pd.DataFrame(X_te, columns=fcols)
    pt_col  = 'PROPERTY_TYPE'
    le_pt   = le_dict.get(pt_col)
    type_acc = {}
    if le_pt is not None:
        raw_types = le_pt.inverse_transform(X_te_df[pt_col].astype(int).values)
        te_idx    = np.where(np.isin(np.arange(len(_df)),
                    train_test_split(np.arange(len(_df)), test_size=0.2, random_state=42)[1]))[0]
        for pt in np.unique(raw_types):
            mask = raw_types == pt
            if mask.sum() > 5:
                type_acc[pt] = {
                    'LR':  accuracy_score(y_te[mask], lr_p[mask]),
                    'RF':  accuracy_score(y_te[mask], rf_p[mask]),
                    'GB':  accuracy_score(y_te[mask], gb_p[mask]),
                    'n':   mask.sum()
                }

    metrics = {
        'lr_acc': accuracy_score(y_te, lr_p),
        'rf_acc': accuracy_score(y_te, rf_p),
        'gb_acc': accuracy_score(y_te, gb_p),
        'lr_f1':  f1_score(y_te, lr_p, average='weighted'),
        'rf_f1':  f1_score(y_te, rf_p, average='weighted'),
        'gb_f1':  f1_score(y_te, gb_p, average='weighted'),
        'gb_cm':      confusion_matrix(y_te, gb_p),
        'gb_cm_norm': confusion_matrix(y_te, gb_p, normalize='true'),
        'cv_rf': cv_rf, 'cv_gb': cv_gb,
        'type_acc': type_acc,
    }
    return rf, gb, lr, sc, le_dict, le_tgt, fcols, X_te, y_te, metrics


@st.cache_resource
def train_regression(_df):
    X, fcols, _ = _feature_matrix(_df)
    def _rm(name, yt, yp):
        mse = mean_squared_error(yt, yp)
        return {'Model': name,
                'MAE':  round(mean_absolute_error(yt, yp), 3),
                'MSE':  round(mse, 3),
                'RMSE': round(np.sqrt(mse), 3),
                'R2':   round(r2_score(yt, yp), 4)}
    results = {}
    for tname, tcol in [('Efficiency Score', 'CURRENT_ENERGY_EFFICIENCY'),
                        ('Cost Saving (GBP)', 'COST_SAVING_POTENTIAL')]:
        y = _df[tcol].values
        Xr, Xt, yr, yt = train_test_split(X, y, test_size=0.2, random_state=42)
        sc  = StandardScaler()
        Xrs, Xts = sc.fit_transform(Xr), sc.transform(Xt)
        lr_r = LinearRegression();            lr_r.fit(Xrs, yr); lp = lr_r.predict(Xts)
        rf_r = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf_r.fit(Xr, yr);  rp = rf_r.predict(Xt)
        gb_r = GradientBoostingRegressor(n_estimators=200, random_state=42)
        gb_r.fit(Xr, yr);  gp = gb_r.predict(Xt)
        results[tname] = {
            'metrics': pd.DataFrame([_rm('Linear Regression', yt, lp),
                                     _rm('Random Forest', yt, rp),
                                     _rm('Gradient Boosting', yt, gp)]),
            'y_test': yt, 'lr_pred': lp, 'rf_pred': rp, 'gb_pred': gp,
        }
    return results


@st.cache_resource
def run_clustering(_df):
    X, fcols, _ = _feature_matrix(_df)
    sc  = StandardScaler()
    Xs  = sc.fit_transform(X)
    inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs).inertia_
                for k in range(2, 11)]
    km  = KMeans(n_clusters=4, random_state=42, n_init=10)
    lbl = km.fit_predict(Xs)
    pca = PCA(n_components=2, random_state=42)
    X2d = pca.fit_transform(Xs)
    tmp = _df.copy(); tmp['_cl'] = lbl
    means  = tmp.groupby('_cl')['CURRENT_ENERGY_EFFICIENCY'].mean()
    sorted_cl = means.sort_values().index.tolist()
    names = {sorted_cl[i]: CLUSTER_ORDER[i] for i in range(4)}
    return inertias, lbl, [names[l] for l in lbl], X2d, names, Xs


@st.cache_data
def load_recs():
    certs = pd.read_csv('certificates.csv', low_memory=False,
                        usecols=['LMK_KEY','CURRENT_ENERGY_RATING','PROPERTY_TYPE','CONSTRUCTION_AGE_BAND'])
    recs  = pd.read_csv('recommendations.csv', low_memory=False)
    return certs.merge(recs, on='LMK_KEY', how='inner')


# ── Load & train ──────────────────────────────────────────────────────────────
df = load_data()
with st.spinner("Loading models..."):
    rf, gb, lr_clf, sc_clf, le_dict, le_tgt, fcols, X_te, y_te, metrics = train_models(df)

gb_cv = metrics['cv_gb'].mean()
rf_cv = metrics['cv_rf'].mean()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0.5rem 0.8rem">
        <div style="font-size:1.5rem;font-weight:700;color:#e6edf3;letter-spacing:-0.5px;">
            Liverpool EPC
        </div>
        <div style="font-size:0.75rem;color:#8b949e;margin-top:0.2rem;">
            COM6003 Data Science
        </div>
    </div>
    <hr style="border:none;border-top:1px solid #30363d;margin:0 0 1rem 0;">
    """, unsafe_allow_html=True)

    page = st.selectbox("Navigate", [
        "Overview",
        "Descriptive Analytics",
        "Diagnostic Analytics",
        "Predictive Models",
        "Energy Rating Predictor",
        "Regression Analysis",
        "Clustering",
        "Recommendation System",
        "Conclusions",
    ], label_visibility="collapsed")

    st.markdown(f"""
    <hr style="border:none;border-top:1px solid #30363d;margin:1rem 0;">
    <div style="font-size:0.78rem;color:#8b949e;line-height:2.1;">
        <div style="color:#c9d1d9;font-weight:600;margin-bottom:0.4rem;">Dataset Summary</div>
        <div>Properties: <span style="color:#58a6ff;font-weight:600;">4,579</span></div>
        <div>Features: <span style="color:#58a6ff;font-weight:600;">25 physical</span></div>
        <div>Target: <span style="color:#58a6ff;font-weight:600;">Energy Rating A–E</span></div>
        <div>Best CV: <span style="color:#3fb950;font-weight:600;">{gb_cv*100:.1f}% (GB)</span></div>
    </div>
    <hr style="border:none;border-top:1px solid #30363d;margin:1rem 0;">
    <div style="font-size:0.72rem;color:#6e7681;text-align:center;line-height:1.8;">
        Buckinghamshire New University<br>Academic Year 2025–26
    </div>
    """, unsafe_allow_html=True)


# ── Page header helper ────────────────────────────────────────────────────────
def _header(title, subtitle, badge=None):
    badge_html = f'<span class="badge">{badge}</span>' if badge else ''
    st.markdown(f"""
    <div class="page-header">
        <div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        {badge_html}
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    _header("Liverpool EPC Analysis",
            "Energy Performance Certificate data — 4,579 residential properties",
            "Local Authority: E08000012")

    avg_eff = df['CURRENT_ENERGY_EFFICIENCY'].mean()
    avg_sav = df['COST_SAVING_POTENTIAL'].mean()
    avg_co2 = df['CO2_EMISSIONS_CURRENT'].mean()
    best_type = df.groupby('PROPERTY_TYPE')['CURRENT_ENERGY_EFFICIENCY'].mean().idxmax()

    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi"><div class="kpi-val">{len(df):,}</div><div class="kpi-label">Properties Assessed</div></div>
      <div class="kpi"><div class="kpi-val">{avg_eff:.1f}</div><div class="kpi-label">Avg Efficiency Score</div></div>
      <div class="kpi"><div class="kpi-val">£{avg_sav:.0f}</div><div class="kpi-label">Avg Annual Saving</div></div>
      <div class="kpi"><div class="kpi-val">{avg_co2:.2f}t</div><div class="kpi-label">Avg CO₂/year</div></div>
      <div class="kpi"><div class="kpi-val">{gb_cv*100:.1f}%</div><div class="kpi-label">Best CV Accuracy</div></div>
      <div class="kpi"><div class="kpi-val">{best_type}</div><div class="kpi-label">Best Performing Type</div></div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1.05, 1])

    with col1:
        _sec("Project Overview")
        _card("This project applies a full data science pipeline to EPC records for Liverpool (Local Authority: E08000012), sourced from the Ministry of Housing, Communities and Local Government open data portal. Properties are rated A–G using the Standard Assessment Procedure (SAP).", 'blue')
        _card("The analysis covers <b>descriptive, diagnostic, predictive, clustering, and recommendation</b> stages — spanning classification, regression, K-Means clustering, and a content-based recommendation engine built on cosine similarity.", 'green')
        _card("All models use 25 physical building features only. SAP-derived scores are excluded from features to prevent data leakage — a critical methodological control ensuring honest, generalisable predictions.", 'amber')

        _sec("Model Performance at a Glance")
        perf_df = pd.DataFrame({
            'Algorithm': ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
            'Test Accuracy': [f"{metrics['lr_acc']*100:.2f}%", f"{metrics['rf_acc']*100:.2f}%", f"{metrics['gb_acc']*100:.2f}%"],
            'F1 (weighted)': [f"{metrics['lr_f1']*100:.2f}%", f"{metrics['rf_f1']*100:.2f}%", f"{metrics['gb_f1']*100:.2f}%"],
            '5-fold CV': ['—', f"{rf_cv*100:.2f}%", f"{gb_cv*100:.2f}%"],
            'Type': ['Linear baseline', 'Bagging ensemble', 'Boosting ensemble'],
        })
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

    with col2:
        _sec("Energy Efficiency by Property Type")
        pt_eff = df.groupby('PROPERTY_TYPE')['CURRENT_ENERGY_EFFICIENCY'].agg(['mean','std']).reset_index()
        pt_eff.columns = ['Type','Mean','Std']
        pt_eff = pt_eff.sort_values('Mean', ascending=False)

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        colors = [TYPE_COLORS.get(t, '#8b949e') for t in pt_eff['Type']]
        bars = ax.barh(pt_eff['Type'], pt_eff['Mean'], xerr=pt_eff['Std'],
                       color=colors, height=0.45, edgecolor='#0d1117', linewidth=0.8,
                       error_kw=dict(ecolor='#6e7681', capsize=4, linewidth=1.2))
        for bar, row in zip(bars, pt_eff.itertuples()):
            ax.text(row.Mean + row.Std + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{row.Mean:.1f} ± {row.Std:.1f}',
                    va='center', fontsize=9, fontweight='600', color='#e6edf3')
        ax.set_xlabel('Average SAP Efficiency Score (with ±1 SD)', fontsize=9.5)
        ax.set_title('Average Energy Efficiency by Property Type', pad=14)
        ax.set_xlim(60, 85)
        ax.xaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        _card(f"Maisonettes lead with an average of 75.4 points, followed by Flats (72.9) and Houses (72.5). Bungalows score lowest at 72.4. The differences are modest, but Maisonettes' party-wall construction reduces heat loss on multiple sides.", 'green')


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DESCRIPTIVE ANALYTICS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Descriptive Analytics":
    _header("Descriptive Analytics", "What has occurred? — characterising Liverpool's housing stock", "Descriptive")

    tab1, tab2, tab3 = st.tabs(["Property Type & Efficiency", "Timeline Analysis", "Tenure & Age"])

    # ── Tab 1: Property type focus ────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            _sec("Average Efficiency Score by Property Type")
            pt = df.groupby('PROPERTY_TYPE')['CURRENT_ENERGY_EFFICIENCY'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(pt.index, pt.values,
                          color=[TYPE_COLORS.get(t,'#888') for t in pt.index],
                          width=0.5, edgecolor='#0d1117', linewidth=0.8)
            for bar, val in zip(bars, pt.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15,
                        f'{val:.1f}', ha='center', fontsize=10, fontweight='700', color='#e6edf3')
            ax.set_ylabel('Mean SAP Score', fontsize=9.5)
            ax.set_title('Mean Energy Efficiency by Property Type')
            ax.set_ylim(68, 78); ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
            fig.tight_layout(); st.pyplot(fig); plt.close()
            _card("Maisonettes record the highest average efficiency (75.4), benefiting from shared-wall heat retention. The spread across all types is narrow (72–75), indicating that property type alone is not a decisive factor — building age and insulation quality matter more.", 'green')

        with col2:
            _sec("Energy Rating Distribution by Property Type")
            ct = pd.crosstab(df['PROPERTY_TYPE'], df['CURRENT_ENERGY_RATING'])
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            bottom = np.zeros(len(ct_pct))
            for rating in ct_pct.columns:
                color = RATING_COLORS.get(rating, '#888')
                vals  = ct_pct[rating].values
                bars  = ax.bar(ct_pct.index, vals, bottom=bottom,
                               label=f'Band {rating}', color=color,
                               width=0.48, edgecolor='#0d1117', linewidth=0.6)
                for bar, val in zip(bars, vals):
                    if val > 14:   # only label segments large enough
                        ax.text(bar.get_x()+bar.get_width()/2,
                                bar.get_y()+bar.get_height()/2,
                                f'{val:.0f}%', ha='center', va='center',
                                fontsize=9, fontweight='700', color='white')
                bottom += vals
            ax.set_ylabel('Proportion (%)', fontsize=9.5)
            ax.set_title('Energy Rating Band Composition by Property Type', pad=12)
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8.5,
                      framealpha=0.9, borderpad=0.8)
            ax.set_ylim(0, 108)
            ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
            fig.tight_layout(); st.pyplot(fig); plt.close()
            _card("Houses have the highest share of Band A properties (2.2%) but also the most Band D/E exposure. Flats cluster overwhelmingly in Band C (62.5%), reflecting their compact, heat-retaining form factor.", 'blue')

        _divider()
        col3, col4 = st.columns(2)

        with col3:
            _sec("CO₂ Emissions by Property Type")
            co2 = df.groupby('PROPERTY_TYPE')['CO2_EMISSIONS_CURRENT'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(co2.index, co2.values,
                          color=[TYPE_COLORS.get(t,'#888') for t in co2.index],
                          width=0.5, edgecolor='#0d1117', linewidth=0.8)
            for bar, val in zip(bars, co2.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                        f'{val:.2f}t', ha='center', fontsize=9.5, fontweight='700', color='#e6edf3')
            ax.set_ylabel('Average CO₂ Emissions (tonnes/year)', fontsize=9.5)
            ax.set_title('Average Annual CO₂ by Property Type')
            ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
            fig.tight_layout(); st.pyplot(fig); plt.close()
            _card("Houses produce 2.39 tonnes CO₂/year on average — nearly twice the Flat average of 1.29t. This reflects their larger floor area and greater exposed surface area. Targeting Houses for retrofit delivers the highest carbon reduction per property.", 'amber')

        with col4:
            _sec("Property Type — Stock Count and Average Efficiency")
            types_sorted = df['PROPERTY_TYPE'].value_counts().index.tolist()
            counts       = df['PROPERTY_TYPE'].value_counts()
            eff_vals     = [df[df['PROPERTY_TYPE']==t]['CURRENT_ENERGY_EFFICIENCY'].mean()
                            for t in types_sorted]
            c_vals       = counts[types_sorted].values
            bar_colors   = [TYPE_COLORS.get(t,'#888') for t in types_sorted]

            fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6.5, 6),
                                                  gridspec_kw={'hspace': 0.55})

            # Top panel — stock count
            bars_t = ax_top.bar(types_sorted, c_vals, color=bar_colors,
                                width=0.48, edgecolor='#0d1117', linewidth=0.8)
            for bar, val in zip(bars_t, c_vals):
                ax_top.text(bar.get_x()+bar.get_width()/2,
                            bar.get_height() + max(c_vals)*0.03,
                            f'{val:,}\n({val/len(df)*100:.1f}%)',
                            ha='center', va='bottom',
                            fontsize=8.5, fontweight='600', color='#e6edf3')
            ax_top.set_ylim(0, max(c_vals) * 1.32)
            ax_top.set_ylabel('Number of Properties', fontsize=9)
            ax_top.set_title('Stock Count by Property Type', fontsize=10, fontweight='700', pad=10)
            ax_top.yaxis.grid(True, alpha=0.35); ax_top.set_axisbelow(True)

            # Bottom panel — average efficiency
            bars_b = ax_bot.bar(types_sorted, eff_vals, color=bar_colors,
                                width=0.48, edgecolor='#0d1117', linewidth=0.8)
            for bar, val in zip(bars_b, eff_vals):
                ax_bot.text(bar.get_x()+bar.get_width()/2,
                            bar.get_height() + 0.12,
                            f'{val:.1f}', ha='center', va='bottom',
                            fontsize=10, fontweight='700', color='#e6edf3')
            ax_bot.set_ylim(68, max(eff_vals) * 1.06)
            ax_bot.set_ylabel('Avg SAP Score', fontsize=9)
            ax_bot.set_title('Average Efficiency Score by Property Type',
                             fontsize=10, fontweight='700', pad=10)
            ax_bot.yaxis.grid(True, alpha=0.35); ax_bot.set_axisbelow(True)

            fig.suptitle('Property Stock vs Energy Efficiency', fontsize=11,
                         fontweight='700', color='#8b949e', y=1.01)
            fig.tight_layout(); st.pyplot(fig); plt.close()
            _card("Despite Houses forming 63.7% of the stock, their average efficiency (72.5) is marginally below Flats and Maisonettes. Volume alone does not translate to better energy performance.", 'blue')

    # ── Tab 2: Timeline ───────────────────────────────────────────────────────
    with tab2:
        _sec("Average Energy Efficiency by Construction Decade and Property Type")
        _card("Construction decade is used as the timeline proxy. Inspection year data spans only 2024–2026 (too narrow for trend analysis), so property age bands provide the meaningful historical dimension, showing how building standards evolved across seven eras.", 'blue')

        age_type_eff = (df.groupby(['PROPERTY_AGE_GROUP','PROPERTY_TYPE'])
                        ['CURRENT_ENERGY_EFFICIENCY'].mean().unstack())
        age_type_eff = age_type_eff.reindex([a for a in AGE_ORDER if a in age_type_eff.index])

        # Filter out cells with fewer than 5 properties to avoid misleading spikes
        age_type_cnt = (df.groupby(['PROPERTY_AGE_GROUP','PROPERTY_TYPE'])
                        .size().unstack(fill_value=0)
                        .reindex(age_type_eff.index, fill_value=0))
        age_type_plot = age_type_eff.copy()
        for pt in age_type_plot.columns:
            if pt in age_type_cnt.columns:
                age_type_plot.loc[age_type_cnt[pt] < 5, pt] = np.nan

        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(age_type_plot))

        for i, ptype in enumerate(age_type_plot.columns):
            vals = age_type_plot[ptype].values
            mask = ~np.isnan(vals)
            if mask.sum() < 2:
                continue
            color = TYPE_COLORS.get(ptype, PALETTE[i])
            ax.plot(x[mask], vals[mask], 'o-',
                    color=color, linewidth=2.5, markersize=8,
                    label=ptype, zorder=4,
                    markeredgecolor='#0d1117', markeredgewidth=1.0)
            # One clean label at the last data point only
            lx, lv = x[mask][-1], vals[mask][-1]
            ax.text(lx, lv + 0.6, f'{lv:.0f}',
                    ha='center', va='bottom', fontsize=9,
                    color=color, fontweight='700')

        ax.set_xticks(x)
        ax.set_xticklabels(age_type_plot.index, fontsize=10, rotation=20, ha='right')
        ax.set_ylabel('Average SAP Efficiency Score', fontsize=10)
        ax.set_title('Energy Efficiency Trend by Construction Decade and Property Type',
                     fontsize=12, fontweight='700', pad=14)
        ax.set_xlim(-0.4, len(x) - 0.4)
        ax.set_ylim(62, 86)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.85,
                  facecolor='#161b22', edgecolor='#30363d')
        ax.yaxis.grid(True, alpha=0.35, linestyle='--')
        ax.set_axisbelow(True)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        _card("Cells with fewer than 5 properties are excluded to avoid misleading averages. All types show an upward efficiency trend from pre-war to modern construction. Houses improve from 69 (Pre-1900) to 79 (Post-2021); Flats from 70 to 79 — both driven by progressively stricter Building Regulations.", 'blue')

        _divider()
        col1, col2 = st.columns(2)

        with col1:
            _sec("Efficiency Gap by Construction Decade (All Types)")
            age_eff = df.groupby('PROPERTY_AGE_GROUP')['CURRENT_ENERGY_EFFICIENCY'].mean().reindex(
                [a for a in AGE_ORDER if a in df['PROPERTY_AGE_GROUP'].values]).dropna()
            age_cnt = df.groupby('PROPERTY_AGE_GROUP').size().reindex(age_eff.index)
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(range(len(age_eff)), age_eff.values,
                          color=plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(age_eff))),
                          width=0.6, edgecolor='#0d1117', linewidth=0.8)
            ax.plot(range(len(age_eff)), age_eff.values, 'o--',
                    color='#58a6ff', linewidth=1.8, markersize=6, zorder=5, alpha=0.9)
            for i, (val, cnt) in enumerate(zip(age_eff.values, age_cnt.values)):
                ax.text(i, val + 0.25, f'{val:.1f}', ha='center', fontsize=8.5, fontweight='600', color='#e6edf3')
                ax.text(i, 63.5, f'n={cnt:,}', ha='center', fontsize=7.5, color='#8b949e')
            ax.set_xticks(range(len(age_eff)))
            ax.set_xticklabels(age_eff.index, rotation=25, ha='right', fontsize=8.5)
            ax.set_ylabel('Average Efficiency Score', fontsize=9.5)
            ax.set_title('Efficiency by Construction Era')
            ax.set_ylim(62, 82); ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
            fig.tight_layout(); st.pyplot(fig); plt.close()
            rng = age_eff.max() - age_eff.min()
            _card(f"Post-2021 properties score {age_eff.max():.1f} vs Pre-1900 at {age_eff.min():.1f} — a {rng:.1f}-point gap reflecting 120 years of progressive building regulation improvements.", 'green')

        with col2:
            _sec("Property Type Share Across Construction Decades")
            age_type_ct = pd.crosstab(df['PROPERTY_AGE_GROUP'], df['PROPERTY_TYPE'])
            age_type_ct = age_type_ct.reindex([a for a in AGE_ORDER if a in age_type_ct.index])
            age_type_pct = age_type_ct.div(age_type_ct.sum(axis=1), axis=0) * 100
            fig, ax = plt.subplots(figsize=(7, 4.5))
            bottom = np.zeros(len(age_type_pct))
            for pt in age_type_pct.columns:
                vals = age_type_pct[pt].values
                bars = ax.bar(range(len(age_type_pct)), vals,
                              bottom=bottom, label=pt,
                              color=TYPE_COLORS.get(pt,'#888'), width=0.58,
                              edgecolor='#0d1117', linewidth=0.5)
                for bar, val in zip(bars, vals):
                    if val > 18:   # only label where there's enough room
                        ax.text(bar.get_x()+bar.get_width()/2,
                                bar.get_y()+bar.get_height()/2,
                                f'{val:.0f}%', ha='center', va='center',
                                fontsize=8.5, fontweight='700', color='white')
                bottom += vals
            ax.set_xticks(range(len(age_type_pct)))
            ax.set_xticklabels(age_type_pct.index, rotation=25, ha='right', fontsize=8.5)
            ax.set_ylabel('Proportion (%)', fontsize=9.5)
            ax.set_title('Property Type Mix by Construction Era', pad=12)
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8.5,
                      framealpha=0.9, borderpad=0.8)
            ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
            fig.tight_layout(); st.pyplot(fig); plt.close()
            _card("Victorian and Edwardian eras (Pre-1900, 1900–1949) are almost exclusively Houses. Flats become significant from 1950 onwards, reflecting post-war high-density social housing. Post-2021 shows a more balanced mix.", 'amber')

    # ── Tab 3: Tenure & Age ───────────────────────────────────────────────────
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            _sec("Efficiency Distribution by Property Type (Violin)")
            types = sorted(df['PROPERTY_TYPE'].unique())
            data  = [df[df['PROPERTY_TYPE']==t]['CURRENT_ENERGY_EFFICIENCY'].values for t in types]
            fig, ax = plt.subplots(figsize=(6, 4))
            parts = ax.violinplot(data, positions=range(len(types)),
                                  widths=0.55, showmedians=True, showextrema=True)
            for i, (pc, t) in enumerate(zip(parts['bodies'], types)):
                pc.set_facecolor(TYPE_COLORS.get(t,'#888'))
                pc.set_alpha(0.65)
                med = np.median(df[df['PROPERTY_TYPE']==t]['CURRENT_ENERGY_EFFICIENCY'])
                ax.text(i, med + 0.4, f'{med:.0f}', ha='center', fontsize=9, fontweight='700', color='#e6edf3')
            for key in ['cmedians','cmins','cmaxes','cbars']:
                if key in parts: parts[key].set_color('#8b949e')
            ax.set_xticks(range(len(types))); ax.set_xticklabels(types, fontsize=9)
            ax.set_ylabel('SAP Efficiency Score', fontsize=9.5)
            ax.set_title('Efficiency Score Distribution by Property Type')
            ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
            fig.tight_layout(); st.pyplot(fig); plt.close()
            _card("All four property types show similar median scores (72–75). Houses have the widest spread — some achieve Band A (score 92+) while others fall to Band E (below 39). Flats have a tighter, more consistent distribution.", 'blue')

        with col2:
            _sec("Tenure Distribution and Average Efficiency")
            tenure_eff = df.groupby('TENURE_CLEAN')['CURRENT_ENERGY_EFFICIENCY'].mean().sort_values(ascending=False)
            tenure_cnt = df.groupby('TENURE_CLEAN').size()
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(tenure_eff.index, tenure_eff.values,
                          color=PALETTE[:len(tenure_eff)], width=0.45,
                          edgecolor='#0d1117', linewidth=0.8)
            for bar, val, cnt in zip(bars, tenure_eff.values, tenure_cnt[tenure_eff.index]):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                        f'{val:.1f}\n(n={cnt:,})', ha='center', fontsize=9, fontweight='600', color='#e6edf3')
            ax.set_ylabel('Average Efficiency Score', fontsize=9.5)
            ax.set_title('Average Efficiency by Tenure')
            ax.set_ylim(68, 78); ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
            ax.tick_params(axis='x', rotation=10)
            fig.tight_layout(); st.pyplot(fig); plt.close()
            _card("Social rented properties perform slightly better on average, likely reflecting prior investment through the Energy Company Obligation (ECO) scheme. Owner-occupied properties show the largest efficiency gap — highlighting underutilised self-improvement potential.", 'amber')


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DIAGNOSTIC ANALYTICS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Diagnostic Analytics":
    _header("Diagnostic Analytics", "Why does energy performance vary? — root cause investigation", "Diagnostic")

    _sec("Correlation Matrix — Key Variables")
    from matplotlib.colors import LinearSegmentedColormap
    _corr_cmap = LinearSegmentedColormap.from_list(
        'epc_corr',
        ['#c0392b', '#922b21', '#2c2c2c', '#1a4a2e', '#1e8449'],
        N=256)
    num_cols = ['CURRENT_ENERGY_EFFICIENCY','POTENTIAL_ENERGY_EFFICIENCY','TOTAL_FLOOR_AREA',
                'CO2_EMISSIONS_CURRENT','HEATING_COST_CURRENT','TOTAL_COST_CURRENT',
                'EFFICIENCY_GAP','COST_SAVING_POTENTIAL','CO2_PER_AREA','NUMBER_HABITABLE_ROOMS']
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap=_corr_cmap, center=0, vmin=-1, vmax=1, ax=ax,
                annot_kws={'size': 9, 'weight': 'bold'},
                linewidths=0.5, linecolor='#0d1117',
                cbar_kws={'shrink': 0.72, 'label': 'Pearson r'})
    # Auto-colour annotation text: white on dark, black on light cells
    for text_obj in ax.texts:
        try:
            val = float(text_obj.get_text())
            text_obj.set_color('white' if abs(val) > 0.25 else '#c9d1d9')
        except ValueError:
            pass
    ax.set_title('Pearson Correlation Matrix — Key Numerical Features', fontsize=13,
                 fontweight='700', pad=16)
    ax.tick_params(axis='x', rotation=42, labelsize=8.5)
    ax.tick_params(axis='y', rotation=0, labelsize=8.5)
    fig.tight_layout(); st.pyplot(fig); plt.close()
    _card("POTENTIAL_ENERGY_EFFICIENCY correlates strongly with current efficiency (r = +0.65). CO₂ emissions and heating costs are negatively correlated (r = −0.41 and −0.53 respectively), confirming lower-rated properties are costlier and more polluting.", 'blue')

    _divider()
    col1, col2 = st.columns(2)

    with col1:
        _sec("Efficiency by Property Type and Wall Insulation")
        wall_type = (df.groupby(['PROPERTY_TYPE','WALLS_ENERGY_EFF'])
                     ['CURRENT_ENERGY_EFFICIENCY'].mean().unstack())
        wall_type = wall_type.reindex(columns=[e for e in EFF_ORDER if e in wall_type.columns])
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        x = np.arange(len(wall_type)); w = 0.15
        colors_w = ['#3fb950','#56d364','#d29922','#e16060','#f85149']
        for i, col_name in enumerate(wall_type.columns):
            offset = (i - len(wall_type.columns)/2) * w + w/2
            bars = ax.bar(x + offset, wall_type[col_name].fillna(0).values,
                          width=w*0.9, color=colors_w[i], edgecolor='#0d1117',
                          linewidth=0.5, label=col_name)
        ax.set_xticks(x); ax.set_xticklabels(wall_type.index, fontsize=9)
        ax.set_ylabel('Average Efficiency Score', fontsize=9.5)
        ax.set_title('Efficiency by Property Type and Wall Insulation Quality')
        ax.legend(fontsize=8, title='Wall Rating', title_fontsize=8.5)
        ax.set_ylim(60, 85); ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        _card("Across all property types, Very Good wall insulation delivers approximately 10–14 additional efficiency points. The gap is most pronounced for Houses — the largest segment and the primary retrofit target.", 'green')

    with col2:
        _sec("Efficiency by Construction Era — Property Type Comparison")
        age_type = (df.groupby(['PROPERTY_AGE_GROUP','PROPERTY_TYPE'])
                    ['CURRENT_ENERGY_EFFICIENCY'].mean().unstack())
        age_type = age_type.reindex([a for a in AGE_ORDER if a in age_type.index])
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        x = np.arange(len(age_type)); w = 0.18
        for i, pt in enumerate(age_type.columns):
            vals = age_type[pt].fillna(0).values
            offset = (i - len(age_type.columns)/2) * w + w/2
            ax.bar(x + offset, vals, width=w*0.9,
                   color=TYPE_COLORS.get(pt,'#888'), label=pt,
                   edgecolor='#0d1117', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(age_type.index, rotation=25, ha='right', fontsize=8.5)
        ax.set_ylabel('Average Efficiency Score', fontsize=9.5)
        ax.set_title('Efficiency by Construction Era and Property Type')
        ax.legend(fontsize=8.5); ax.set_ylim(55, 85)
        ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        _card("Pre-1900 Bungalows score lowest of all type-era combinations, while Post-2021 properties consistently outperform regardless of type. Building regulations introduced in 1965, 1990, and 2010 are visible as step-change improvements.", 'amber')

    _divider()
    col3, col4 = st.columns(2)

    with col3:
        _sec("CO₂ Emissions: Property Type vs Construction Decade")
        co2_pivot = (df.groupby(['PROPERTY_AGE_GROUP','PROPERTY_TYPE'])
                     ['CO2_EMISSIONS_CURRENT'].mean().unstack())
        co2_pivot = co2_pivot.reindex([a for a in AGE_ORDER if a in co2_pivot.index])
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for pt in co2_pivot.columns:
            vals = co2_pivot[pt].values
            mask = ~np.isnan(vals)
            ax.plot(np.arange(len(co2_pivot))[mask], vals[mask], 'o-',
                    color=TYPE_COLORS.get(pt,'#888'), linewidth=2,
                    markersize=6, label=pt)
        ax.set_xticks(range(len(co2_pivot)))
        ax.set_xticklabels(co2_pivot.index, rotation=25, ha='right', fontsize=8.5)
        ax.set_ylabel('Average CO₂ (tonnes/year)', fontsize=9.5)
        ax.set_title('CO₂ Emissions by Era and Property Type')
        ax.legend(fontsize=8.5); ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        _card("Houses emit consistently higher CO₂ across all eras due to their larger floor area and greater exposed surface. Pre-1900 Houses average 2.6t CO₂/year. Flats remain below 1.5t across all eras.", 'red')

    with col4:
        _sec("Efficiency Gap by Tenure — Which Group Has Most to Gain?")
        gap_t = df.groupby('TENURE_CLEAN')['EFFICIENCY_GAP'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        bars = ax.bar(gap_t.index, gap_t.values, color=PALETTE[:len(gap_t)],
                      width=0.45, edgecolor='#0d1117', linewidth=0.8)
        for bar, val in zip(bars, gap_t.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.06,
                    f'{val:.2f} pts', ha='center', fontsize=10, fontweight='600', color='#e6edf3')
        ax.set_ylabel('Average Efficiency Gap (potential − current)', fontsize=9.5)
        ax.set_title('Untapped Efficiency Potential by Tenure')
        ax.tick_params(axis='x', rotation=12)
        ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        _card(f"Owner-occupied properties have the largest efficiency gap ({gap_t.max():.2f} points), representing the greatest untapped improvement potential. These households have the financial capacity to retrofit but face less regulatory pressure than landlords.", 'amber')


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICTIVE MODELS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Predictive Models":
    _header("Predictive Models", "Classification of energy rating from 25 physical building features", "ML Classification")

    _card(f"Three classifiers predict Energy Rating band (A–E) using <b>25 physical building characteristics</b>. No SAP-derived scores are included — all columns derived from CURRENT_ENERGY_EFFICIENCY are excluded to prevent data leakage. 80/20 stratified split; cross-validation on training set only.", 'blue')

    col1, col2 = st.columns(2)

    with col1:
        _sec("Accuracy and F1 Score — All Models")
        fig, ax = plt.subplots(figsize=(6.5, 4))
        names = ['Logistic\nRegression','Random\nForest','Gradient\nBoosting']
        accs  = [metrics['lr_acc']*100, metrics['rf_acc']*100, metrics['gb_acc']*100]
        f1s   = [metrics['lr_f1']*100,  metrics['rf_f1']*100,  metrics['gb_f1']*100]
        x, w  = np.arange(3), 0.32
        b1 = ax.bar(x-w/2, accs, w, color=['#3b4c6b','#1a6b3c','#1f3d6b'],
                    edgecolor='#0d1117', linewidth=0.8, label='Accuracy')
        b2 = ax.bar(x+w/2, f1s,  w, color=['#58a6ff','#3fb950','#79c0ff'],
                    edgecolor='#0d1117', linewidth=0.8, label='F1 (weighted)')
        for bar in list(b1)+list(b2):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f'{bar.get_height():.1f}%', ha='center', fontsize=8.5, fontweight='600', color='#e6edf3')
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
        ax.set_ylim(55, max(accs+f1s)*1.12)
        ax.set_ylabel('Score (%)', fontsize=9.5)
        ax.legend(fontsize=9); ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
        ax.set_title('Classification Performance — Test Set')
        fig.tight_layout(); st.pyplot(fig); plt.close()
        _card(f"Gradient Boosting achieves the highest test accuracy ({metrics['gb_acc']*100:.2f}%) and F1 score ({metrics['gb_f1']*100:.2f}%). Logistic Regression ({metrics['lr_acc']*100:.2f}%) establishes the linear baseline.", 'green')

    with col2:
        _sec("Confusion Matrix — Gradient Boosting")
        cm = metrics['gb_cm']; cmn = metrics['gb_cm_norm']
        fig, ax = plt.subplots(figsize=(6.5, 4))
        sns.heatmap(cm, annot=False, cmap='Blues', ax=ax,
                    xticklabels=le_tgt.classes_, yticklabels=le_tgt.classes_,
                    linewidths=0.4, linecolor='#0d1117')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j+0.5, i+0.38, str(cm[i,j]),
                        ha='center', va='center', fontsize=10, fontweight='700',
                        color='white' if cmn[i,j]>0.5 else 'black')
                ax.text(j+0.5, i+0.65, f'({cmn[i,j]*100:.0f}%)',
                        ha='center', va='center', fontsize=7.5,
                        color='white' if cmn[i,j]>0.5 else '#444')
        ax.set_xlabel('Predicted Band', fontsize=9.5)
        ax.set_ylabel('Actual Band', fontsize=9.5)
        ax.set_title('Gradient Boosting Confusion Matrix (count + row %)')
        fig.tight_layout(); st.pyplot(fig); plt.close()
        _card("Classification errors occur predominantly between adjacent bands (C–D, D–E) where SAP scores straddle threshold boundaries. Band C (the dominant class) achieves the highest recall.", 'blue')

    _divider()
    _sec("Model Accuracy by Property Type")
    type_acc = metrics.get('type_acc', {})
    if type_acc:
        col3, col4 = st.columns(2)
        with col3:
            types_avail = list(type_acc.keys())
            x = np.arange(len(types_avail)); w = 0.25
            fig, ax = plt.subplots(figsize=(7, 4))
            for i, (mdl, color) in enumerate([('LR','#58a6ff'),('RF','#3fb950'),('GB','#d29922')]):
                vals = [type_acc[t][mdl]*100 for t in types_avail]
                bars = ax.bar(x + (i-1)*w, vals, w, label=mdl,
                              color=color, edgecolor='#0d1117', linewidth=0.7)
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                            f'{val:.0f}%', ha='center', fontsize=7.5, fontweight='600', color='#e6edf3')
            ax.set_xticks(x); ax.set_xticklabels(types_avail, fontsize=9)
            ax.set_ylabel('Accuracy (%)', fontsize=9.5)
            ax.set_title('Model Accuracy Breakdown by Property Type')
            ax.set_ylim(50, 100)
            ax.legend(fontsize=9); ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
            fig.tight_layout(); st.pyplot(fig); plt.close()
            _card("All three models perform best on Houses, the dominant class. Lower accuracy on Bungalows and Maisonettes reflects their smaller sample sizes and greater feature heterogeneity.", 'amber')

        with col4:
            _sec("Feature Importances — Top 15 (Random Forest)")
            fi = pd.Series(rf.feature_importances_, index=fcols).nlargest(15).sort_values()
            colors_fi = ['#f78166' if 'PROPERTY_TYPE' in name else
                         '#3fb950' if any(x in name for x in ['WALLS','ROOF','WINDOW','HEAT','LIGHT']) else
                         '#58a6ff' for name in fi.index]
            fig, ax = plt.subplots(figsize=(7, 5))
            bars = ax.barh(fi.index, fi.values, color=colors_fi,
                           edgecolor='#0d1117', linewidth=0.7, height=0.6)
            xlim = fi.max() * 1.4
            ax.set_xlim(0, xlim)
            for bar, val in zip(bars, fi.values):
                ax.text(val + xlim*0.01, bar.get_y()+bar.get_height()/2,
                        f'{val:.4f}', va='center', fontsize=8.5, fontweight='600', color='#e6edf3')
            ax.set_xlabel('Mean Decrease in Impurity', fontsize=9.5)
            ax.set_title('Top 15 Feature Importances\n(orange=type, green=insulation/heating, blue=other)')
            ax.xaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
            fig.tight_layout(); st.pyplot(fig); plt.close()

    _divider()
    _sec("5-Fold Cross-Validation — Training Set Only")
    cv_rf = metrics['cv_rf']; cv_gb = metrics['cv_gb']
    st.dataframe(pd.DataFrame({
        'Model':   ['Random Forest','Gradient Boosting'],
        'Fold 1': [f'{cv_rf[0]*100:.2f}%', f'{cv_gb[0]*100:.2f}%'],
        'Fold 2': [f'{cv_rf[1]*100:.2f}%', f'{cv_gb[1]*100:.2f}%'],
        'Fold 3': [f'{cv_rf[2]*100:.2f}%', f'{cv_gb[2]*100:.2f}%'],
        'Fold 4': [f'{cv_rf[3]*100:.2f}%', f'{cv_gb[3]*100:.2f}%'],
        'Fold 5': [f'{cv_rf[4]*100:.2f}%', f'{cv_gb[4]*100:.2f}%'],
        'Mean':   [f'{cv_rf.mean()*100:.2f}%', f'{cv_gb.mean()*100:.2f}%'],
        'Std Dev':[f'+/-{cv_rf.std()*100:.2f}%', f'+/-{cv_gb.std()*100:.2f}%'],
    }), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ENERGY RATING PREDICTOR
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Energy Rating Predictor":
    _header("Energy Rating Predictor",
            f"Enter building details — model predicts rating for all 4 property types simultaneously",
            "Live Prediction")

    _card("Property Type is <b>not a user input here</b>. Enter the building's physical features once, and the model automatically runs predictions for <b>House, Flat, Bungalow, and Maisonette</b> — showing how the same building characteristics would perform across all property types.", 'blue')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div style="font-size:0.82rem;color:#8b949e;font-weight:600;margin-bottom:0.8rem;text-transform:uppercase;letter-spacing:0.7px;">Property Basics</div>', unsafe_allow_html=True)
        built_form    = st.selectbox("Built Form", ['Mid-Terrace','Semi-Detached','Detached','End-Terrace','Not Recorded'])
        age_group     = st.selectbox("Construction Age", AGE_ORDER)
        tenure        = st.selectbox("Tenure", ['Owner-occupied','Social Rented','Private Rented','Other'])
        floor_area    = st.slider("Floor Area (m²)", 20, 250, 75)
        floor_height  = st.slider("Floor Height (m)", 2.0, 3.5, 2.4, step=0.1)
        hab_rooms     = st.slider("Habitable Rooms", 1, 12, 4)
        heat_rooms    = st.slider("Heated Rooms",    1, 12, 4)
    with col2:
        st.markdown('<div style="font-size:0.82rem;color:#8b949e;font-weight:600;margin-bottom:0.8rem;text-transform:uppercase;letter-spacing:0.7px;">Building Features</div>', unsafe_allow_html=True)
        multi_glaze   = st.slider("Double Glazing (%)",      0, 100, 80)
        low_e_light   = st.slider("Low Energy Lighting (%)", 0, 100, 50)
        ext_count     = st.slider("Extensions",              0, 5,   0)
        wind_t        = st.slider("Wind Turbines",           0, 3,   0)
        photo         = st.slider("Photovoltaic (%)",        0, 100, 0)
        main_fuel     = st.selectbox("Main Fuel", ['mains gas (not community)','electricity (not community)','mains gas (community)','oil','LPG'])
        tariff        = st.selectbox("Energy Tariff", ['standard','off-peak 7 hour','24 hour','off-peak 10 hour'])
        mains_gas     = st.selectbox("Mains Gas?", ['Y','N'])
        solar_water   = st.selectbox("Solar Water Heating?", ['N','Y'])
    with col3:
        st.markdown('<div style="font-size:0.82rem;color:#8b949e;font-weight:600;margin-bottom:0.8rem;text-transform:uppercase;letter-spacing:0.7px;">Assessor Efficiency Ratings</div>', unsafe_allow_html=True)
        walls_eff    = st.selectbox("Wall Insulation",   EFF_ORDER)
        roof_eff     = st.selectbox("Roof Insulation",   EFF_ORDER)
        win_eff      = st.selectbox("Window Glazing",    EFF_ORDER)
        mheat_eff    = st.selectbox("Main Heating",      EFF_ORDER)
        mheatc_eff   = st.selectbox("Heating Controls",  EFF_ORDER)
        hw_eff       = st.selectbox("Hot Water",         EFF_ORDER)
        light_eff    = st.selectbox("Lighting",          EFF_ORDER)
        insp_year    = st.slider("Inspection Year", 2010, 2026, 2025)

    _divider()

    if st.button("Run Prediction for All Property Types", use_container_width=True):

        ALL_TYPES = ['House', 'Flat', 'Bungalow', 'Maisonette']
        rdesc     = {'A':'Excellent','B':'Very Good','C':'Good','D':'Below Average','E':'Poor'}
        band_score= {'A': 93, 'B': 85, 'C': 73, 'D': 62, 'E': 48}

        # Build a prediction for each property type
        type_results = {}
        for pt in ALL_TYPES:
            inp_d = {
                'TOTAL_FLOOR_AREA': floor_area, 'FLOOR_HEIGHT': floor_height,
                'NUMBER_HABITABLE_ROOMS': hab_rooms, 'NUMBER_HEATED_ROOMS': heat_rooms,
                'MULTI_GLAZE_PROPORTION': multi_glaze, 'LOW_ENERGY_LIGHTING': low_e_light,
                'EXTENSION_COUNT': ext_count, 'WIND_TURBINE_COUNT': wind_t, 'PHOTO_SUPPLY': photo,
                'PROPERTY_TYPE': pt, 'BUILT_FORM': built_form,
                'PROPERTY_AGE_GROUP': age_group, 'TENURE_CLEAN': tenure,
                'MAINS_GAS_FLAG': mains_gas, 'SOLAR_WATER_HEATING_FLAG': solar_water,
                'MAIN_FUEL': main_fuel, 'ENERGY_TARIFF': tariff,
                'WALLS_ENERGY_EFF': walls_eff, 'ROOF_ENERGY_EFF': roof_eff,
                'WINDOWS_ENERGY_EFF': win_eff, 'MAINHEAT_ENERGY_EFF': mheat_eff,
                'MAINHEATC_ENERGY_EFF': mheatc_eff, 'HOT_WATER_ENERGY_EFF': hw_eff,
                'LIGHTING_ENERGY_EFF': light_eff, 'INSPECTION_YEAR': insp_year,
            }
            inp = pd.DataFrame([inp_d])
            for col_name, le in le_dict.items():
                if col_name in inp.columns:
                    raw = inp[col_name].astype(str).iloc[0]
                    inp[col_name] = le.transform([raw if raw in le.classes_ else le.classes_[0]])
            for col_name in fcols:
                if col_name not in inp.columns: inp[col_name] = 0
            inp = inp[fcols]

            lbl_gb  = le_tgt.inverse_transform([gb.predict(inp)[0]])[0]
            lbl_rf  = le_tgt.inverse_transform([rf.predict(inp)[0]])[0]
            prob_gb = gb.predict_proba(inp)[0]
            conf_gb = max(prob_gb) * 100
            type_results[pt] = {
                'gb': lbl_gb, 'rf': lbl_rf,
                'conf': conf_gb, 'prob': prob_gb,
                'score': band_score.get(lbl_gb, 70),
            }

        # ── Row 1: 4 prediction boxes ─────────────────────────────────────────
        _sec("Predicted Energy Rating — All Property Types (same building inputs)")
        cols = st.columns(4)
        for col, pt in zip(cols, ALL_TYPES):
            res = type_results[pt]
            c   = RATING_COLORS.get(res['gb'], '#888')
            tc  = TYPE_COLORS.get(pt, '#888')
            agree_txt = "RF agrees" if res['gb'] == res['rf'] else f"RF: {res['rf']}"
            agree_col = "#3fb950" if res['gb'] == res['rf'] else "#d29922"
            with col:
                st.markdown(
                    f'<div class="pred-result" style="background:{c}14;border-color:{c}55;">'
                    f'<div style="font-size:0.7rem;color:{tc};font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:0.8px;margin-bottom:0.4rem;">{pt}</div>'
                    f'<div class="pred-grade" style="color:{c};font-size:4.5rem;">{res["gb"]}</div>'
                    f'<div style="color:#c9d1d9;font-size:0.82rem;margin-top:0.3rem;">'
                    f'{rdesc.get(res["gb"],"")} &nbsp;|&nbsp; {res["conf"]:.0f}% conf.</div>'
                    f'<div style="color:{agree_col};font-size:0.75rem;font-weight:600;margin-top:0.4rem;">'
                    f'{agree_txt}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        _divider()

        # ── Row 2: Confidence comparison bar chart ────────────────────────────
        _sec("Prediction Confidence by Band — All Property Types")
        classes = le_tgt.classes_
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
        for ax, pt in zip(axes, ALL_TYPES):
            res    = type_results[pt]
            probs  = res['prob'][:len(classes)] * 100
            colors_b = [RATING_COLORS.get(c,'#888') for c in classes]
            bars   = ax.bar(classes, probs, color=colors_b,
                            width=0.5, edgecolor='#0d1117', linewidth=0.8)
            for bar, val in zip(bars, probs):
                if val > 1:
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                            f'{val:.0f}%', ha='center', fontsize=8, fontweight='700', color='#e6edf3')
            ax.set_title(f'{pt}\nPredicted: Band {res["gb"]}',
                         fontsize=9.5, fontweight='700',
                         color=TYPE_COLORS.get(pt,'#e6edf3'))
            ax.set_ylim(0, 115)
            ax.set_xlabel('Rating Band', fontsize=8.5)
            ax.yaxis.grid(True, alpha=0.35); ax.set_axisbelow(True)
            if ax != axes[0]: ax.set_ylabel('')
        axes[0].set_ylabel('Probability (%)', fontsize=9)
        plt.suptitle('Gradient Boosting Prediction Confidence per Band — by Property Type',
                     fontsize=11, fontweight='700', y=1.01)
        fig.tight_layout(); st.pyplot(fig); plt.close()

        _divider()

        # ── Row 3: Predicted score vs actual dataset distribution ─────────────
        _sec("Where Does Each Predicted Rating Sit in Liverpool's Actual Distribution?")
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
        for ax, pt in zip(axes, ALL_TYPES):
            res       = type_results[pt]
            pt_data   = df[df['PROPERTY_TYPE'] == pt]['CURRENT_ENERGY_EFFICIENCY']
            pred_sc   = res['score']
            tc        = TYPE_COLORS.get(pt, '#888')
            rc        = RATING_COLORS.get(res['gb'], '#888')
            ax.hist(pt_data, bins=25, color='#21262d', edgecolor='#30363d',
                    linewidth=0.4, alpha=0.95)
            ax.axvline(pred_sc, color=rc, linewidth=2.2, linestyle='--',
                       label=f'Band {res["gb"]} (~{pred_sc})')
            ax.axvline(pt_data.mean(), color='#58a6ff', linewidth=1.5, linestyle=':',
                       label=f'Avg {pt_data.mean():.1f}')
            pct = (pt_data < pred_sc).mean() * 100
            ax.set_title(f'{pt}  |  Band {res["gb"]}\ntop {100-pct:.0f}% of {pt}s',
                         fontsize=9, fontweight='700',
                         color=TYPE_COLORS.get(pt,'#e6edf3'))
            ax.set_xlabel('SAP Score', fontsize=8.5)
            ax.legend(fontsize=7.5, loc='upper left')
            ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        axes[0].set_ylabel('Number of Properties', fontsize=9)
        plt.suptitle('Predicted Score vs Actual Distribution for Each Property Type in Liverpool',
                     fontsize=11, fontweight='700', y=1.01)
        fig.tight_layout(); st.pyplot(fig); plt.close()

        _divider()

        # ── Row 4: Comparison summary table + bar chart ───────────────────────
        _sec("Property Type Comparison Summary")
        col_a, col_b = st.columns([1, 1.4])

        with col_a:
            pt_summary = df.groupby('PROPERTY_TYPE').agg(
                Dataset_Avg=('CURRENT_ENERGY_EFFICIENCY','mean'),
                Avg_CO2=('CO2_EMISSIONS_CURRENT','mean'),
                Avg_Saving=('COST_SAVING_POTENTIAL','mean'),
                Count=('PROPERTY_TYPE','count')
            ).round(2).reset_index()
            pt_summary['Predicted_Band'] = pt_summary['PROPERTY_TYPE'].map(
                {pt: type_results[pt]['gb'] for pt in ALL_TYPES if pt in type_results})
            pt_summary['GB_Confidence']  = pt_summary['PROPERTY_TYPE'].map(
                {pt: f"{type_results[pt]['conf']:.1f}%" for pt in ALL_TYPES if pt in type_results})
            pt_summary.columns = ['Type','Dataset Avg','CO₂ (t/yr)','Saving (£/yr)','n','Predicted Band','Confidence']
            st.dataframe(pt_summary, use_container_width=True, hide_index=True)
            _card(f"With identical building inputs, the model may predict different ratings per property type. This reflects the model learning type-specific patterns from 4,579 Liverpool properties.", 'green')

        with col_b:
            fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
            types_p  = pt_summary['Type'].values
            colors_p = [TYPE_COLORS.get(t,'#888') for t in types_p]

            for ax, (col_key, title, fmt) in zip(axes, [
                ('Dataset Avg', 'Dataset Avg Efficiency', '{:.1f}'),
                ('CO₂ (t/yr)',  'Avg CO₂ / Year',         '{:.2f}t'),
                ('Saving (£/yr)','Avg Cost Saving',        '£{:.0f}'),
            ]):
                vals = pt_summary[col_key].values
                bars = ax.bar(types_p, vals, color=colors_p,
                              width=0.5, edgecolor='#0d1117', linewidth=0.7)
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.01,
                            fmt.format(val), ha='center', fontsize=8, fontweight='600', color='#e6edf3')
                ax.set_title(title, fontsize=9, fontweight='700')
                ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
                ax.tick_params(axis='x', rotation=14, labelsize=7.5)
                ax.set_ylim(0, max(vals)*1.22)
            fig.tight_layout(); st.pyplot(fig); plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 6 — REGRESSION ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Regression Analysis":
    _header("Regression Analysis", "Predicting continuous outcomes — efficiency score and cost saving", "ML Regression")
    _card("Regression models predict <b>continuous numerical values</b>. Two targets: (1) Current Energy Efficiency Score, (2) Cost Saving Potential (£/year). Metrics: MAE, MSE, RMSE, R². The same 25 physical features are used. No SAP-derived data included.", 'blue')

    with st.spinner("Training regression models..."):
        reg_res = train_regression(df)

    tab1, tab2 = st.tabs(["Efficiency Score", "Cost Saving Potential"])

    for tab, tname in zip([tab1, tab2], ['Efficiency Score','Cost Saving (GBP)']):
        with tab:
            res = reg_res[tname]; m = res['metrics']
            best_i = m['R2'].idxmax(); best_m = m.loc[best_i,'Model']
            best_r2 = m.loc[best_i,'R2']; best_rmse = m.loc[best_i,'RMSE']

            col1, col2 = st.columns([1.1,1])
            with col1:
                _sec(f"Regression Metrics — {tname}")
                st.dataframe(
                    m.style
                     .highlight_min(subset=['MAE','MSE','RMSE'], color='#0c1f14')
                     .highlight_max(subset=['R2'],              color='#0c1f14')
                     .highlight_max(subset=['MAE','MSE','RMSE'],color='#200d0d')
                     .highlight_min(subset=['R2'],              color='#200d0d')
                     .format({'MAE':'{:.3f}','MSE':'{:.3f}','RMSE':'{:.3f}','R2':'{:.4f}'}),
                    use_container_width=True, hide_index=True)
                _card(f"<b>{best_m}</b> achieves the best performance: R²={best_r2:.4f}, RMSE={best_rmse:.3f}. Green = best per metric, red = worst. RMSE is expressed in the same units as the target (SAP points or £), making it directly interpretable.", 'green')

                fig, ax = plt.subplots(figsize=(6.5, 4))
                labels = ['Lin. Reg.','Rand. Forest','Grad. Boost']
                x, w = np.arange(3), 0.3
                b1 = ax.bar(x-w/2, m['MAE'].values,  w, label='MAE',
                            color=['#3b4c6b','#1a6b3c','#1f3d6b'], edgecolor='#0d1117', linewidth=0.7)
                b2 = ax.bar(x+w/2, m['RMSE'].values, w, label='RMSE',
                            color=['#58a6ff','#3fb950','#79c0ff'], edgecolor='#0d1117', linewidth=0.7)
                all_vals = [bar.get_height() for bar in list(b1)+list(b2)]
                ax.set_ylim(0, max(all_vals) * 1.28)
                for bar in list(b1)+list(b2):
                    ax.text(bar.get_x()+bar.get_width()/2,
                            bar.get_height() + max(all_vals)*0.025,
                            f'{bar.get_height():.2f}',
                            ha='center', va='bottom', fontsize=8.5,
                            fontweight='600', color='#e6edf3')
                ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
                ax.legend(fontsize=9); ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
                ax.set_ylabel('Error (lower = better)', fontsize=9.5)
                ax.set_title('MAE and RMSE Comparison by Model', pad=12)
                fig.tight_layout(); st.pyplot(fig); plt.close()

            with col2:
                _sec("Actual vs Predicted — Best Model")
                yp = res[{'Linear Regression':'lr_pred','Random Forest':'rf_pred','Gradient Boosting':'gb_pred'}[best_m]]
                yt = res['y_test']
                idx_s = np.random.RandomState(42).choice(len(yt), min(600, len(yt)), replace=False)
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(yt[idx_s], yp[idx_s], alpha=0.35, s=14,
                           c='#58a6ff', edgecolors='none')
                lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
                ax.plot([lo, hi], [lo, hi], '--', color='#f85149', linewidth=1.5, label='Perfect fit')
                ax.set_xlabel(f'Actual {tname}', fontsize=10)
                ax.set_ylabel(f'Predicted {tname}', fontsize=10)
                ax.set_title(f'Actual vs Predicted — {best_m}\nR²={best_r2:.4f} | RMSE={best_rmse:.3f} | n={len(idx_s)}')
                ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
                fig.tight_layout(); st.pyplot(fig); plt.close()
                _card(f"Points clustered close to the diagonal confirm strong predictive power. Residuals widen at extreme values — a common pattern for ensemble regressors on bounded targets.", 'blue')


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 7 — CLUSTERING
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Clustering":
    _header("Clustering Analysis", "K-Means segmentation of Liverpool's housing stock", "Unsupervised ML")
    _card("K-Means groups 4,579 properties into natural segments based on their 25 standardised physical features. The Elbow Method selects K=4. PCA reduces dimensionality to 2D for visualisation.", 'blue')

    with st.spinner("Running K-Means..."):
        inertias, labels, label_names, X2d, name_map, X_sc = run_clustering(df)

    col1, col2 = st.columns(2)

    with col1:
        _sec("Elbow Method — Selecting Optimal K")
        fig, ax = plt.subplots(figsize=(6.5, 4))
        k_range = list(range(2, 11))
        ax.plot(k_range, inertias, 'o-', color='#58a6ff', linewidth=2.2, markersize=7,
                markerfacecolor='#f85149', markeredgecolor='#0d1117', markeredgewidth=1.2)
        ax.axvline(4, color='#d29922', linestyle='--', linewidth=1.5, alpha=0.85, label='K=4 (chosen)')
        for k, v in zip(k_range, inertias):
            ax.text(k, v + max(inertias)*0.012, f'{v:,.0f}', ha='center', fontsize=7.5, color='#8b949e')
        ax.set_xlabel('Number of Clusters (K)', fontsize=10)
        ax.set_ylabel('Within-Cluster Sum of Squares (Inertia)', fontsize=9.5)
        ax.set_title('K-Means Elbow Method')
        ax.legend(fontsize=9); ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        _card("Inertia drops steeply K=2 to K=4, then flattens — the elbow at K=4 indicates four meaningful natural groupings. Adding further clusters yields diminishing returns.", 'green')

    with col2:
        _sec("PCA 2D Cluster Visualisation")
        cl_color_map = dict(zip(CLUSTER_ORDER, CLUSTER_COLORS))
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for cl_name in CLUSTER_ORDER:
            mask = np.array(label_names) == cl_name
            if mask.sum() == 0: continue
            ax.scatter(X2d[mask,0], X2d[mask,1],
                       c=cl_color_map.get(cl_name,'#888'), s=10,
                       alpha=0.45, edgecolors='none', label=f'{cl_name} (n={mask.sum():,})')
        ax.set_xlabel('PCA Component 1', fontsize=9.5)
        ax.set_ylabel('PCA Component 2', fontsize=9.5)
        ax.set_title('K=4 Clusters — PCA Projection')
        ax.legend(fontsize=8, markerscale=2.5)
        ax.grid(True, alpha=0.25)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        _card("Cluster separation is strongest along PCA Component 1, which represents overall building fabric quality. The Low Efficiency cluster occupies the left region; High Efficiency the right.", 'blue')

    _divider()
    _sec("Cluster Composition by Property Type")

    df_cl = df.copy()
    df_cl['CLUSTER_NAME'] = label_names

    col3, col4 = st.columns(2)

    with col3:
        ct_type = pd.crosstab(df_cl['CLUSTER_NAME'], df_cl['PROPERTY_TYPE'])
        ct_type = ct_type.reindex([c for c in CLUSTER_ORDER if c in ct_type.index])
        ct_pct  = ct_type.div(ct_type.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        bottom = np.zeros(len(ct_pct))
        for pt in ct_pct.columns:
            vals = ct_pct[pt].values
            bars = ax.bar(ct_pct.index, vals, bottom=bottom, label=pt,
                          color=TYPE_COLORS.get(pt,'#888'), edgecolor='#0d1117', linewidth=0.5, width=0.55)
            for bar, val in zip(bars, vals):
                if val > 7:
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_y()+bar.get_height()/2,
                            f'{val:.0f}%', ha='center', va='center',
                            fontsize=8.5, fontweight='600', color='white')
            bottom += vals
        ax.set_ylabel('Proportion (%)', fontsize=9.5)
        ax.set_title('Property Type Mix within Each Cluster')
        ax.legend(fontsize=8.5, loc='lower right')
        ax.tick_params(axis='x', rotation=12)
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        _card("Flats dominate the High Efficiency cluster (94.8%), reflecting their compact, heat-retaining form. Bungalows are concentrated in Low Efficiency (89.3%) due to their large exposed roof-to-floor ratio and pre-war construction.", 'amber')

    with col4:
        profile_cols = ['CURRENT_ENERGY_EFFICIENCY','CO2_EMISSIONS_CURRENT',
                        'HEATING_COST_CURRENT','COST_SAVING_POTENTIAL','TOTAL_FLOOR_AREA']
        profile = df_cl.groupby('CLUSTER_NAME')[profile_cols].mean().round(2)
        cnt     = df_cl['CLUSTER_NAME'].value_counts().rename('Count')
        profile = profile.join(cnt).reindex([c for c in CLUSTER_ORDER if c in profile.index])

        _sec("Cluster Profiles")
        st.dataframe(profile, use_container_width=True)

        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        eff_vals = profile['CURRENT_ENERGY_EFFICIENCY'].values
        cl_names = profile.index.tolist()
        bars = ax.barh(cl_names, eff_vals,
                       color=[cl_color_map.get(c,'#888') for c in cl_names],
                       height=0.45, edgecolor='#0d1117', linewidth=0.8)
        for bar, val in zip(bars, eff_vals):
            ax.text(val+0.3, bar.get_y()+bar.get_height()/2,
                    f'{val:.1f}', va='center', fontsize=10, fontweight='700', color='#e6edf3')
        ax.set_xlabel('Average SAP Efficiency Score', fontsize=9.5)
        ax.set_title('Average Efficiency per Cluster')
        ax.xaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
        fig.tight_layout(); st.pyplot(fig); plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 8 — RECOMMENDATION SYSTEM
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Recommendation System":
    _header("Recommendation System", "Frequency-based and content-based improvement recommendations", "RecSys")
    _card("Two complementary methods: <b>Part A — Frequency-Based</b> identifies the most common improvements across 10,297 EPC records, filterable by property type. <b>Part B — Content-Based</b> uses cosine similarity on the 25 building features to surface recommendations from structurally similar properties.", 'blue')

    with st.spinner("Loading recommendation data..."):
        recs_df = load_recs()

    tab_a, tab_b = st.tabs(["Part A: Frequency-Based", "Part B: Content-Based"])

    with tab_a:
        col_filter, _ = st.columns([1, 2])
        with col_filter:
            pt_filter = st.selectbox("Filter by Property Type",
                                     ['All Types'] + sorted(recs_df['PROPERTY_TYPE'].dropna().unique().tolist()))

        filtered = recs_df if pt_filter == 'All Types' else recs_df[recs_df['PROPERTY_TYPE'] == pt_filter]

        # ── Top 12 — full width ───────────────────────────────────────────────
        _sec(f"Top 12 Most Frequent Improvements — {pt_filter}")
        top = filtered['IMPROVEMENT_SUMMARY_TEXT'].value_counts().head(12)
        fig, ax = plt.subplots(figsize=(13, 5.5))
        colors_r = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(top)))[::-1]
        bars = ax.barh(range(len(top)), top.values, color=colors_r,
                       height=0.65, edgecolor='#0d1117', linewidth=0.5)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, fontsize=10)
        for bar, val in zip(bars, top.values):
            pct = val / len(filtered) * 100
            ax.text(val + top.max()*0.008, bar.get_y()+bar.get_height()/2,
                    f'{val:,}  ({pct:.1f}%)', va='center',
                    fontsize=9.5, fontweight='600', color='#e6edf3')
        ax.set_xlabel('Number of Recommendation Records', fontsize=10)
        ax.set_title(f'Most Frequently Recommended Energy Improvements — {pt_filter}',
                     fontsize=13, fontweight='700', pad=14)
        ax.set_xlim(0, top.max()*1.32)
        ax.xaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        _card("Solar photovoltaic panels are the most recommended improvement city-wide (28.9% of all records), driven by the large stock of solid-walled pre-1950 properties unable to benefit from cavity fill. Internal wall insulation and floor insulation follow.", 'blue')

        _divider()

        # ── By rating band — full width, tall ─────────────────────────────────
        _sec(f"Top Improvements by Energy Rating Band — {pt_filter}")
        bands = [b for b in ['E','D','C','B'] if b in filtered['CURRENT_ENERGY_RATING'].values]
        n_b   = len(bands)
        if n_b > 0:
            fig, axes = plt.subplots(1, n_b, figsize=(5.5*n_b, 6))
            if n_b == 1: axes = [axes]
            bc_map = {'E':'#f85149','D':'#d29922','C':'#d29922','B':'#3fb950'}
            for ax, band in zip(axes, bands):
                b_data = filtered[filtered['CURRENT_ENERGY_RATING']==band]
                top_b  = b_data['IMPROVEMENT_SUMMARY_TEXT'].value_counts().head(6)
                bc     = bc_map.get(band,'#58a6ff')
                bars   = ax.barh(range(len(top_b)), top_b.values, color=bc,
                                 height=0.65, edgecolor='#0d1117', linewidth=0.5, alpha=0.9)
                ax.set_yticks(range(len(top_b)))
                ax.set_yticklabels(
                    [t[:35]+'...' if len(t)>35 else t for t in top_b.index],
                    fontsize=9)
                for bar, val in zip(bars, top_b.values):
                    ax.text(val + top_b.max()*0.025,
                            bar.get_y()+bar.get_height()/2,
                            f'{val:,}', va='center',
                            fontsize=9.5, fontweight='700', color='#e6edf3')
                ax.set_title(f'Band {band}  |  n={len(b_data):,}',
                             fontsize=12, fontweight='700',
                             color=bc_map.get(band,'#e6edf3'), pad=10)
                ax.set_xlim(0, top_b.max()*1.4)
                ax.set_xlabel('Frequency', fontsize=9.5)
                ax.xaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
            plt.suptitle(f'Top 6 Improvements by Current Energy Rating Band — {pt_filter}',
                         fontsize=13, fontweight='700', y=1.01)
            fig.tight_layout(); st.pyplot(fig); plt.close()
            _card("Band E properties are consistently directed toward structural insulation (walls, floors). As properties improve toward Band B, recommendations shift to solar generation and smart heating controls — confirming a logical retrofit pathway.", 'green')

    with tab_b:
        _sec("Content-Based Filtering — Find Similar Properties")
        _card("Select a property index. The system finds its 5 nearest neighbours using cosine similarity on 25 physical features (via NearestNeighbors, brute-force cosine metric) and surfaces their most frequent improvement recommendations.", 'blue')

        with st.spinner("Building similarity model..."):
            _, lbl_cbf, lnames_cbf, _, nm_cbf, Xs_cbf = run_clustering(df)
            nn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
            nn.fit(Xs_cbf)

        col1, col2 = st.columns([1, 2])

        with col1:
            prop_idx = st.number_input("Property Index", min_value=0, max_value=len(df)-1, value=100, step=1)
            row = df.iloc[int(prop_idx)]
            for label, key in [('Property Type','PROPERTY_TYPE'),('Age Band','PROPERTY_AGE_GROUP'),
                                ('Floor Area','TOTAL_FLOOR_AREA'),('Energy Rating','CURRENT_ENERGY_RATING'),
                                ('Efficiency Score','CURRENT_ENERGY_EFFICIENCY'),('Cluster',None)]:
                val = lnames_cbf[int(prop_idx)] if key is None else row.get(key,'—')
                if key == 'TOTAL_FLOOR_AREA': val = f'{val:.1f} m²'
                if key == 'CURRENT_ENERGY_EFFICIENCY': val = f'{val:.1f}'
                st.markdown(f'<div class="card card-blue" style="padding:0.45rem 0.8rem;margin:0.2rem 0;">'
                            f'<b>{label}:</b> {val}</div>', unsafe_allow_html=True)

        with col2:
            dist, idx = nn.kneighbors(Xs_cbf[int(prop_idx)].reshape(1,-1))
            nb_idx = idx[0][1:]
            nb_df  = df.iloc[nb_idx]
            sim_d  = pd.DataFrame({
                'Type':      nb_df['PROPERTY_TYPE'].values,
                'Age':       nb_df['PROPERTY_AGE_GROUP'].values,
                'Floor (m²)':nb_df['TOTAL_FLOOR_AREA'].round(1).values,
                'Rating':    nb_df['CURRENT_ENERGY_RATING'].values,
                'Efficiency':nb_df['CURRENT_ENERGY_EFFICIENCY'].round(1).values,
                'Similarity':[f'{(1-d)*100:.1f}%' for d in dist[0][1:]],
            })
            sim_d.index = range(1, len(sim_d)+1)
            st.dataframe(sim_d, use_container_width=True)

        nb_ratings = nb_df['CURRENT_ENERGY_RATING'].values
        combined_recs = []
        for rating in nb_ratings:
            band_recs = recs_df[recs_df['CURRENT_ENERGY_RATING']==rating]['IMPROVEMENT_SUMMARY_TEXT'].tolist()
            combined_recs.extend(band_recs)
        if not combined_recs:
            combined_recs = recs_df['IMPROVEMENT_SUMMARY_TEXT'].tolist()
        top_cbf = pd.Series(combined_recs).value_counts().head(8)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        colors_cbf = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_cbf)))[::-1]
        bars = ax.barh(range(len(top_cbf)), top_cbf.values, color=colors_cbf,
                       height=0.6, edgecolor='#0d1117', linewidth=0.5)
        ax.set_yticks(range(len(top_cbf)))
        ax.set_yticklabels(top_cbf.index, fontsize=9)
        for bar, val in zip(bars, top_cbf.values):
            pct = val / sum(top_cbf.values) * 100
            ax.text(val+top_cbf.max()*0.01, bar.get_y()+bar.get_height()/2,
                    f'{val} ({pct:.0f}%)', va='center', fontsize=9, fontweight='600', color='#e6edf3')
        ax.set_xlabel('Frequency among similar properties', fontsize=9.5)
        ax.set_title(f'Content-Based Recommendations for Property {int(prop_idx)}'
                     f' | Cluster: {lnames_cbf[int(prop_idx)]}')
        ax.set_xlim(0, top_cbf.max()*1.35)
        ax.xaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        _card(f"Recommendations are derived from the 5 properties most similar to Property {int(prop_idx)} in physical characteristics. The frequency rank surfaces the most consensus-supported improvements for this specific building profile.", 'green')


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 9 — CONCLUSIONS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Conclusions":
    _header("Conclusions and Recommendations", "Key findings and policy implications", "COM6003")

    pt_eff = df.groupby('PROPERTY_TYPE')['CURRENT_ENERGY_EFFICIENCY'].mean()
    pre1900 = df[df['PROPERTY_AGE_GROUP']=='Pre-1900']['CURRENT_ENERGY_EFFICIENCY'].mean()
    post2021= df[df['PROPERTY_AGE_GROUP']=='Post-2021']['CURRENT_ENERGY_EFFICIENCY'].mean()
    avg_sav = df['COST_SAVING_POTENTIAL'].mean()

    col1, col2 = st.columns(2)

    with col1:
        _sec("Key Analytical Findings")
        findings = [
            ("Housing Stock Profile",
             f"4,579 properties assessed. 68.6% rated Band C. Houses dominate (63.7%), with 44.1% of all stock built 1900–1949.",
             'blue'),
            ("Property Type vs Efficiency",
             f"Maisonettes lead at 75.4 SAP points; Bungalows lowest at 72.4. Houses produce 2.39t CO₂/year — 86% more than Flats (1.29t) due to larger floor area and greater heat loss surface.",
             'green'),
            ("Construction Era Impact",
             f"Pre-1900 properties average {pre1900:.1f} SAP points vs {post2021:.1f} for Post-2021 — an {post2021-pre1900:.1f}-point gap. Progressive building regulation is the primary driver.",
             'amber'),
            ("Classification Performance",
             f"Gradient Boosting achieved {metrics['gb_acc']*100:.2f}% test accuracy and {gb_cv*100:.2f}% cross-validation accuracy from 25 physical features, with no SAP-derived leakage.",
             'blue'),
            ("Regression Insights",
             "Gradient Boosting outperforms Linear Regression on both regression targets (efficiency score and cost saving), confirming non-linear relationships in EPC data.",
             'green'),
            ("Clustering — 4 Segments",
             "K-Means identified 4 housing segments. Flats dominate High Efficiency (94.8%); Bungalows concentrate in Low Efficiency (89.3%). 62.7% of all properties fall in 'Below Average' — the primary policy target.",
             'amber'),
        ]
        for title, body, variant in findings:
            _card(f"<b>{title}:</b> {body}", variant)

    with col2:
        _sec("Policy Recommendations")
        policies = [
            ("1. Retrofit Pre-1950 Houses",
             "Over 2,700 properties built before 1950 are predominantly Houses with solid walls and no loft insulation. A targeted cavity-fill and loft programme would yield the highest city-wide efficiency gain.",
             'red'),
            ("2. Prioritise Bungalows for Roof Insulation",
             "Bungalows show the lowest average efficiency (72.4) and are overwhelmingly in the Low Efficiency cluster. Their single-storey form means the roof represents the majority of heat loss surface.",
             'amber'),
            ("3. Target Below Average Cluster",
             "2,869 properties (62.7%) in the Below Average cluster represent the single largest improvement opportunity. Data-driven targeting using the ML classifier can identify these properties from administrative records.",
             'blue'),
            ("4. House-Specific CO₂ Programme",
             "Houses emit nearly double the CO₂ of Flats. A House-specific retrofit programme combining wall insulation and heating modernisation would disproportionately reduce Liverpool's residential carbon footprint.",
             'green'),
            (f"5. Deploy Predictive Model ({metrics['gb_acc']*100:.1f}% accuracy)",
             "The Gradient Boosting classifier can triage low-performing properties before physical assessment — reducing programme delivery costs and enabling better targeting of limited retrofit funding.",
             'blue'),
        ]
        for title, body, variant in policies:
            _card(f"<b style='color:#e6edf3;'>{title}</b><br><span style='font-size:0.85rem;'>{body}</span>", variant)

        _divider()
        _sec("City-Wide Savings Potential")
        total_sav = avg_sav * len(df)
        st.markdown(f"""
        <div class="kpi-row">
          <div class="kpi"><div class="kpi-val">£{avg_sav:.0f}</div><div class="kpi-label">Avg Saving/Property/Year</div></div>
          <div class="kpi"><div class="kpi-val">£{total_sav/1e6:.2f}M</div><div class="kpi-label">City-Wide Annual Potential</div></div>
          <div class="kpi"><div class="kpi-val">2,869</div><div class="kpi-label">Below Average Properties</div></div>
        </div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#6e7681;font-size:0.75rem;padding:2rem 0 1rem 0;border-top:1px solid #21262d;margin-top:2.5rem;">
    Liverpool EPC Analysis &nbsp;&middot;&nbsp; COM6003 Data Science &nbsp;&middot;&nbsp;
    Buckinghamshire New University &nbsp;&middot;&nbsp; 2025–26
</div>""", unsafe_allow_html=True)
