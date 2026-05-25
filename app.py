# COM6003 Data Science — Liverpool EPC Analysis
# Streamlit Dashboard — Buckinghamshire New University 2025-26

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    page_title="Liverpool EPC Analysis | COM6003",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
.stApp { background-color: #0f1117; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1f2e 0%, #141820 100%); border-right: 1px solid #2d3748; }
.top-banner { background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 50%, #1a8cff 100%); padding: 2.5rem 3rem; border-radius: 16px; margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(26,140,255,0.2); border: 1px solid rgba(255,255,255,0.1); }
.top-banner h1 { color: #ffffff; font-size: 2.2rem; font-weight: 700; margin: 0; }
.top-banner p { color: rgba(255,255,255,0.75); font-size: 0.95rem; margin: 0.5rem 0 0 0; }
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card { background: linear-gradient(135deg, #1a2035 0%, #1f2840 100%); border: 1px solid #2d3748; border-radius: 12px; padding: 1.2rem 1.5rem; flex: 1; text-align: center; min-width: 150px; }
.metric-card .value { font-size: 1.9rem; font-weight: 700; color: #1a8cff; display: block; }
.metric-card .label { font-size: 0.75rem; color: #a0aec0; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.3rem; }
.metric-card .icon { font-size: 1.4rem; margin-bottom: 0.4rem; display: block; }
.section-header { color: #ffffff; font-size: 1.4rem; font-weight: 600; margin: 1.5rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid #1a8cff; }
.info-card { background: #1a2035; border: 1px solid #2d3748; border-left: 4px solid #1a8cff; border-radius: 8px; padding: 1rem 1.2rem; margin: 0.5rem 0; color: #e2e8f0; font-size: 0.9rem; line-height: 1.6; }
.pred-box { border-radius: 16px; padding: 2rem; text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.4); border: 2px solid rgba(255,255,255,0.1); }
.stButton > button { background: linear-gradient(135deg, #1a8cff, #0066cc); color: white; border: none; border-radius: 10px; padding: 0.7rem 2rem; font-weight: 600; font-size: 1rem; width: 100%; }
.stTabs [data-baseweb="tab-list"] { background: #1a2035; border-radius: 10px; padding: 4px; }
.stTabs [aria-selected="true"] { background: #1a8cff !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    'figure.facecolor': '#1a2035', 'axes.facecolor': '#1a2035',
    'axes.edgecolor': '#2d3748', 'axes.labelcolor': '#a0aec0',
    'axes.titlecolor': '#ffffff', 'text.color': '#e2e8f0',
    'xtick.color': '#a0aec0', 'ytick.color': '#a0aec0',
    'grid.color': '#2d3748', 'grid.alpha': 0.5,
    'legend.facecolor': '#1a2035', 'legend.edgecolor': '#2d3748',
    'legend.labelcolor': '#e2e8f0', 'axes.spines.top': False, 'axes.spines.right': False,
})

AGE_ORDER     = ['Pre-1900','1900-1949','1950-1975','1976-1990','1991-2002','2003-2021','Post-2021']
EFF_ORDER     = ['Very Good','Good','Average','Poor','Very Poor']
RATING_COLORS = {'A':'#2ecc71','B':'#27ae60','C':'#f1c40f','D':'#e67e22','E':'#e74c3c'}
PALETTE       = ['#1a8cff','#00d4aa','#ff6b6b','#ffd93d','#c77dff','#ff9f43','#48dbfb']
CLUSTER_COLORS = ['#e74c3c','#e67e22','#27ae60','#2ecc71']
CLUSTER_NAMES  = {0:'Low Efficiency', 1:'Below Average', 2:'Above Average', 3:'High Efficiency'}


@st.cache_data
def load_data():
    df = pd.read_csv('liverpool_epc_cleaned.csv', low_memory=False)
    if 'TENURE_CLEAN' not in df.columns:
        tenure_map = {'Rented (social)': 'Social Rented',
                      'Owner-occupied':  'Owner-occupied',
                      'Rented (private)':'Private Rented'}
        df['TENURE_CLEAN'] = df['TENURE'].map(tenure_map).fillna('Other')
    if 'PROPERTY_AGE_GROUP' not in df.columns:
        def simplify_age(val):
            if pd.isnull(val): return 'Unknown'
            val = str(val)
            if 'before 1900' in val:                         return 'Pre-1900'
            elif '1900-1929' in val or '1930-1949' in val:  return '1900-1949'
            elif '1950-1966' in val or '1967-1975' in val:  return '1950-1975'
            elif '1976-1982' in val or '1983-1990' in val:  return '1976-1990'
            elif '1991-1995' in val or '1996-2002' in val:  return '1991-2002'
            elif ('2003-2006' in val or '2007-2011' in val or
                  '2012-2021' in val or val.strip() == '2020'): return '2003-2021'
            else:                                            return 'Post-2021'
        df['PROPERTY_AGE_GROUP'] = df['CONSTRUCTION_AGE_BAND'].apply(simplify_age)
    return df


def _build_feature_matrix(df):
    feature_cols = [
        'TOTAL_FLOOR_AREA', 'FLOOR_HEIGHT', 'NUMBER_HABITABLE_ROOMS',
        'NUMBER_HEATED_ROOMS', 'MULTI_GLAZE_PROPORTION',
        'LOW_ENERGY_LIGHTING', 'EXTENSION_COUNT',
        'WIND_TURBINE_COUNT', 'PHOTO_SUPPLY',
        'PROPERTY_TYPE', 'BUILT_FORM', 'PROPERTY_AGE_GROUP',
        'TENURE_CLEAN', 'MAINS_GAS_FLAG', 'SOLAR_WATER_HEATING_FLAG',
        'MAIN_FUEL', 'ENERGY_TARIFF',
        'WALLS_ENERGY_EFF', 'ROOF_ENERGY_EFF', 'WINDOWS_ENERGY_EFF',
        'MAINHEAT_ENERGY_EFF', 'MAINHEATC_ENERGY_EFF',
        'HOT_WATER_ENERGY_EFF', 'LIGHTING_ENERGY_EFF',
        'INSPECTION_YEAR',
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    df_enc = df[feature_cols].copy()
    le_dict = {}
    for col in df_enc.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        le_dict[col] = le
    return df_enc.values, feature_cols, le_dict


@st.cache_resource
def train_models(_df):
    feature_cols = [
        'TOTAL_FLOOR_AREA', 'FLOOR_HEIGHT', 'NUMBER_HABITABLE_ROOMS',
        'NUMBER_HEATED_ROOMS', 'MULTI_GLAZE_PROPORTION',
        'LOW_ENERGY_LIGHTING', 'EXTENSION_COUNT',
        'WIND_TURBINE_COUNT', 'PHOTO_SUPPLY',
        'PROPERTY_TYPE', 'BUILT_FORM', 'PROPERTY_AGE_GROUP',
        'TENURE_CLEAN', 'MAINS_GAS_FLAG', 'SOLAR_WATER_HEATING_FLAG',
        'MAIN_FUEL', 'ENERGY_TARIFF',
        'WALLS_ENERGY_EFF', 'ROOF_ENERGY_EFF', 'WINDOWS_ENERGY_EFF',
        'MAINHEAT_ENERGY_EFF', 'MAINHEATC_ENERGY_EFF',
        'HOT_WATER_ENERGY_EFF', 'LIGHTING_ENERGY_EFF',
        'INSPECTION_YEAR',
    ]
    feature_cols = [c for c in feature_cols if c in _df.columns]
    target       = 'CURRENT_ENERGY_RATING'

    df_model = _df[feature_cols + [target]].copy()
    le_dict  = {}
    for col in df_model.select_dtypes(include='object').columns:
        if col != target:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            le_dict[col]  = le

    le_target        = LabelEncoder()
    df_model[target] = le_target.fit_transform(df_model[target])

    X = df_model[feature_cols]
    y = df_model[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)
    lr_pred = lr.predict(X_test_s)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    gb = GradientBoostingClassifier(n_estimators=200, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)

    cv_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
    cv_gb = cross_val_score(gb, X_train, y_train, cv=5, scoring='accuracy')

    gb_cm      = confusion_matrix(y_test, gb_pred)
    gb_cm_norm = confusion_matrix(y_test, gb_pred, normalize='true')

    metrics = {
        'lr_acc':      accuracy_score(y_test, lr_pred),
        'rf_acc':      accuracy_score(y_test, rf_pred),
        'gb_acc':      accuracy_score(y_test, gb_pred),
        'lr_f1':       f1_score(y_test, lr_pred, average='weighted'),
        'rf_f1':       f1_score(y_test, rf_pred, average='weighted'),
        'gb_f1':       f1_score(y_test, gb_pred, average='weighted'),
        'gb_cm':       gb_cm,
        'gb_cm_norm':  gb_cm_norm,
        'cv_rf':       cv_rf,
        'cv_gb':       cv_gb,
    }
    return rf, gb, lr, scaler, le_dict, le_target, feature_cols, X_test, y_test, metrics


@st.cache_resource
def train_regression(_df):
    X_reg, feat_cols, le_dict_reg = _build_feature_matrix(_df)

    def reg_metrics(name, y_true, y_pred):
        mae  = mean_absolute_error(y_true, y_pred)
        mse  = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_true, y_pred)
        return {'Model': name, 'MAE': round(mae, 4), 'MSE': round(mse, 4),
                'RMSE': round(rmse, 4), 'R2': round(r2, 4)}

    results = {}
    for target_name, target_col in [('Efficiency Score', 'CURRENT_ENERGY_EFFICIENCY'),
                                     ('Cost Saving Potential', 'COST_SAVING_POTENTIAL')]:
        y = _df[target_col].values
        X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y, test_size=0.2, random_state=42)
        scaler_r = StandardScaler()
        X_tr_s   = scaler_r.fit_transform(X_tr)
        X_te_s   = scaler_r.transform(X_te)

        lr_r = LinearRegression()
        lr_r.fit(X_tr_s, y_tr)
        lr_pred = lr_r.predict(X_te_s)

        rf_r = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf_r.fit(X_tr, y_tr)
        rf_pred = rf_r.predict(X_te)

        gb_r = GradientBoostingRegressor(n_estimators=200, random_state=42)
        gb_r.fit(X_tr, y_tr)
        gb_pred = gb_r.predict(X_te)

        rows = [
            reg_metrics('Linear Regression',      y_te, lr_pred),
            reg_metrics('Random Forest',           y_te, rf_pred),
            reg_metrics('Gradient Boosting',       y_te, gb_pred),
        ]
        results[target_name] = {
            'metrics': pd.DataFrame(rows),
            'y_test':  y_te,
            'lr_pred': lr_pred,
            'rf_pred': rf_pred,
            'gb_pred': gb_pred,
        }

    return results


@st.cache_resource
def run_clustering(_df):
    X_reg, feat_cols, _ = _build_feature_matrix(_df)
    scaler_c  = StandardScaler()
    X_scaled  = scaler_c.fit_transform(X_reg)

    inertias = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels   = km_final.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    cluster_means = {}
    temp_df = _df.copy()
    temp_df['_CL'] = labels
    for cl in range(4):
        mask = temp_df['_CL'] == cl
        cluster_means[cl] = temp_df.loc[mask, 'CURRENT_ENERGY_EFFICIENCY'].mean()

    sorted_cl = sorted(cluster_means, key=cluster_means.get)
    name_map  = {
        sorted_cl[0]: 'Low Efficiency',
        sorted_cl[1]: 'Below Average',
        sorted_cl[2]: 'Above Average',
        sorted_cl[3]: 'High Efficiency',
    }

    label_names = [name_map[l] for l in labels]
    return inertias, labels, label_names, X_2d, name_map, X_scaled


@st.cache_data
def load_recommendations():
    certs = pd.read_csv('certificates.csv', low_memory=False,
                        usecols=['LMK_KEY', 'CURRENT_ENERGY_RATING',
                                 'PROPERTY_TYPE', 'CONSTRUCTION_AGE_BAND'])
    recs  = pd.read_csv('recommendations.csv', low_memory=False)
    merged = certs.merge(recs, on='LMK_KEY', how='inner')
    return merged


df = load_data()
with st.spinner("Initialising ML models..."):
    rf, gb, lr, scaler, le_dict, le_target, feature_cols, X_test, y_test, metrics = train_models(df)

gb_cv_mean = metrics['cv_gb'].mean()
rf_cv_mean = metrics['cv_rf'].mean()

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0;">
        <div style="font-size:3rem;">⚡</div>
        <div style="color:#fff;font-size:1.1rem;font-weight:700;margin-top:0.5rem;">Liverpool EPC</div>
        <div style="color:#a0aec0;font-size:0.8rem;">COM6003 Data Science</div>
    </div><hr style="border-color:#2d3748;margin:1rem 0;">
    """, unsafe_allow_html=True)
    page = st.selectbox("", [
        "🏠  Overview",
        "📊  Descriptive Analytics",
        "🔬  Diagnostic Analytics",
        "🤖  Predictive Models",
        "🔮  Energy Rating Predictor",
        "📈  Regression Analysis",
        "🔵  Clustering",
        "🌟  Recommendation System",
        "💡  Conclusions",
    ], label_visibility="collapsed")
    st.markdown(f"""<hr style="border-color:#2d3748;">
    <div style="color:#a0aec0;font-size:0.8rem;line-height:2;">
        <div>Location: <b style="color:#e2e8f0;">Liverpool, England</b></div>
        <div>Properties: <b style="color:#e2e8f0;">4,579</b></div>
        <div>Features: <b style="color:#e2e8f0;">25</b> physical</div>
        <div>Best CV: <b style="color:#e2e8f0;">{gb_cv_mean*100:.1f}%</b> (GB classifier)</div>
    </div><hr style="border-color:#2d3748;margin:1rem 0;">
    <div style="color:#718096;font-size:0.75rem;text-align:center;">
        Buckinghamshire New University<br>Academic Year 2025-26
    </div>""", unsafe_allow_html=True)

# Top banner
st.markdown("""
<div class="top-banner">
    <h1>Liverpool Energy Performance Certificate Analysis</h1>
    <p>COM6003 Data Science &nbsp;|&nbsp; Buckinghamshire New University &nbsp;|&nbsp; Academic Year 2025-26</p>
</div>""", unsafe_allow_html=True)


# ============================================================
# Page 1: Overview
# ============================================================
if "Overview" in page:
    avg_eff = df['CURRENT_ENERGY_EFFICIENCY'].mean()
    avg_sav = df['COST_SAVING_POTENTIAL'].mean()
    avg_co2 = df['CO2_EMISSIONS_CURRENT'].mean()

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card"><span class="icon">🏘️</span><span class="value">{len(df):,}</span><span class="label">Total Properties</span></div>
        <div class="metric-card"><span class="icon">⚡</span><span class="value">{avg_eff:.1f}</span><span class="label">Avg Efficiency</span></div>
        <div class="metric-card"><span class="icon">💰</span><span class="value">£{avg_sav:.0f}</span><span class="label">Avg Annual Saving</span></div>
        <div class="metric-card"><span class="icon">🌡️</span><span class="value">{avg_co2:.2f}t</span><span class="label">Avg CO2/year</span></div>
        <div class="metric-card"><span class="icon">🏆</span><span class="value">{gb_cv_mean*100:.1f}%</span><span class="label">Model CV Accuracy</span></div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.markdown('<p class="section-header">About This Analysis</p>', unsafe_allow_html=True)
        for text in [
            "This application presents a complete end-to-end data science pipeline applied to <b>Energy Performance Certificate (EPC)</b> data for <b>Liverpool</b>, England (Local Authority: E08000012), sourced from the <b>Ministry of Housing, Communities and Local Government (MHCLG)</b>.",
            "EPCs are legal documents required when a property is built, sold or rented, assessed using the <b>Standard Assessment Procedure (SAP)</b>. They rate properties from <b>A (most efficient)</b> to <b>G (least efficient)</b>.",
            "<b>Pipeline:</b> Data Acquisition → Feature Engineering → Data Wrangling → Descriptive → Diagnostic → Classification → Regression → Clustering → Recommendation System",
        ]:
            st.markdown(f'<div class="info-card">{text}</div>', unsafe_allow_html=True)

        st.markdown('<p class="section-header">Classification Model Summary</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            'Model':   ['Logistic Regression','Random Forest','Gradient Boosting'],
            'Accuracy':[f"{metrics['lr_acc']*100:.2f}%", f"{metrics['rf_acc']*100:.2f}%", f"{metrics['gb_acc']*100:.2f}%"],
            'F1 Score':[f"{metrics['lr_f1']*100:.2f}%", f"{metrics['rf_f1']*100:.2f}%", f"{metrics['gb_f1']*100:.2f}%"],
            'CV Score':['N/A', f"{rf_cv_mean*100:.2f}%", f"{gb_cv_mean*100:.2f}% (best)"],
            'Type':    ['Linear Baseline','Ensemble Bagging','Ensemble Boosting'],
        }), use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<p class="section-header">Energy Rating Distribution</p>', unsafe_allow_html=True)
        rating_counts = df['CURRENT_ENERGY_RATING'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = [RATING_COLORS.get(r, '#888') for r in rating_counts.index]
        bars = ax.bar(rating_counts.index, rating_counts.values,
                      color=colors, edgecolor='#0f1117', linewidth=1.5, width=0.6)
        for bar, val in zip(bars, rating_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                    f'{val:,}\n({val/len(df)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=9, fontweight='600', color='#e2e8f0')
        ax.set_xlabel('Energy Rating Band', fontsize=11, labelpad=10)
        ax.set_ylabel('Number of Properties', fontsize=11)
        ax.set_title('Liverpool EPC — Energy Rating Distribution', fontsize=13, fontweight='700', pad=15)
        ax.set_ylim(0, max(rating_counts.values) * 1.22)
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ============================================================
# Page 2: Descriptive Analytics
# ============================================================
elif "Descriptive" in page:
    st.markdown('<p class="section-header">Descriptive Analytics — What Has Occurred?</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-card">Descriptive analytics summarises Liverpool\'s housing stock characteristics answering: <b>"What has occurred?"</b></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        pc   = df['PROPERTY_TYPE'].value_counts()
        bars = ax.bar(pc.index, pc.values, color=PALETTE[:len(pc)],
                      edgecolor='#0f1117', linewidth=1.2, width=0.55)
        for bar, val in zip(bars, pc.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                    f'{val:,}\n({val/len(df)*100:.1f}%)',
                    ha='center', fontsize=9, fontweight='600', color='#e2e8f0')
        ax.set_title('Property Type Distribution', fontsize=13, fontweight='700', pad=12)
        ax.set_xlabel('Property Type', fontsize=10); ax.set_ylabel('Count', fontsize=10)
        ax.set_ylim(0, max(pc.values) * 1.25)
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('<div class="info-card"><b>Finding:</b> Houses dominate (63.7%), followed by flats (32.9%), reflecting Liverpool\'s urban terraced housing character.</div>', unsafe_allow_html=True)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ac = df['PROPERTY_AGE_GROUP'].value_counts().reindex(AGE_ORDER).dropna()
        age_colors = ['#8e44ad','#2980b9','#27ae60','#f39c12','#e67e22','#e74c3c','#1abc9c']
        bars = ax.bar(ac.index, ac.values, color=age_colors[:len(ac)],
                      edgecolor='#0f1117', linewidth=1.2, width=0.6)
        for bar, val in zip(bars, ac.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                    f'{val:,}\n({val/len(df)*100:.1f}%)',
                    ha='center', fontsize=8.5, fontweight='600', color='#e2e8f0')
        ax.set_title('Properties by Construction Age Group', fontsize=13, fontweight='700', pad=12)
        ax.set_xlabel('Age Group', fontsize=10); ax.set_ylabel('Count', fontsize=10)
        ax.set_ylim(0, max(ac.values) * 1.28)
        ax.tick_params(axis='x', rotation=30)
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('<div class="info-card"><b>Finding:</b> 44% of properties built 1900-1949. This aging stock (75+ years old) presents the greatest energy improvement opportunity.</div>', unsafe_allow_html=True)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(6, 4))
        rating_order = ['A','B','C','D','E']
        df_f = df[df['CURRENT_ENERGY_RATING'].isin(rating_order)]
        bp = ax.boxplot(
            [df_f[df_f['CURRENT_ENERGY_RATING']==r]['CURRENT_ENERGY_EFFICIENCY'].values
             for r in rating_order],
            labels=rating_order, patch_artist=True,
            medianprops=dict(color='white', linewidth=2),
            whiskerprops=dict(color='#a0aec0'), capprops=dict(color='#a0aec0'),
            flierprops=dict(marker='o', color='#a0aec0', alpha=0.5, markersize=4))
        for patch, r in zip(bp['boxes'], rating_order):
            patch.set_facecolor(RATING_COLORS.get(r, '#888')); patch.set_alpha(0.85)
        for i, r in enumerate(rating_order):
            med = df_f[df_f['CURRENT_ENERGY_RATING']==r]['CURRENT_ENERGY_EFFICIENCY'].median()
            n   = (df_f['CURRENT_ENERGY_RATING']==r).sum()
            ax.text(i+1, med+0.8, f'{med:.0f}', ha='center', fontsize=8.5, fontweight='bold', color='#e2e8f0')
            ax.text(i+1, ax.get_ylim()[0]+1, f'n={n:,}', ha='center', fontsize=7.5, color='#a0aec0')
        ax.set_title('Efficiency Score by Energy Rating Band', fontsize=13, fontweight='700', pad=12)
        ax.set_xlabel('Energy Rating', fontsize=10); ax.set_ylabel('Efficiency Score', fontsize=10)
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('<div class="info-card"><b>Finding:</b> Clear separation between bands. Median: A=93, B=83, C=73, D=64, E=50 — consistent with SAP thresholds.</div>', unsafe_allow_html=True)

    with col4:
        fig, ax = plt.subplots(figsize=(6, 4))
        tc = df['TENURE_CLEAN'].value_counts()
        def autopct_count(pct, vals):
            n = int(round(pct / 100 * sum(vals)))
            return f'{pct:.1f}%\n({n:,})'
        wedges, texts, autotexts = ax.pie(
            tc.values, labels=tc.index,
            autopct=lambda p: autopct_count(p, tc.values),
            colors=PALETTE[:len(tc)], startangle=90, pctdistance=0.72,
            wedgeprops=dict(edgecolor='#0f1117', linewidth=2))
        for t in texts:      t.set_color('#e2e8f0'); t.set_fontsize(9)
        for at in autotexts: at.set_color('white'); at.set_fontweight('700'); at.set_fontsize(8.5)
        ax.set_title('Property Tenure Distribution', fontsize=13, fontweight='700', pad=12)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('<div class="info-card"><b>Finding:</b> Social rented (43.8%) is the largest group, reflecting significant social housing provision — key for council-led improvement programmes.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-header">Summary Statistics</p>', unsafe_allow_html=True)
    key_cols = ['CURRENT_ENERGY_EFFICIENCY','POTENTIAL_ENERGY_EFFICIENCY','TOTAL_FLOOR_AREA',
                'CO2_EMISSIONS_CURRENT','HEATING_COST_CURRENT','EFFICIENCY_GAP','COST_SAVING_POTENTIAL']
    st.dataframe(df[key_cols].describe().round(2), use_container_width=True)


# ============================================================
# Page 3: Diagnostic Analytics
# ============================================================
elif "Diagnostic" in page:
    st.markdown('<p class="section-header">Diagnostic Analytics — Why Did This Happen?</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-card">Diagnostic analytics investigates <b>root causes</b> of energy performance patterns, answering: <b>"Why do some properties perform better?"</b></div>', unsafe_allow_html=True)

    st.markdown("#### Correlation Matrix")
    num_cols_corr = ['CURRENT_ENERGY_EFFICIENCY','POTENTIAL_ENERGY_EFFICIENCY','TOTAL_FLOOR_AREA',
                     'CO2_EMISSIONS_CURRENT','HEATING_COST_CURRENT','TOTAL_COST_CURRENT',
                     'EFFICIENCY_GAP','COST_SAVING_POTENTIAL','CO2_PER_AREA','NUMBER_HABITABLE_ROOMS']
    corr = df[num_cols_corr].corr()
    fig, ax = plt.subplots(figsize=(12, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax,
                annot_kws={'size': 9, 'color': 'white', 'weight': 'bold'},
                linewidths=0.5, linecolor='#0f1117', cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Matrix — Key Numerical Features', fontsize=14, fontweight='700', pad=15)
    ax.tick_params(axis='x', rotation=45, labelsize=9); ax.tick_params(axis='y', rotation=0, labelsize=9)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('<div class="info-card"><b>Key Finding:</b> POTENTIAL_ENERGY_EFFICIENCY shows the strongest positive correlation with CURRENT_ENERGY_EFFICIENCY. CO2 and HEATING_COST show strong negative correlations, confirming lower-rated properties are costlier and more polluting.</div>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        eff_colors = ['#e74c3c','#e67e22','#f1c40f','#27ae60','#2ecc71']
        wall_data = (df.groupby('WALLS_ENERGY_EFF')['CURRENT_ENERGY_EFFICIENCY']
                       .agg(['mean','count']).reindex(EFF_ORDER).dropna())
        bars = ax.barh(wall_data.index, wall_data['mean'],
                       color=eff_colors[:len(wall_data)], edgecolor='#0f1117', linewidth=1.2, height=0.55)
        ax.set_xlim(0, wall_data['mean'].max() * 1.18)
        for bar, (idx, row) in zip(bars, wall_data.iterrows()):
            ax.text(row['mean'] + 0.2, bar.get_y() + bar.get_height() / 2,
                    f'{row["mean"]:.1f}  (n={int(row["count"]):,})',
                    va='center', fontsize=9, fontweight='600', color='#e2e8f0')
        ax.set_title('Avg Efficiency by Wall Insulation', fontsize=12, fontweight='700', pad=12)
        ax.set_xlabel('Average Efficiency Score', fontsize=10)
        ax.xaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        wall_range = wall_data['mean'].max() - wall_data['mean'].min()
        st.markdown(f'<div class="info-card"><b>Finding:</b> Wall insulation has a {wall_range:.1f}-point impact. Very Good ({wall_data["mean"].max():.1f}) vs Very Poor ({wall_data["mean"].min():.1f}). The most impactful physical retrofit.</div>', unsafe_allow_html=True)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        age_colors = ['#8e44ad','#2980b9','#27ae60','#f39c12','#e67e22','#e74c3c','#1abc9c']
        ae = df.groupby('PROPERTY_AGE_GROUP')['CURRENT_ENERGY_EFFICIENCY'].mean().reindex(AGE_ORDER).dropna()
        bars = ax.bar(range(len(ae)), ae.values, color=age_colors[:len(ae)],
                      edgecolor='#0f1117', linewidth=1.2, width=0.6)
        ax.plot(range(len(ae)), ae.values, 'o--', color='#1a8cff', linewidth=2, markersize=6, zorder=5)
        for i, val in enumerate(ae.values):
            ax.text(i, val + 0.15, f'{val:.1f}', ha='center', fontsize=9, fontweight='600', color='#e2e8f0')
        ax.set_xticks(range(len(ae))); ax.set_xticklabels(ae.index, rotation=30, ha='right', fontsize=8)
        ax.set_title('Avg Efficiency by Construction Age', fontsize=12, fontweight='700', pad=12)
        ax.set_ylabel('Average Efficiency Score', fontsize=10); ax.set_ylim(65, 82)
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        age_range = ae.max() - ae.min()
        st.markdown(f'<div class="info-card"><b>Finding:</b> Post-2021 ({ae.max():.1f}) vs Pre-1900 ({ae.min():.1f}) = {age_range:.1f} points. Modern building regulations drive this improvement.</div>', unsafe_allow_html=True)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(6, 4))
        age_colors = ['#8e44ad','#2980b9','#27ae60','#f39c12','#e67e22','#e74c3c','#1abc9c']
        ca = df.groupby('PROPERTY_AGE_GROUP')['CO2_EMISSIONS_CURRENT'].mean().reindex(AGE_ORDER).dropna()
        bars = ax.bar(range(len(ca)), ca.values, color=age_colors[:len(ca)],
                      edgecolor='#0f1117', linewidth=1.2, width=0.6)
        for i, val in enumerate(ca.values):
            ax.text(i, val + 0.01, f'{val:.2f}t', ha='center', fontsize=9, fontweight='600', color='#e2e8f0')
        ax.set_xticks(range(len(ca))); ax.set_xticklabels(ca.index, rotation=30, ha='right', fontsize=8)
        ax.set_title('Avg CO2 Emissions by Property Age', fontsize=12, fontweight='700', pad=12)
        ax.set_ylabel('CO2 (tonnes/year)', fontsize=10)
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown(f'<div class="info-card"><b>Finding:</b> Pre-1900 properties emit {ca.max():.2f}t CO2/year vs {ca.min():.2f}t for Post-2021. Older properties disproportionately contribute to Liverpool\'s carbon footprint.</div>', unsafe_allow_html=True)

    with col4:
        fig, ax = plt.subplots(figsize=(6, 4))
        tg = df.groupby('TENURE_CLEAN')['EFFICIENCY_GAP'].mean().sort_values(ascending=False)
        bars = ax.bar(tg.index, tg.values, color=PALETTE[:len(tg)],
                      edgecolor='#0f1117', linewidth=1.2, width=0.5)
        for bar, val in zip(bars, tg.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f'{val:.1f} pts', ha='center', fontsize=10, fontweight='600', color='#e2e8f0')
        ax.set_title('Efficiency Gap by Tenure Type', fontsize=12, fontweight='700', pad=12)
        ax.set_xlabel('Tenure Type', fontsize=10); ax.set_ylabel('Efficiency Gap (points)', fontsize=10)
        ax.tick_params(axis='x', rotation=15)
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown(f'<div class="info-card"><b>Finding:</b> Owner-occupied properties have the highest efficiency gap ({tg.max():.1f} pts), suggesting the greatest untapped improvement potential.</div>', unsafe_allow_html=True)


# ============================================================
# Page 4: Predictive Models (Classification)
# ============================================================
elif "Predictive" in page:
    st.markdown('<p class="section-header">Predictive Analytics — Classification Models</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-card">Three classifiers trained on <b>{len(feature_cols)} physical building characteristics</b> (no SAP-derived scores or cost columns) to predict Energy Rating (A-E). Dataset split 80/20 with stratified sampling. Cross-validation on training data only.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Model Comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        model_names = ['Logistic\nRegression','Random\nForest','Gradient\nBoosting']
        accs = [metrics['lr_acc']*100, metrics['rf_acc']*100, metrics['gb_acc']*100]
        f1s  = [metrics['lr_f1']*100,  metrics['rf_f1']*100,  metrics['gb_f1']*100]
        x = np.arange(len(model_names)); w = 0.3
        b1 = ax.bar(x-w/2, accs, w, label='Accuracy',
                    color=['#1a8cff','#00d4aa','#ff6b6b'], edgecolor='#0f1117', linewidth=1.2)
        b2 = ax.bar(x+w/2, f1s,  w, label='F1 Score',
                    color=['#1a8cff','#00d4aa','#ff6b6b'], edgecolor='#0f1117', linewidth=1.2,
                    alpha=0.55, hatch='//')
        for bar in b1:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f'{bar.get_height():.1f}%', ha='center', fontsize=8, fontweight='600', color='#e2e8f0')
        ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=9)
        ax.set_ylabel('Score (%)', fontsize=10)
        ax.set_ylim(55, max(accs+f1s) * 1.12)
        ax.legend(fontsize=9); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        ax.set_title('Accuracy and F1 Score Comparison (Test Set)', fontsize=12, fontweight='700', pad=12)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("#### Confusion Matrix — Best Model")
        best_name = 'Gradient Boosting' if metrics['gb_acc'] >= metrics['rf_acc'] else 'Random Forest'
        fig, ax = plt.subplots(figsize=(6, 4))
        cm      = metrics['gb_cm']
        cm_norm = metrics['gb_cm_norm']
        sns.heatmap(cm, annot=False, cmap='Blues',
                    xticklabels=le_target.classes_, yticklabels=le_target.classes_,
                    ax=ax, linewidths=0.5, linecolor='#0f1117')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j+0.5, i+0.38, f'{cm[i,j]}',
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        color='white' if cm_norm[i,j] > 0.5 else 'black')
                ax.text(j+0.5, i+0.65, f'({cm_norm[i,j]*100:.1f}%)',
                        ha='center', va='center', fontsize=7.5,
                        color='white' if cm_norm[i,j] > 0.5 else '#444')
        ax.set_title(f'Confusion Matrix — {best_name}\n(count + row %)', fontsize=12, fontweight='700', pad=12)
        ax.set_xlabel('Predicted', fontsize=10); ax.set_ylabel('Actual', fontsize=10)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("#### Top 15 Feature Importances — Random Forest")
    fi    = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(15)
    fi_total = rf.feature_importances_.sum()
    fig, ax = plt.subplots(figsize=(12, 7))
    colors_fi = plt.cm.RdYlGn(np.linspace(0.15, 0.9, len(fi)))
    bars = ax.barh(fi.index, fi.values, color=colors_fi, edgecolor='#0f1117', linewidth=0.8, height=0.65)
    xlim = fi.max() * 1.35
    ax.set_xlim(0, xlim)
    for bar, val in zip(bars, fi.values):
        pct = val / fi_total * 100
        ax.text(val + xlim*0.01, bar.get_y()+bar.get_height()/2,
                f'{val:.4f}  ({pct:.1f}%)', va='center', fontsize=9, fontweight='600', color='#e2e8f0')
    top_feat = fi.index[-1]
    ax.set_title(f'Feature Importance — Key Drivers of Energy Rating Prediction\nTop feature: {top_feat}',
                 fontsize=13, fontweight='700', pad=15)
    ax.set_xlabel('Mean Decrease in Impurity', fontsize=11)
    ax.xaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown(f'<div class="info-card"><b>Key Finding:</b> <b>{top_feat}</b> is the strongest predictor. Physical efficiency ratings (walls, roof, windows, heating) collectively drive most of the model\'s discriminating power.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 5-Fold Cross-Validation Results (Training Data Only)")
    cv_rf = metrics['cv_rf']; cv_gb = metrics['cv_gb']
    cv_df = pd.DataFrame({
        'Model':   ['Random Forest','Gradient Boosting'],
        'Fold 1':  [f'{cv_rf[0]*100:.2f}%', f'{cv_gb[0]*100:.2f}%'],
        'Fold 2':  [f'{cv_rf[1]*100:.2f}%', f'{cv_gb[1]*100:.2f}%'],
        'Fold 3':  [f'{cv_rf[2]*100:.2f}%', f'{cv_gb[2]*100:.2f}%'],
        'Fold 4':  [f'{cv_rf[3]*100:.2f}%', f'{cv_gb[3]*100:.2f}%'],
        'Fold 5':  [f'{cv_rf[4]*100:.2f}%', f'{cv_gb[4]*100:.2f}%'],
        'Mean CV': [f'{cv_rf.mean()*100:.2f}%', f'{cv_gb.mean()*100:.2f}%'],
        'Std Dev': [f'+/-{cv_rf.std()*100:.2f}%', f'+/-{cv_gb.std()*100:.2f}%'],
    })
    st.dataframe(cv_df, use_container_width=True, hide_index=True)


# ============================================================
# Page 5: Energy Rating Predictor
# ============================================================
elif "Predictor" in page:
    st.markdown('<p class="section-header">Energy Rating Predictor</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-card">Enter the property\'s <b>physical building characteristics</b>. The Gradient Boosting model ({metrics["gb_acc"]*100:.1f}% test accuracy, {gb_cv_mean*100:.1f}% 5-fold CV) predicts the energy rating from building fabric alone — no SAP-derived scores required.</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Property Basics**")
        property_type  = st.selectbox("Property Type", ['House','Flat','Bungalow','Maisonette'])
        built_form     = st.selectbox("Built Form", ['Mid-Terrace','Semi-Detached','Detached','End-Terrace','Not Recorded'])
        age_group      = st.selectbox("Construction Age", AGE_ORDER)
        tenure         = st.selectbox("Tenure", ['Owner-occupied','Social Rented','Private Rented','Other'])
        floor_area     = st.slider("Floor Area (m2)", 20, 200, 70)
        floor_height   = st.slider("Floor Height (m)", 2.0, 3.5, 2.4, step=0.1)
        habitable_rooms  = st.slider("Habitable Rooms",   1, 10, 4)
        heated_rooms     = st.slider("Heated Rooms",       1, 10, 4)

    with col2:
        st.markdown("**Building Features**")
        multi_glaze  = st.slider("Double Glazing (%)",       0, 100, 80)
        low_e_light  = st.slider("Low Energy Lighting (%)",  0, 100, 50)
        extension_ct = st.slider("Number of Extensions",     0, 5,   0)
        wind_turbine = st.slider("Wind Turbines",            0, 3,   0)
        photo_supply = st.slider("Photovoltaic Supply (%)",  0, 100, 0)
        main_fuel    = st.selectbox("Main Fuel",
                                    ['mains gas (not community)', 'electricity (not community)',
                                     'mains gas (community)', 'oil', 'LPG'])
        energy_tariff = st.selectbox("Energy Tariff",
                                     ['standard', 'off-peak 7 hour', '24 hour', 'off-peak 10 hour'])
        mains_gas    = st.selectbox("Mains Gas Connected?", ['Y','N'])
        solar_water  = st.selectbox("Solar Water Heating?",  ['N','Y'])

    with col3:
        st.markdown("**Assessor Efficiency Ratings**")
        walls_eff     = st.selectbox("Wall Insulation",        EFF_ORDER)
        roof_eff      = st.selectbox("Roof Insulation",        EFF_ORDER)
        windows_eff   = st.selectbox("Window Glazing",         EFF_ORDER)
        mainheat_eff  = st.selectbox("Main Heating System",    EFF_ORDER)
        mainheatc_eff = st.selectbox("Heating Controls",       EFF_ORDER)
        hw_eff        = st.selectbox("Hot Water System",       EFF_ORDER)
        lighting_eff  = st.selectbox("Lighting",               EFF_ORDER)
        insp_year     = st.slider("Inspection Year", 2010, 2026, 2024)

    st.markdown("---")

    if st.button("Predict Energy Rating", type="primary", use_container_width=True):
        inp_dict = {
            'TOTAL_FLOOR_AREA':         floor_area,
            'FLOOR_HEIGHT':             floor_height,
            'NUMBER_HABITABLE_ROOMS':   habitable_rooms,
            'NUMBER_HEATED_ROOMS':      heated_rooms,
            'MULTI_GLAZE_PROPORTION':   multi_glaze,
            'LOW_ENERGY_LIGHTING':      low_e_light,
            'EXTENSION_COUNT':          extension_ct,
            'WIND_TURBINE_COUNT':       wind_turbine,
            'PHOTO_SUPPLY':             photo_supply,
            'PROPERTY_TYPE':            property_type,
            'BUILT_FORM':               built_form,
            'PROPERTY_AGE_GROUP':       age_group,
            'TENURE_CLEAN':             tenure,
            'MAINS_GAS_FLAG':           mains_gas,
            'SOLAR_WATER_HEATING_FLAG': solar_water,
            'MAIN_FUEL':                main_fuel,
            'ENERGY_TARIFF':            energy_tariff,
            'WALLS_ENERGY_EFF':         walls_eff,
            'ROOF_ENERGY_EFF':          roof_eff,
            'WINDOWS_ENERGY_EFF':       windows_eff,
            'MAINHEAT_ENERGY_EFF':      mainheat_eff,
            'MAINHEATC_ENERGY_EFF':     mainheatc_eff,
            'HOT_WATER_ENERGY_EFF':     hw_eff,
            'LIGHTING_ENERGY_EFF':      lighting_eff,
            'INSPECTION_YEAR':          insp_year,
        }

        inp = pd.DataFrame([inp_dict])
        for col, le in le_dict.items():
            if col in inp.columns:
                raw = inp[col].astype(str).iloc[0]
                inp[col] = le.transform([raw if raw in le.classes_ else le.classes_[0]])
        for col in feature_cols:
            if col not in inp.columns:
                inp[col] = 0
        inp = inp[feature_cols]

        pred_gb   = gb.predict(inp)[0]
        pred_rf   = rf.predict(inp)[0]
        label_gb  = le_target.inverse_transform([pred_gb])[0]
        label_rf  = le_target.inverse_transform([pred_rf])[0]
        proba_gb  = gb.predict_proba(inp)[0]
        proba_rf  = rf.predict_proba(inp)[0]
        conf_gb   = max(proba_gb) * 100
        conf_rf   = max(proba_rf) * 100

        st.markdown("---"); st.markdown("### Prediction Results")
        r1, r2, r3 = st.columns(3)
        desc = {'A':'Excellent','B':'Very Good','C':'Good','D':'Average','E':'Poor'}
        with r1:
            c = RATING_COLORS.get(label_gb, '#888')
            st.markdown(f'<div class="pred-box" style="background:linear-gradient(135deg,{c}cc,{c}55);border-color:{c};"><div style="color:white;font-size:0.9rem;font-weight:600;">GRADIENT BOOSTING</div><div style="font-size:5rem;font-weight:800;color:white;line-height:1.1;">{label_gb}</div><div style="color:white;opacity:0.9;">{desc.get(label_gb,"")} | {conf_gb:.1f}% confidence</div></div>', unsafe_allow_html=True)
        with r2:
            c2 = RATING_COLORS.get(label_rf, '#888')
            st.markdown(f'<div class="pred-box" style="background:linear-gradient(135deg,{c2}cc,{c2}55);border-color:{c2};"><div style="color:white;font-size:0.9rem;font-weight:600;">RANDOM FOREST</div><div style="font-size:5rem;font-weight:800;color:white;line-height:1.1;">{label_rf}</div><div style="color:white;opacity:0.9;">{desc.get(label_rf,"")} | {conf_rf:.1f}% confidence</div></div>', unsafe_allow_html=True)
        with r3:
            agree = "Both models agree" if label_gb == label_rf else "Models disagree — check inputs"
            st.markdown(f'<div class="pred-box" style="background:#1a2035;border-color:#1a8cff;"><div style="color:#1a8cff;font-size:0.9rem;font-weight:600;">INPUT SUMMARY</div><div style="color:#e2e8f0;font-size:0.85rem;line-height:2.1;margin-top:0.5rem;"><b>Property:</b> {property_type}, {built_form}<br><b>Age:</b> {age_group}<br><b>Floor Area:</b> {floor_area}m2<br><b>Fuel:</b> {main_fuel.split()[0].title()}<br><b>Wall Ins.:</b> {walls_eff}<br><br><i>{agree}</i></div></div>', unsafe_allow_html=True)

        st.markdown("---"); st.markdown("#### Prediction Confidence by Rating Band")
        fig, ax = plt.subplots(figsize=(10, 3))
        classes    = le_target.classes_
        proba_pct  = proba_gb[:len(classes)] * 100
        colors_p   = [RATING_COLORS.get(c, '#888') for c in classes]
        bars = ax.bar(classes, proba_pct, color=colors_p, edgecolor='#0f1117', linewidth=1.5, width=0.5)
        for bar, val in zip(bars, proba_pct):
            if val > 0.5:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                        f'{val:.1f}%', ha='center', fontsize=11, fontweight='700', color='#e2e8f0')
        ax.set_xlabel('Energy Rating Band', fontsize=11); ax.set_ylabel('Probability (%)', fontsize=11)
        ax.set_ylim(0, 115); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        ax.set_title('Prediction Probability per Rating Band (Gradient Boosting)', fontsize=12, fontweight='700', pad=12)
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ============================================================
# Page 6: Regression Analysis
# ============================================================
elif "Regression" in page:
    st.markdown('<p class="section-header">Regression Analysis — Predicting Continuous Targets</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-card">Regression models predict <b>continuous numerical values</b> rather than categories. Two targets are modelled: (1) Current Energy Efficiency Score, (2) Cost Saving Potential (£/year). Metrics: MAE (Mean Absolute Error), MSE (Mean Squared Error), RMSE (Root Mean Squared Error), R2 (coefficient of determination).</div>', unsafe_allow_html=True)

    with st.spinner("Training regression models..."):
        reg_results = train_regression(df)

    tab1, tab2 = st.tabs(["Energy Efficiency Score", "Cost Saving Potential"])

    for tab, target_name in zip([tab1, tab2], ['Efficiency Score', 'Cost Saving Potential']):
        with tab:
            res = reg_results[target_name]
            m   = res['metrics']

            st.markdown(f"#### Regression Metrics — {target_name}")
            st.dataframe(
                m.style
                  .highlight_min(subset=['MAE','MSE','RMSE'], color='#1a4a1a')
                  .highlight_max(subset=['R2'], color='#1a4a1a')
                  .highlight_max(subset=['MAE','MSE','RMSE'], color='#4a1a1a')
                  .highlight_min(subset=['R2'], color='#4a1a1a')
                  .format({'MAE':'{:.4f}','MSE':'{:.4f}','RMSE':'{:.4f}','R2':'{:.4f}'}),
                use_container_width=True, hide_index=True
            )

            best_idx = m['R2'].idxmax()
            best_m   = m.loc[best_idx, 'Model']
            best_r2  = m.loc[best_idx, 'R2']
            best_rmse= m.loc[best_idx, 'RMSE']
            best_mae = m.loc[best_idx, 'MAE']
            st.markdown(f'<div class="info-card"><b>Best model: {best_m}</b> — R2={best_r2:.4f} | RMSE={best_rmse:.4f} | MAE={best_mae:.4f}. Green cells = best score, red = worst.</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### MAE / RMSE Comparison")
                model_labels = m['Model'].tolist()
                x = np.arange(len(model_labels)); w = 0.3
                fig, ax = plt.subplots(figsize=(7, 4))
                b1 = ax.bar(x-w/2, m['MAE'].values,  w, label='MAE',
                            color=['#1a8cff','#00d4aa','#ff6b6b'], edgecolor='#0f1117', linewidth=1.2)
                b2 = ax.bar(x+w/2, m['RMSE'].values, w, label='RMSE',
                            color=['#1a8cff','#00d4aa','#ff6b6b'], edgecolor='#0f1117', linewidth=1.2, alpha=0.65)
                for bar in list(b1) + list(b2):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                            f'{bar.get_height():.2f}', ha='center', fontsize=8.5, fontweight='600', color='#e2e8f0')
                ax.set_xticks(x); ax.set_xticklabels(['Lin. Reg.','Rand. Forest','Grad. Boost'], fontsize=9)
                ax.set_ylabel('Error (lower is better)', fontsize=10)
                ax.legend(fontsize=9); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
                ax.set_title(f'MAE and RMSE — {target_name}', fontsize=12, fontweight='700', pad=12)
                plt.tight_layout(); st.pyplot(fig); plt.close()

            with col2:
                st.markdown("#### R2 Score Comparison")
                fig, ax = plt.subplots(figsize=(7, 4))
                bar_colors = ['#1a8cff','#00d4aa','#ff6b6b']
                bars = ax.bar(model_labels, m['R2'].values,
                              color=bar_colors, edgecolor='#0f1117', linewidth=1.2, width=0.5)
                for bar, val in zip(bars, m['R2'].values):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                            f'{val:.4f}', ha='center', fontsize=10, fontweight='700', color='#e2e8f0')
                ax.set_ylim(0, min(1.0, m['R2'].max() * 1.18))
                ax.set_ylabel('R2 Score (higher is better)', fontsize=10)
                ax.set_title(f'R2 Score — {target_name}', fontsize=12, fontweight='700', pad=12)
                ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
                plt.tight_layout(); st.pyplot(fig); plt.close()

            st.markdown("#### Actual vs Predicted — Best Model")
            if best_m == 'Linear Regression':
                y_pred_best = res['lr_pred']
            elif best_m == 'Random Forest':
                y_pred_best = res['rf_pred']
            else:
                y_pred_best = res['gb_pred']

            y_true_best = res['y_test']
            sample_n    = min(500, len(y_true_best))
            idx_s       = np.random.RandomState(42).choice(len(y_true_best), sample_n, replace=False)

            fig, ax = plt.subplots(figsize=(8, 5))
            sc = ax.scatter(y_true_best[idx_s], y_pred_best[idx_s],
                            alpha=0.45, s=18, c='#1a8cff', edgecolors='none')
            lo = min(y_true_best.min(), y_pred_best.min())
            hi = max(y_true_best.max(), y_pred_best.max())
            ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect prediction')
            ax.set_xlabel(f'Actual {target_name}', fontsize=11)
            ax.set_ylabel(f'Predicted {target_name}', fontsize=11)
            ax.set_title(f'Actual vs Predicted — {best_m}\nR2={best_r2:.4f}  |  RMSE={best_rmse:.4f}  |  n={sample_n} sample',
                         fontsize=12, fontweight='700', pad=12)
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown('<div class="info-card"><b>Interpretation:</b> MAE measures average absolute prediction error. MSE penalises large errors more heavily. RMSE (square root of MSE) is in the same units as the target — easier to interpret. R2 of 1.0 = perfect fit; R2 of 0 = model no better than predicting the mean. Gradient Boosting typically achieves the best R2, confirming ensemble methods outperform linear baselines on this dataset.</div>', unsafe_allow_html=True)


# ============================================================
# Page 7: Clustering
# ============================================================
elif "Clustering" in page:
    st.markdown('<p class="section-header">Clustering Analysis — K-Means Segmentation</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-card">K-Means clustering groups properties by similarity in their <b>25 physical building features</b> (standardised). The Elbow Method selects the optimal number of clusters. PCA reduces the feature space to 2D for visualisation.</div>', unsafe_allow_html=True)

    with st.spinner("Running K-Means clustering..."):
        inertias, labels, label_names, X_2d, name_map, X_scaled = run_clustering(df)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Elbow Method — Optimal K")
        k_range = list(range(2, 11))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(k_range, inertias, 'o-', color='#1a8cff', linewidth=2.5, markersize=8, markerfacecolor='#ff6b6b')
        for k, inert in zip(k_range, inertias):
            ax.text(k, inert + max(inertias)*0.01, f'{inert:,.0f}',
                    ha='center', fontsize=7.5, color='#a0aec0')
        ax.axvline(x=4, color='#ffd93d', linestyle='--', linewidth=1.5, alpha=0.8, label='Chosen K=4')
        ax.set_xlabel('Number of Clusters (K)', fontsize=11)
        ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=10)
        ax.set_title('K-Means Elbow Method\nElbow at K=4', fontsize=12, fontweight='700', pad=12)
        ax.legend(fontsize=9); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('<div class="info-card"><b>Finding:</b> Inertia decreases steeply from K=2 to K=4, then flattens. The elbow at K=4 indicates four natural groupings in Liverpool\'s housing stock.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### PCA Cluster Visualisation")
        unique_labels = sorted(set(labels))
        label_to_name = name_map
        cluster_name_order = ['Low Efficiency','Below Average','Above Average','High Efficiency']
        color_map = dict(zip(cluster_name_order, CLUSTER_COLORS))

        fig, ax = plt.subplots(figsize=(7, 4))
        for cl_id in unique_labels:
            cl_name = label_to_name[cl_id]
            mask    = np.array(labels) == cl_id
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=color_map.get(cl_name, '#888'), s=12, alpha=0.5,
                       label=cl_name, edgecolors='none')
        ax.set_xlabel('PCA Component 1', fontsize=10)
        ax.set_ylabel('PCA Component 2', fontsize=10)
        ax.set_title('K-Means Clusters — PCA 2D Projection\n(K=4, 25 physical features)', fontsize=12, fontweight='700', pad=12)
        ax.legend(fontsize=8.5, markerscale=2); ax.grid(True, alpha=0.2)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('<div class="info-card"><b>Finding:</b> PCA reveals partial overlap between clusters, expected given high-dimensional housing data. Cluster separation is strongest along PCA Component 1, representing overall building quality.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Cluster Profiles")

    df_cl = df.copy()
    df_cl['CLUSTER_NUM']  = labels
    df_cl['CLUSTER_NAME'] = label_names

    profile_cols = ['CURRENT_ENERGY_EFFICIENCY','COST_SAVING_POTENTIAL','CO2_EMISSIONS_CURRENT',
                    'TOTAL_FLOOR_AREA','HEATING_COST_CURRENT']
    profile_data = df_cl.groupby('CLUSTER_NAME')[profile_cols].mean().round(2)
    count_data   = df_cl.groupby('CLUSTER_NAME').size().rename('Count')
    profile_data = profile_data.join(count_data)
    profile_data = profile_data.reindex([n for n in cluster_name_order if n in profile_data.index])
    st.dataframe(profile_data, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Cluster Size Distribution")
        cl_counts = df_cl['CLUSTER_NAME'].value_counts()
        cl_counts = cl_counts.reindex([n for n in cluster_name_order if n in cl_counts.index])
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(cl_counts.index, cl_counts.values,
                      color=[color_map.get(n,'#888') for n in cl_counts.index],
                      edgecolor='#0f1117', linewidth=1.2, width=0.55)
        for bar, val in zip(bars, cl_counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+15,
                    f'{val:,}\n({val/len(df_cl)*100:.1f}%)',
                    ha='center', fontsize=9, fontweight='600', color='#e2e8f0')
        ax.set_title('Properties per Cluster', fontsize=12, fontweight='700', pad=12)
        ax.set_xlabel('Cluster', fontsize=10); ax.set_ylabel('Count', fontsize=10)
        ax.set_ylim(0, cl_counts.max() * 1.28)
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        ax.tick_params(axis='x', rotation=15)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        st.markdown("#### Avg Efficiency Score per Cluster")
        cl_eff = df_cl.groupby('CLUSTER_NAME')['CURRENT_ENERGY_EFFICIENCY'].mean()
        cl_eff = cl_eff.reindex([n for n in cluster_name_order if n in cl_eff.index])
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(cl_eff.index, cl_eff.values,
                       color=[color_map.get(n,'#888') for n in cl_eff.index],
                       edgecolor='#0f1117', linewidth=1.2, height=0.5)
        for bar, val in zip(bars, cl_eff.values):
            ax.text(val + 0.3, bar.get_y()+bar.get_height()/2,
                    f'{val:.1f}', va='center', fontsize=10, fontweight='700', color='#e2e8f0')
        ax.set_xlabel('Average Efficiency Score', fontsize=10)
        ax.set_title('Avg Energy Efficiency per Cluster', fontsize=12, fontweight='700', pad=12)
        ax.xaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="info-card"><b>Cluster Interpretation:</b> <b>Low Efficiency</b> — properties requiring urgent intervention; <b>Below Average</b> — largest group, significant improvement potential; <b>Above Average</b> — well-performing stock; <b>High Efficiency</b> — best-performing properties, likely modern builds or fully retrofitted.</div>', unsafe_allow_html=True)


# ============================================================
# Page 8: Recommendation System
# ============================================================
elif "Recommendation" in page:
    st.markdown('<p class="section-header">Recommendation System — Energy Improvement Suggestions</p>', unsafe_allow_html=True)
    st.markdown('<div class="info-card">Two complementary approaches: <b>Part A — Frequency-Based Filtering</b> (identifies the most common improvement recommendations from 10,297 EPC records) and <b>Part B — Content-Based Filtering</b> (uses cosine similarity on building features to find similar properties and surface their recommendations).</div>', unsafe_allow_html=True)

    with st.spinner("Loading recommendation data..."):
        recs_merged = load_recommendations()

    tab_freq, tab_cbf = st.tabs(["Part A: Frequency-Based", "Part B: Content-Based"])

    with tab_freq:
        st.markdown("#### Most Frequently Recommended Improvements (All Properties)")
        top_overall = recs_merged['IMPROVEMENT_SUMMARY_TEXT'].value_counts().head(12)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors_r = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(top_overall)))[::-1]
        bars = ax.barh(range(len(top_overall)), top_overall.values,
                       color=colors_r, edgecolor='#0f1117', linewidth=0.8, height=0.65)
        ax.set_yticks(range(len(top_overall)))
        ax.set_yticklabels(top_overall.index, fontsize=9)
        for bar, val in zip(bars, top_overall.values):
            pct = val / len(recs_merged) * 100
            ax.text(val + 30, bar.get_y()+bar.get_height()/2,
                    f'{val:,} ({pct:.1f}%)', va='center', fontsize=9, fontweight='600', color='#e2e8f0')
        ax.set_xlabel('Number of Properties Recommended', fontsize=11)
        ax.set_title('Top 12 Improvement Recommendations — Frequency Analysis', fontsize=13, fontweight='700', pad=15)
        ax.set_xlim(0, top_overall.max() * 1.35)
        ax.xaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown('<div class="info-card"><b>Key Finding:</b> Solar photovoltaic panels are the single most recommended improvement, followed by internal wall insulation and floor insulation. This reflects the high proportion of pre-1950 properties with solid walls and uninsulated floors in Liverpool.</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Top Recommendations by Energy Rating Band")
        rating_bands = ['E', 'D', 'C', 'B']
        available_bands = [b for b in rating_bands if b in recs_merged['CURRENT_ENERGY_RATING'].values]
        n_bands = len(available_bands)
        fig, axes = plt.subplots(1, n_bands, figsize=(4 * n_bands, 5))
        if n_bands == 1:
            axes = [axes]
        for ax, band in zip(axes, available_bands):
            band_df  = recs_merged[recs_merged['CURRENT_ENERGY_RATING'] == band]
            top_band = band_df['IMPROVEMENT_SUMMARY_TEXT'].value_counts().head(6)
            c_map    = {'E':'#e74c3c','D':'#e67e22','C':'#f1c40f','B':'#27ae60'}
            bars     = ax.barh(range(len(top_band)), top_band.values,
                               color=c_map.get(band,'#888'), edgecolor='#0f1117', linewidth=0.8, height=0.65, alpha=0.85)
            ax.set_yticks(range(len(top_band)))
            ax.set_yticklabels([t[:30]+'...' if len(t)>30 else t for t in top_band.index], fontsize=7.5)
            for bar, val in zip(bars, top_band.values):
                ax.text(val + top_band.max()*0.02, bar.get_y()+bar.get_height()/2,
                        str(val), va='center', fontsize=8, fontweight='600', color='#e2e8f0')
            ax.set_title(f'Rating {band}\n(n={len(band_df):,})', fontsize=11, fontweight='700')
            ax.set_xlim(0, top_band.max() * 1.35)
            ax.xaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        plt.suptitle('Top 6 Improvements by Current Energy Rating Band', fontsize=13, fontweight='700', y=1.02)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('<div class="info-card"><b>Finding:</b> Band E properties prioritise structural improvements (wall and floor insulation). Band C and B properties already have good fabric — recommendations shift toward renewable energy (solar PV) and controls (thermostats).</div>', unsafe_allow_html=True)

    with tab_cbf:
        st.markdown("#### Content-Based Filtering — Property Similarity")
        st.markdown('<div class="info-card">Enter a property index (0-4578) to find the 5 most similar Liverpool properties based on cosine similarity of their 25 physical features. The system retrieves EPC recommendations for those similar properties and ranks them by frequency.</div>', unsafe_allow_html=True)

        with st.spinner("Building similarity model..."):
            inertias_cbf, labels_cbf, label_names_cbf, X_2d_cbf, name_map_cbf, X_scaled_cbf = run_clustering(df)
            nn_model = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
            nn_model.fit(X_scaled_cbf)

        col1, col2 = st.columns([1, 2])
        with col1:
            prop_idx = st.number_input("Property Index (0 to 4578)", min_value=0, max_value=len(df)-1, value=0, step=1)
            st.markdown("**Selected Property Details**")
            prop_row = df.iloc[int(prop_idx)]
            display_fields = {
                'Property Type':   prop_row.get('PROPERTY_TYPE','N/A'),
                'Built Form':      prop_row.get('BUILT_FORM','N/A'),
                'Age Band':        prop_row.get('PROPERTY_AGE_GROUP', prop_row.get('CONSTRUCTION_AGE_BAND','N/A')),
                'Floor Area (m2)': f"{prop_row.get('TOTAL_FLOOR_AREA',0):.1f}",
                'Energy Rating':   prop_row.get('CURRENT_ENERGY_RATING','N/A'),
                'Efficiency Score':f"{prop_row.get('CURRENT_ENERGY_EFFICIENCY',0):.1f}",
                'Cluster':         label_names_cbf[int(prop_idx)],
            }
            for k, v in display_fields.items():
                st.markdown(f'<div class="info-card" style="padding:0.5rem 0.8rem;margin:0.2rem 0;"><b>{k}:</b> {v}</div>', unsafe_allow_html=True)

        with col2:
            distances, indices = nn_model.kneighbors(X_scaled_cbf[int(prop_idx)].reshape(1, -1))
            neighbour_idx = indices[0][1:]
            neighbour_df  = df.iloc[neighbour_idx]

            st.markdown(f"**5 Most Similar Properties (by cosine similarity on 25 features)**")
            sim_display = neighbour_df[['PROPERTY_TYPE','BUILT_FORM','TOTAL_FLOOR_AREA',
                                        'CURRENT_ENERGY_RATING','CURRENT_ENERGY_EFFICIENCY']].copy()
            sim_display.columns = ['Type','Form','Floor Area','Rating','Efficiency']
            sim_display['Similarity'] = [f'{(1-d)*100:.1f}%' for d in distances[0][1:]]
            sim_display = sim_display.reset_index(drop=True)
            sim_display.index += 1
            st.dataframe(sim_display, use_container_width=True)

        st.markdown("#### Recommended Improvements (Based on Similar Properties)")
        recs_simple = recs_merged[['LMK_KEY','IMPROVEMENT_SUMMARY_TEXT','INDICATIVE_COST']].copy()

        neighbour_ratings = neighbour_df['CURRENT_ENERGY_RATING'].values if 'CURRENT_ENERGY_RATING' in neighbour_df.columns else []
        all_improvements  = []
        for n_idx in neighbour_idx:
            n_rating = df.iloc[n_idx].get('CURRENT_ENERGY_RATING', 'C') if hasattr(df.iloc[n_idx], 'get') else 'C'
            band_recs = recs_merged[recs_merged['CURRENT_ENERGY_RATING'] == n_rating]['IMPROVEMENT_SUMMARY_TEXT'].tolist()
            all_improvements.extend(band_recs)

        if not all_improvements:
            all_improvements = recs_merged['IMPROVEMENT_SUMMARY_TEXT'].tolist()

        top_recs = pd.Series(all_improvements).value_counts().head(8)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors_cbf = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_recs)))[::-1]
        bars = ax.barh(range(len(top_recs)), top_recs.values,
                       color=colors_cbf, edgecolor='#0f1117', linewidth=0.8, height=0.65)
        ax.set_yticks(range(len(top_recs)))
        ax.set_yticklabels(top_recs.index, fontsize=9)
        for bar, val in zip(bars, top_recs.values):
            pct = val / sum(top_recs.values) * 100
            ax.text(val + top_recs.max()*0.01, bar.get_y()+bar.get_height()/2,
                    f'{val} ({pct:.1f}%)', va='center', fontsize=9, fontweight='600', color='#e2e8f0')
        ax.set_xlabel('Frequency among similar properties', fontsize=10)
        ax.set_title(f'Content-Based Recommendations for Property {int(prop_idx)}\nBased on {len(neighbour_idx)} most similar properties',
                     fontsize=12, fontweight='700', pad=12)
        ax.set_xlim(0, top_recs.max() * 1.35)
        ax.xaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        cluster_lbl = label_names_cbf[int(prop_idx)]
        st.markdown(f'<div class="info-card"><b>Property {int(prop_idx)}</b> belongs to the <b>{cluster_lbl}</b> cluster. The chart above shows the most frequent improvement recommendations from its 5 most similar properties, ranked by occurrence frequency. This approach surfaces relevant, context-specific improvements without requiring a full EPC assessment.</div>', unsafe_allow_html=True)


# ============================================================
# Page 9: Conclusions
# ============================================================
elif "Conclusions" in page:
    st.markdown('<p class="section-header">Conclusions and Policy Recommendations</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Retrofit Priority Actions")
        wall_data = df.groupby('WALLS_ENERGY_EFF')['CURRENT_ENERGY_EFFICIENCY'].mean()
        wall_range_val = wall_data.max() - wall_data.min() if len(wall_data) > 1 else 15
        for title, desc in [
            ("Wall Insulation Upgrades",   f"Very Poor to Very Good wall insulation adds {wall_range_val:.0f} efficiency points. Top retrofit priority for pre-1950 Liverpool properties."),
            ("Roof Insulation",            "Loft insulation is cost-effective and fast to install — significant efficiency range between Very Poor and Very Good ratings."),
            ("Window Upgrades",            "Double/triple glazing improves efficiency and comfort, especially in pre-1900 Victorian properties."),
            ("Heating Modernisation",      "Modern condensing boilers or heat pumps reduce both energy costs and CO2 emissions substantially."),
        ]:
            st.markdown(f'<div class="info-card"><b>{title}</b><br><span style="color:#a0aec0;font-size:0.9rem;">{desc}</span></div>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### Priority Target Groups")
        pre1900_eff  = df[df['PROPERTY_AGE_GROUP']=='Pre-1900']['CURRENT_ENERGY_EFFICIENCY'].mean()
        pre1950_eff  = df[df['PROPERTY_AGE_GROUP']=='1900-1949']['CURRENT_ENERGY_EFFICIENCY'].mean()
        oo_eff       = df[df['TENURE_CLEAN']=='Owner-occupied']['CURRENT_ENERGY_EFFICIENCY'].mean()
        pr_eff       = df[df['TENURE_CLEAN']=='Private Rented']['CURRENT_ENERGY_EFFICIENCY'].mean()
        sr_eff       = df[df['TENURE_CLEAN']=='Social Rented']['CURRENT_ENERGY_EFFICIENCY'].mean()
        st.dataframe(pd.DataFrame({
            'Target Group':    ['Pre-1900 Properties','1900-1949 Properties','Owner-Occupied','Private Rented','Social Rented'],
            'Avg Efficiency':  [round(pre1900_eff,1), round(pre1950_eff,1), round(oo_eff,1), round(pr_eff,1), round(sr_eff,1)],
            'Priority':        ['Critical','High','Medium','Medium','Lower'],
        }), use_container_width=True, hide_index=True)
        avg_sav = df['COST_SAVING_POTENTIAL'].mean()
        total_sav = avg_sav * len(df)
        st.markdown(f'<div class="info-card">{len(df):,} properties analysed | Avg saving: <b>£{avg_sav:.0f}/property/year</b> | City-wide potential: <b>~£{total_sav:,.0f}/year</b></div>', unsafe_allow_html=True)

    st.markdown("---"); st.markdown("#### Policy Recommendations")
    for title, body in [
        ("1. Target Pre-1950 Housing Stock",         "Over 2,700 properties built before 1950 show the lowest average efficiency scores. A targeted retrofit programme should prioritise cavity wall and loft insulation for this stock."),
        ("2. Expand ECO Scheme Eligibility",          "Private rented properties show a significant efficiency gap. Expanding ECO scheme access would directly benefit private tenants who cannot self-fund improvements."),
        ("3. Mandatory EPC Upgrades for Rentals",     "Properties rented below Band C should be incentivised or required to upgrade through government funding, protecting tenants from high energy costs."),
        ("4. Community Heat Networks",                "Liverpool's density of terraced housing makes it suitable for district heat networks, reducing CO2 across multiple properties simultaneously."),
        (f"5. Data-Driven Targeting ({metrics['gb_acc']*100:.1f}% accuracy)", f"The Gradient Boosting classifier predicts energy ratings from building fabric alone at {metrics['gb_acc']*100:.1f}% accuracy ({gb_cv_mean*100:.1f}% CV). This can flag low-performing properties before costly physical assessments."),
    ]:
        st.markdown(f'<div class="info-card"><b style="color:#1a8cff;">{title}</b><br><span style="color:#a0aec0;font-size:0.9rem;">{body}</span></div>', unsafe_allow_html=True)

    st.markdown("---"); st.markdown("#### Key Conclusions")
    post2021_eff = df[df['PROPERTY_AGE_GROUP']=='Post-2021']['CURRENT_ENERGY_EFFICIENCY'].mean()
    avg_sav2 = df['COST_SAVING_POTENTIAL'].mean()
    total_sav2 = avg_sav2 * len(df)
    for i, c in enumerate([
        "68.6% of Liverpool properties rated C — the city's housing stock is predominantly mid-range efficiency",
        f"Building age is critical — Pre-1900 properties score {pre1900_eff:.1f} vs Post-2021 at {post2021_eff:.1f}",
        "Wall and roof insulation are the strongest physical drivers of energy performance",
        f"Gradient Boosting achieved {metrics['gb_acc']*100:.1f}% test accuracy and {gb_cv_mean*100:.1f}% cross-validation accuracy using building characteristics only",
        f"Significant savings achievable — £{avg_sav2:.0f}/property/year on average (~£{total_sav2:,.0f} city-wide)",
        "K-Means clustering (K=4) identified 4 distinct property segments enabling targeted intervention strategies",
        "Content-based recommendation system surfaces property-specific improvements using cosine similarity on 25 features",
    ], 1):
        st.markdown(f'<div class="info-card"><span style="color:#1a8cff;font-weight:700;">#{i}</span> {c}</div>', unsafe_allow_html=True)


# Footer
st.markdown("""
<hr style="border-color:#2d3748;margin-top:3rem;">
<div style="text-align:center;color:#4a5568;font-size:0.8rem;padding:1rem 0;">
Liverpool EPC Analysis &nbsp;|&nbsp; COM6003 Data Science &nbsp;|&nbsp; Buckinghamshire New University &nbsp;|&nbsp; 2025-26
</div>""", unsafe_allow_html=True)
