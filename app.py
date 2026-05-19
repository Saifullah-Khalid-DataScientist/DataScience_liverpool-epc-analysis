# COM6003 Data Science — Liverpool EPC Analysis
# Streamlit Dashboard — Buckinghamshire New University 2025-26

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
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


@st.cache_resource
def train_models(_df):
    # 25 physical building characteristics — no SAP-derived scores or cost columns
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
    page = st.selectbox("", ["🏠  Overview","📊  Descriptive Analytics","🔬  Diagnostic Analytics",
                              "🤖  Predictive Models","🔮  Energy Rating Predictor","💡  Recommendations"],
                        label_visibility="collapsed")
    st.markdown(f"""<hr style="border-color:#2d3748;">
    <div style="color:#a0aec0;font-size:0.8rem;line-height:2;">
        <div>📍 <b style="color:#e2e8f0;">Liverpool, England</b></div>
        <div>🏘️ <b style="color:#e2e8f0;">4,579</b> properties</div>
        <div>📌 <b style="color:#e2e8f0;">25</b> model features</div>
        <div>🏆 <b style="color:#e2e8f0;">{gb_cv_mean*100:.1f}%</b> CV accuracy (GB)</div>
    </div><hr style="border-color:#2d3748;margin:1rem 0;">
    <div style="color:#718096;font-size:0.75rem;text-align:center;">
        Buckinghamshire New University<br>Academic Year 2025–26
    </div>""", unsafe_allow_html=True)

# Top banner
st.markdown("""
<div class="top-banner">
    <h1>⚡ Liverpool Energy Performance Certificate Analysis</h1>
    <p>COM6003 Data Science &nbsp;|&nbsp; Buckinghamshire New University &nbsp;|&nbsp; Academic Year 2025–26</p>
</div>""", unsafe_allow_html=True)


# Page 1: Overview
if "Overview" in page:
    avg_eff = df['CURRENT_ENERGY_EFFICIENCY'].mean()
    avg_sav = df['COST_SAVING_POTENTIAL'].mean()
    avg_co2 = df['CO2_EMISSIONS_CURRENT'].mean()

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card"><span class="icon">🏘️</span><span class="value">{len(df):,}</span><span class="label">Total Properties</span></div>
        <div class="metric-card"><span class="icon">⚡</span><span class="value">{avg_eff:.1f}</span><span class="label">Avg Efficiency</span></div>
        <div class="metric-card"><span class="icon">💰</span><span class="value">£{avg_sav:.0f}</span><span class="label">Avg Annual Saving</span></div>
        <div class="metric-card"><span class="icon">🌡️</span><span class="value">{avg_co2:.2f}t</span><span class="label">Avg CO₂/year</span></div>
        <div class="metric-card"><span class="icon">🏆</span><span class="value">{gb_cv_mean*100:.1f}%</span><span class="label">Model CV Accuracy</span></div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.markdown('<p class="section-header">About This Analysis</p>', unsafe_allow_html=True)
        for text in [
            "This application presents a complete end-to-end data science pipeline applied to <b>Energy Performance Certificate (EPC)</b> data for <b>Liverpool</b>, England (Local Authority: E08000012), sourced from the <b>Ministry of Housing, Communities and Local Government (MHCLG)</b>.",
            "EPCs are legal documents required when a property is built, sold or rented, assessed using the <b>Standard Assessment Procedure (SAP)</b>. They rate properties from <b>A (most efficient)</b> to <b>G (least efficient)</b>.",
            "<b>Pipeline:</b> Data Acquisition → Feature Engineering → Data Wrangling → Descriptive Analytics → Diagnostic Analytics → Predictive Modelling → Recommendations",
        ]:
            st.markdown(f'<div class="info-card">{text}</div>', unsafe_allow_html=True)

        st.markdown('<p class="section-header">Model Performance Summary</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            'Model':   ['Logistic Regression','Random Forest','Gradient Boosting'],
            'Accuracy':[f"{metrics['lr_acc']*100:.2f}%", f"{metrics['rf_acc']*100:.2f}%", f"{metrics['gb_acc']*100:.2f}%"],
            'F1 Score':[f"{metrics['lr_f1']*100:.2f}%", f"{metrics['rf_f1']*100:.2f}%", f"{metrics['gb_f1']*100:.2f}%"],
            'CV Score':['—', f"{rf_cv_mean*100:.2f}%", f"{gb_cv_mean*100:.2f}% (best)"],
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


# Page 2: Descriptive Analytics
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
        st.markdown('<div class="info-card"><b>Finding:</b> 44% of properties built 1900–1949. This aging stock (75+ years old) presents the greatest energy improvement opportunity.</div>', unsafe_allow_html=True)

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


# Page 3: Diagnostic Analytics
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
        ax.set_title('Avg CO₂ Emissions by Property Age', fontsize=12, fontweight='700', pad=12)
        ax.set_ylabel('CO₂ (tonnes/year)', fontsize=10)
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown(f'<div class="info-card"><b>Finding:</b> Pre-1900 properties emit {ca.max():.2f}t CO₂/year vs {ca.min():.2f}t for Post-2021. Older properties disproportionately contribute to Liverpool\'s carbon footprint.</div>', unsafe_allow_html=True)

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


# Page 4: Predictive Models
elif "Predictive" in page:
    st.markdown('<p class="section-header">Predictive Analytics — Machine Learning Models</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-card">Three classifiers trained on <b>{len(feature_cols)} physical building characteristics</b> (no SAP-derived scores or cost columns) to predict Energy Rating (A–E). Dataset split 80/20 with stratified sampling. Cross-validation on training data only.</div>', unsafe_allow_html=True)

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
    st.markdown(f'<div class="info-card"><b>Key Finding:</b> <b>{top_feat}</b> is the strongest predictor. Assessor-observed efficiency ratings (walls, roof, windows, heating) collectively drive most of the model\'s discriminating power.</div>', unsafe_allow_html=True)

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
        'Std Dev': [f'±{cv_rf.std()*100:.2f}%', f'±{cv_gb.std()*100:.2f}%'],
    })
    st.dataframe(cv_df, use_container_width=True, hide_index=True)


# Page 5: Energy Rating Predictor
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
        floor_area     = st.slider("Floor Area (m²)", 20, 200, 70)
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
            st.markdown(f'<div class="pred-box" style="background:#1a2035;border-color:#1a8cff;"><div style="color:#1a8cff;font-size:0.9rem;font-weight:600;">INPUT SUMMARY</div><div style="color:#e2e8f0;font-size:0.85rem;line-height:2.1;margin-top:0.5rem;"><b>Property:</b> {property_type}, {built_form}<br><b>Age:</b> {age_group}<br><b>Floor Area:</b> {floor_area}m²<br><b>Fuel:</b> {main_fuel.split()[0].title()}<br><b>Wall Ins.:</b> {walls_eff}<br><br><i>{agree}</i></div></div>', unsafe_allow_html=True)

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


# Page 6: Recommendations
elif "Recommendations" in page:
    st.markdown('<p class="section-header">Recommendations and Conclusions</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Retrofit Priority Actions")
        for title, desc in [
            ("Wall Insulation Upgrades",   f"Very Poor to Very Good wall insulation adds {df.groupby('WALLS_ENERGY_EFF')['CURRENT_ENERGY_EFFICIENCY'].mean().max() - df.groupby('WALLS_ENERGY_EFF')['CURRENT_ENERGY_EFFICIENCY'].mean().min():.0f} efficiency points. Top retrofit priority for pre-1950 Liverpool properties."),
            ("Roof Insulation",            "Loft insulation is cost-effective and fast to install — significant efficiency range between Very Poor and Very Good ratings."),
            ("Window Upgrades",            "Double/triple glazing improves efficiency and comfort, especially in pre-1900 Victorian properties."),
            ("Heating Modernisation",      "Modern condensing boilers or heat pumps reduce both energy costs and CO₂ emissions substantially."),
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
        ("4. Community Heat Networks",                "Liverpool's density of terraced housing makes it suitable for district heat networks, reducing CO₂ across multiple properties simultaneously."),
        (f"5. Data-Driven Targeting ({metrics['gb_acc']*100:.1f}% accuracy)", f"The Gradient Boosting model predicts energy ratings from building fabric alone at {metrics['gb_acc']*100:.1f}% accuracy ({gb_cv_mean*100:.1f}% CV). This can flag low-performing properties before costly physical assessments."),
    ]:
        st.markdown(f'<div class="info-card"><b style="color:#1a8cff;">{title}</b><br><span style="color:#a0aec0;font-size:0.9rem;">{body}</span></div>', unsafe_allow_html=True)

    st.markdown("---"); st.markdown("#### Key Conclusions")
    for i, c in enumerate([
        "68.6% of Liverpool properties rated C — the city's housing stock is predominantly mid-range efficiency",
        f"Building age is critical — Pre-1900 properties score {pre1900_eff:.1f} vs Post-2021 at {df[df['PROPERTY_AGE_GROUP']=='Post-2021']['CURRENT_ENERGY_EFFICIENCY'].mean():.1f}",
        "Wall and roof insulation are the strongest physical drivers of energy performance",
        f"Gradient Boosting achieved {metrics['gb_acc']*100:.1f}% test accuracy and {gb_cv_mean*100:.1f}% cross-validation accuracy using building characteristics only",
        f"Significant savings achievable — £{avg_sav:.0f}/property/year on average (£{total_sav:,.0f} city-wide)",
        "Social rented properties show a smaller efficiency gap — prior investment schemes have had a positive effect",
    ], 1):
        st.markdown(f'<div class="info-card"><span style="color:#1a8cff;font-weight:700;">#{i}</span> {c}</div>', unsafe_allow_html=True)


# Footer
st.markdown("""
<hr style="border-color:#2d3748;margin-top:3rem;">
<div style="text-align:center;color:#4a5568;font-size:0.8rem;padding:1rem 0;">
Liverpool EPC Analysis &nbsp;|&nbsp; COM6003 Data Science &nbsp;|&nbsp; Buckinghamshire New University &nbsp;|&nbsp; 2025–26
</div>""", unsafe_allow_html=True)
