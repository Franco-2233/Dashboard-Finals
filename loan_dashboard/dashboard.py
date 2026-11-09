"""
Loan Approval Intelligence Dashboard
Audience: Credit Risk Manager and Head of Lending
Run:  streamlit run dashboard.py   (keep Loan_clean.csv and the model_outputs/ folder next to this file)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Loan Approval Intelligence Dashboard", page_icon="💵",
                   layout="wide", initial_sidebar_state="expanded")

BASE = Path(__file__).parent
BANK_RED, BANK_GREEN, NAVY = "#e61a2d", "#00a44d", "#102033"
SLATE, INK, MUTED = "#25466D", "#17212b", "#10273d"
SURFACE, BG, BORDER = "#F5EBEB", "#BBBBBB", "#e3e7eb"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background: {BG};
        color: {INK};
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    [data-testid="stSidebar"] {{
        background: {NAVY};
        border-right: 0;
    }}

    [data-testid="stSidebar"] * {{
        color: #f4f7fa !important;
    }}

    [data-testid="stSidebar"] .stMarkdown p {{
        color: #b9c5d1 !important;
    }}

    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="multiselect"] > div {{
        background: #1b2c40;
        border-color: #314459;
    }}

    [data-testid="stSidebar"] hr {{
        border-color: #314459;
    }}

    .exec-brand {{
        padding: 8px 0 22px 0;
        border-bottom: 1px solid #314459;
        margin-bottom: 22px;
    }}

    .exec-brand .mark {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: {BANK_RED};
        font-weight: 800;
        font-size: 18px;
        margin-right: 10px;
    }}

    .exec-brand .name {{
        font-size: 16px;
        font-weight: 800;
        letter-spacing: .02em;
    }}

    .exec-brand .sub {{
        color: #9eacba !important;
        font-size: 11px;
        margin-left: 49px;
        margin-top: -8px;
    }}

    h1 {{
        color: {NAVY} !important;
        font-weight: 800 !important;
        letter-spacing: -0.035em;
        margin-bottom: 2px !important;
    }}

    h2, h3 {{
        color: {NAVY} !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }}

    .exec-kicker {{
        color: {BANK_RED};
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin-bottom: 5px;
    }}

    .exec-subtitle {{
        color: {SLATE};
        font-size: 13px;
        margin-bottom: 20px;
    }}

    [data-testid="stMetric"] {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 17px 18px;
        box-shadow: 0 2px 8px rgba(16,32,51,.04);
        position: relative;
        overflow: hidden;
    }}

    [data-testid="stMetric"]:before {{
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: {BANK_RED};
    }}

    [data-testid="stMetric"] label {{
        color: {MUTED} !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: .06em;
    }}

    [data-testid="stMetricValue"] {{
        color: {NAVY} !important;
        font-size: 25px !important;
        font-weight: 800 !important;
    }}

    [data-testid="stMetricDelta"] {{
        font-size: 11px !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: transparent;
        border-bottom: 1px solid {BORDER};
    }}

    .stTabs [data-baseweb="tab"] {{
        color: {SLATE};
        font-weight: 600;
        font-size: 12px;
        padding: 11px 15px;
    }}

    .stTabs [aria-selected="true"] {{
        color: {BANK_RED} !important;
    }}

    .stTabs [data-baseweb="tab-highlight"] {{
        background: {BANK_RED};
    }}

    .stButton > button {{
        border-radius: 8px;
        border: 1px solid {BORDER};
        font-weight: 600;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 10px;
        overflow: hidden;
    }}

    .stAlert {{
        border-radius: 10px;
    }}

    footer {{
        visibility: hidden;
    }}

    /* Plotly containers */
    [data-testid="stPlotlyChart"] {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 6px;
        box-shadow: 0 2px 8px rgba(16,32,51,.03);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""<style>
    .insight {{background:#fff;border-left:4px solid {BANK_GREEN};border-radius:8px;padding:10px 14px;
              margin:-6px 0 18px 0;font-size:13px;color:{INK};box-shadow:0 1px 4px rgba(16,32,51,.06);}}
    .insight b.k {{color:{NAVY};}}
    .audience {{background:{NAVY};color:#f4f7fa;border-radius:10px;padding:12px 16px;font-size:13px;margin-bottom:16px;}}
    </style>""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ data
@st.cache_data
def load_data():
    path = BASE / "Loan_clean.csv"
    df = pd.read_csv(path, parse_dates=["ApplicationDate"])
    df["Status"] = df["LoanApproved"].map({1: "Approved", 0: "Rejected"})
    return df


@st.cache_data
def load_model_files():
    out = BASE / "model_outputs"
    return (json.loads((out / "model_meta.json").read_text()),
            pd.read_csv(out / "feature_importance.csv"),
            pd.read_csv(out / "model_comparison.csv", index_col=0),
            pd.read_csv(out / "test_predictions.csv"))


@st.cache_resource
def train_whatif_model(features, C, seed):
    """Re-fit the notebook's logistic regression on the same 80% training split (no pickle needed)."""
    df = load_data()
    cats = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    nums = [c for c in features if c not in cats]
    pre = ColumnTransformer([("num", StandardScaler(), nums), ("cat", OneHotEncoder(handle_unknown="ignore"), cats)])
    pipe = Pipeline([("pre", pre), ("m", LogisticRegression(C=C, max_iter=5000))])
    X, y = df[list(features)], df["LoanApproved"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    pipe.fit(Xtr, ytr)
    return pipe, roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])


try:
    df = load_data()
except FileNotFoundError:
    st.error("Loan_clean.csv was not found. Place it in the same folder as dashboard.py.")
    st.stop()
MODEL_OK = (BASE / "model_outputs" / "model_meta.json").exists()


def style(fig, height=380):
    fig.update_layout(template="plotly_white", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
                      font=dict(family="Inter, Arial, sans-serif", color=INK, size=11),
                      title=dict(font=dict(size=14, color=NAVY), x=0.02, xanchor="left"),
                      margin=dict(l=45, r=25, t=55, b=45),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
                      hoverlabel=dict(bgcolor=NAVY, font_color="white"))
    fig.update_xaxes(showgrid=False, linecolor=BORDER)
    fig.update_yaxes(gridcolor="#edf0f2", zeroline=False, linecolor=BORDER)
    return fig


def note(insight, implication):
    st.markdown(f'<div class="insight"><b class="k">Insight:</b> {insight}<br>'
                f'<b class="k">So what:</b> {implication}</div>', unsafe_allow_html=True)


def rate_by(frame, col):
    g = frame.groupby(col, observed=True).agg(Applications=("LoanApproved", "size"), Rate=("LoanApproved", "mean"))
    return g.reset_index()


# ------------------------------------------------------------------ header + filters
st.markdown('<div class="exec-kicker">Lending &amp; Credit Risk Intelligence</div>', unsafe_allow_html=True)
st.title("Loan Approval Intelligence Dashboard")
st.markdown(
    '<div class="audience"><b>Built for:</b> the Credit Risk Manager and the Head of Lending. '
    '<b>Decisions supported:</b> (1) where to tighten or relax approval cut-offs, '
    '(2) which applicant segments to review for consistency and fairness, '
    '(3) whether to pilot model-assisted triage of applications.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="exec-brand"><div><span class="mark">B</span><span class="name">LENDING DASHBOARD</span></div>'
                '<div class="sub">LOAN APPLICATIONS</div></div>', unsafe_allow_html=True)
    st.header("Filters")
    years = sorted(df["ApplicationDate"].dt.year.unique().tolist())
    sel_years = st.multiselect("Application year", years, default=years)
    purposes = sorted(df["LoanPurpose"].unique())
    sel_purposes = st.multiselect("Loan purpose", purposes, default=purposes)
    emp = sorted(df["EmploymentStatus"].unique())
    sel_emp = st.multiselect("Employment status", emp, default=emp)
    edu = ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
    sel_edu = st.multiselect("Education level", edu, default=edu)
    st.caption("Filters apply to the Overview and Approval Drivers tabs. The Model tab always uses the held-out test set.")

f = df[df["ApplicationDate"].dt.year.isin(sel_years) & df["LoanPurpose"].isin(sel_purposes)
       & df["EmploymentStatus"].isin(sel_emp) & df["EducationLevel"].isin(sel_edu)].copy()
if f.empty:
    st.warning("No applications match the current filters. Widen the selection in the sidebar.")
    st.stop()

# ------------------------------------------------------------------ KPIs
overall = f["LoanApproved"].mean()
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Applications", f"{len(f):,}")
k2.metric("Approval rate", f"{overall:.1%}")
k3.metric("Approved loan volume", f"${f.loc[f.LoanApproved == 1, 'LoanAmount'].sum() / 1e6:,.1f}M")
k4.metric("Avg total debt-to-income", f"{f['TotalDebtToIncomeRatio'].mean():.2f}")
k5.metric("Avg credit score", f"{f['CreditScore'].mean():.0f}")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Overview", "Approval Drivers", "Model & Triage", "About the Data"])

# ------------------------------------------------------------------ TAB 1
with tab1:
    st.subheader("Is the approval rate stable over time?")
    yr = f.assign(Year=f["ApplicationDate"].dt.year).groupby("Year").agg(
        Applications=("LoanApproved", "size"), Rate=("LoanApproved", "mean")).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=yr["Year"], y=yr["Applications"], name="Applications", marker_color="#c9d1d9", yaxis="y2", opacity=.6))
    fig.add_trace(go.Scatter(x=yr["Year"], y=yr["Rate"], name="Approval rate", mode="lines+markers",
                             line=dict(color=BANK_RED, width=3)))
    fig.add_hline(y=overall, line_dash="dot", line_color=NAVY, annotation_text=f"Overall {overall:.1%}")
    fig.update_layout(title="Approval rate by application year", yaxis=dict(tickformat=".0%", title="Approval rate", rangemode="tozero"),
                      yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Applications"))
    st.plotly_chart(style(fig), width="stretch")
    if len(yr) > 1:
        hi, lo = yr.loc[yr.Rate.idxmax()], yr.loc[yr.Rate.idxmin()]
        spread = (hi.Rate - lo.Rate) * 100
        note(f"Approval ranged from {lo.Rate:.1%} ({int(lo.Year)}) to {hi.Rate:.1%} ({int(hi.Year)}), a {spread:.1f}-point spread around the overall {overall:.1%}.",
             "Year-to-year swings this small suggest a consistent approval policy." if spread <= 8
             else "The swing is large enough to justify checking whether policy or applicant mix changed.")
    else:
        note(f"Only one year is selected; its approval rate is {overall:.1%}.", "Select several years to judge stability.")

    st.subheader("Which segments are approved most and least often?")
    group_labels = {"Education level": "EducationLevel", "Employment status": "EmploymentStatus",
                    "Home ownership": "HomeOwnershipStatus", "Loan purpose": "LoanPurpose", "Marital status": "MaritalStatus"}
    pick = st.selectbox("Compare approval rate by", list(group_labels), index=0)
    col = group_labels[pick]
    g = rate_by(f, col).sort_values("Rate", ascending=False)
    fig = px.bar(g, x=col, y="Rate", text=g["Rate"].map("{:.0%}".format), hover_data={"Applications": True, "Rate": ":.1%"},
                 color="Rate", color_continuous_scale=[[0, BANK_RED], [1, BANK_GREEN]], title=f"Approval rate by {pick.lower()}")
    fig.add_hline(y=overall, line_dash="dot", line_color=NAVY)
    fig.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False, xaxis_title=None, yaxis_title="Approval rate")
    st.plotly_chart(style(fig, 360), width="stretch")
    big = g[g["Applications"] >= 30]
    if len(big) >= 2:
        top, bot = big.iloc[0], big.iloc[-1]
        sens = " Marital status is a sensitive attribute, so large gaps here warrant a fair-lending review." if col == "MaritalStatus" else ""
        note(f"{top[col]} applicants are approved {top.Rate:.1%} of the time versus {bot.Rate:.1%} for {bot[col]} (a {(top.Rate - bot.Rate) * 100:.0f}-point gap; groups under 30 applications ignored).",
             "Check whether the gap is explained by income and debt (Approval Drivers tab) before treating it as a policy difference." + sens)
    else:
        note("Too few applications per group under the current filters to compare.", "Widen the filters.")

# ------------------------------------------------------------------ TAB 2
with tab2:
    st.subheader("How much debt is too much?")
    bands = [0, .2, .3, .4, .5, .75, 10]
    labels = ["Under 0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50", "0.50-0.75", "Over 0.75"]
    f["DTI band"] = pd.cut(f["TotalDebtToIncomeRatio"], bands, labels=labels)
    d = rate_by(f, "DTI band")
    fig = px.bar(d, x="DTI band", y="Rate", text=d["Rate"].map("{:.0%}".format), hover_data={"Applications": True, "Rate": ":.1%"},
                 title="Approval rate by total debt-to-income band", color_discrete_sequence=[NAVY])
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Approval rate", xaxis_title="Total debt-to-income ratio")
    st.plotly_chart(style(fig, 360), width="stretch")
    ok = d[d["Applications"] >= 30]
    if len(ok) >= 2:
        first, last = ok.iloc[0], ok.iloc[-1]
        low_rows = ok[ok["Rate"] < .05]
        cut = f" Every band from {low_rows.iloc[0]['DTI band']} upward is approved under 5% of the time." if len(low_rows) else ""
        note(f"Approval falls from {first.Rate:.0%} in the lowest-debt band ({first['DTI band']}) to {last.Rate:.0%} in the highest ({last['DTI band']}).{cut}",
             "Debt-to-income works as a de facto cut-off: advise high-DTI applicants up front, or pre-screen them, to save underwriting time.")
    else:
        note("Not enough applications per band under the current filters.", "Widen the filters.")

    st.subheader("What separates approved from rejected applicants?")
    sample = f.sample(min(len(f), 1500), random_state=1)
    fig = px.scatter(sample, x="MonthlyIncome", y="TotalDebtToIncomeRatio", color="Status", opacity=.6,
                     color_discrete_map={"Approved": BANK_GREEN, "Rejected": BANK_RED},
                     title="Monthly income vs. total debt-to-income (each dot is an application)",
                     hover_data=["LoanAmount", "CreditScore"])
    fig.update_yaxes(range=[0, 1.5], title="Total debt-to-income ratio")
    fig.update_xaxes(title="Monthly income ($)")
    st.plotly_chart(style(fig, 420), width="stretch")
    med = f.groupby("Status")[["MonthlyIncome", "TotalDebtToIncomeRatio"]].median()
    if {"Approved", "Rejected"} <= set(med.index):
        note(f"Approved applicants have a median monthly income of ${med.loc['Approved', 'MonthlyIncome']:,.0f} and debt-to-income of {med.loc['Approved', 'TotalDebtToIncomeRatio']:.2f}, versus ${med.loc['Rejected', 'MonthlyIncome']:,.0f} and {med.loc['Rejected', 'TotalDebtToIncomeRatio']:.2f} for rejected ones.",
             "Approvals cluster in the high-income, low-debt corner; applicants outside it need an unusual strength elsewhere to be approved.")
    else:
        note("Only one outcome is present under the current filters.", "Widen the filters to compare approved and rejected applicants.")

# ------------------------------------------------------------------ TAB 3
with tab3:
    if not MODEL_OK:
        st.warning("Model files not found. Run the Colab notebook and place the model_outputs/ folder next to dashboard.py.")
    else:
        meta, imp, comp, pred = load_model_files()
        a, b, c, d_ = st.columns(4)
        r = comp.loc[meta["model_name"]]
        a.metric("Best model", meta["model_name"])
        b.metric("Test AUC", f"{r['Test AUC']:.3f}")
        c.metric("Test F1", f"{r['F1']:.3f}")
        d_.metric("Baseline accuracy", f"{comp.loc['Baseline (majority)', 'Accuracy']:.1%}")

        st.subheader("What does the model rely on most?")
        top = imp.head(10).iloc[::-1]
        fig = px.bar(top, x="Importance (AUC drop)", y="Feature", orientation="h", color_discrete_sequence=[NAVY],
                     title="Top 10 features by permutation importance (held-out test set)")
        fig.update_layout(yaxis_title=None, xaxis_title="Drop in AUC when the feature is shuffled")
        st.plotly_chart(style(fig, 400), width="stretch")
        t5 = imp.head(5)["Feature"].tolist()
        share2 = imp.head(2)["Importance (AUC drop)"].sum() / imp["Importance (AUC drop)"].clip(lower=0).sum()
        note(f"{t5[0]} and {t5[1]} alone account for {share2:.0%} of the model's total importance; the top five are {', '.join(t5)}.",
             "Underwriting guidance and applicant advice should centre on these few factors; most other fields barely move the decision.")

        st.subheader("How much of the queue could be fast-tracked?")
        lo_t, hi_t = meta["t_low"], meta["t_high"]
        order = ["Auto-approve", "Manual review", "Auto-reject"]
        tri = pred.groupby("Triage").agg(Applications=("LoanApproved", "size"), TrueApprovalRate=("LoanApproved", "mean")).reindex(order).reset_index()
        fig = px.bar(tri, x="Triage", y="Applications", text="Applications", color="Triage",
                     color_discrete_map={"Auto-approve": BANK_GREEN, "Manual review": "#9aa6b2", "Auto-reject": BANK_RED},
                     hover_data={"TrueApprovalRate": ":.1%"}, title="Held-out test applications by triage route")
        fig.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(style(fig, 340), width="stretch")
        n = len(pred)
        auto = tri.set_index("Triage")
        prec = auto.loc["Auto-approve", "TrueApprovalRate"]
        racc = 1 - auto.loc["Auto-reject", "TrueApprovalRate"]
        share = 1 - auto.loc["Manual review", "Applications"] / n
        note(f"With thresholds of {lo_t:.2f} (reject) and {hi_t:.2f} (approve), {share:.0%} of {n} test applications are auto-decided: {prec:.1%} of auto-approvals were truly approved and {racc:.1%} of auto-rejections were truly rejected.",
             f"A pilot could free underwriters to focus on the other {1 - share:.0%}. The accuracy target is a policy choice: stricter targets send more cases to people.")

        st.subheader("Try an applicant (what-if scorer)")
        feats = tuple(meta["features"])
        model, auc_chk = train_whatif_model(feats, meta["whatif_C"], meta["seed"])
        typical = {c: (df[c].median() if pd.api.types.is_numeric_dtype(df[c]) else df[c].mode()[0]) for c in feats}
        s1, s2, s3 = st.columns(3)
        mi = s1.slider("Monthly income ($)", 1000, 25000, int(typical["MonthlyIncome"]), 250)
        dti = s1.slider("Total debt-to-income", 0.05, 1.5, float(round(typical["TotalDebtToIncomeRatio"], 2)), 0.05)
        la = s2.slider("Loan amount ($)", 1000, 100000, int(typical["LoanAmount"]), 500)
        ld = s2.select_slider("Loan duration (months)", [12, 24, 36, 48, 60, 72, 84, 96, 108, 120], value=int(typical["LoanDuration"]) if int(typical["LoanDuration"]) in range(12, 121, 12) else 48)
        nw = s3.slider("Net worth ($)", 0, 500000, int(min(typical["NetWorth"], 500000)), 5000)
        es = s3.selectbox("Employment status", sorted(df["EmploymentStatus"].unique()), index=sorted(df["EmploymentStatus"].unique()).index(typical["EmploymentStatus"]))
        row = pd.DataFrame([typical])[list(feats)]
        row.loc[0, ["MonthlyIncome", "TotalDebtToIncomeRatio", "LoanAmount", "LoanDuration", "NetWorth", "EmploymentStatus"]] = [mi, dti, la, ld, nw, es]
        for col_ in row.columns:
            if pd.api.types.is_numeric_dtype(df[col_]):
                row[col_] = row[col_].astype(float)
        p = float(model.predict_proba(row)[:, 1][0])
        route = "Auto-approve" if p >= hi_t else ("Auto-reject" if p <= lo_t else "Manual review")
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=p * 100, number={"suffix": "%"},
                                       title={"text": f"Predicted approval probability, route: {route}"},
                                       gauge={"axis": {"range": [0, 100]}, "bar": {"color": NAVY},
                                              "steps": [{"range": [0, lo_t * 100], "color": "#f4b6bc"},
                                                        {"range": [lo_t * 100, hi_t * 100], "color": "#d9dee3"},
                                                        {"range": [hi_t * 100, 100], "color": "#b7e4c7"}]}))
        st.plotly_chart(style(gauge, 280), width="stretch")
        st.caption("All other fields are held at typical (median or most common) values, and debt-to-income is set independently of the loan terms, "
                   "so treat results as illustrative. Uses the logistic regression model; grey band = manual-review zone.")

        with st.expander("Model comparison (held-out test set)"):
            st.dataframe(comp.style.format("{:.3f}"), width="stretch")

# ------------------------------------------------------------------ TAB 4
with tab4:
    st.subheader("About the data and method")
    st.markdown(
        f"- **Data:** {len(df):,} loan applications ({df['ApplicationDate'].min():%b %Y} to {df['ApplicationDate'].max():%b %Y}), "
        f"{df.shape[1] - 1} fields, no missing values or duplicate rows. Overall approval rate {df['LoanApproved'].mean():.1%}.\n"
        "- **Model:** classification of the approval decision from application-time information. `RiskScore`, interest rates and monthly "
        "payment are excluded because they are set after the bank assesses the applicant (leakage; see the notebook).\n"
        "- **Limitations:** the model learns the bank's past decisions, not repayment behaviour. Performance is unusually high, which is typical of "
        "rule-based or synthetic data. The triage accuracy target is an illustrative policy assumption.")
    cq = df.isna().sum().sum()
    q1, q2, q3 = st.columns(3)
    q1.metric("Rows", f"{len(df):,}")
    q2.metric("Columns", f"{df.shape[1]}")
    q3.metric("Missing cells", f"{int(cq):,}")

st.markdown('<div style="text-align:right;color:#7a8794;font-size:10px;letter-spacing:.05em;">LENDING &amp; CREDIT RISK INTELLIGENCE</div>',
            unsafe_allow_html=True)
