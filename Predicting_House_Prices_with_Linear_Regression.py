"""
LEVEL 2 · TASK 1 — Predicting House Prices with Linear Regression
Run:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")
T = "plotly_dark"

st.markdown(
    """<style>
.hero{padding:2rem 2.2rem;border-radius:20px;color:#fff;margin-bottom:1.2rem;
background:linear-gradient(135deg,#0ea5e9 0%,#6366f1 55%,#a855f7 100%);
box-shadow:0 10px 30px rgba(99,102,241,.35)}
.hero h1{margin:0;font-size:2.2rem;color:#fff}.hero p{margin:.4rem 0 0;opacity:.92}
div[data-testid="stMetric"]{background:rgba(127,127,127,.08);border:1px solid rgba(127,127,127,.22);
padding:14px 16px;border-radius:14px}
.big{font-size:2.6rem;font-weight:800;background:linear-gradient(90deg,#38bdf8,#a855f7);
-webkit-background-clip:text;-webkit-text-fill-color:transparent}
</style>""",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero"><h1>🏠 House Price Predictor</h1>'
    "<p>End-to-end linear regression: EDA → feature selection → modelling → evaluation → "
    "coefficient analysis → Ridge / Lasso comparison → live predictions</p></div>",
    unsafe_allow_html=True,
)


# ───────────────────────── data ─────────────────────────
@st.cache_data
def demo_data(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    loc = rng.choice(["Downtown", "Suburb", "Riverside", "Countryside"], n, p=[.25, .4, .2, .15])
    area = rng.normal(1800, 550, n).clip(500, 5000)
    bed = np.clip((area / 600 + rng.normal(0, .7, n)).round(), 1, 7)
    bath = np.clip((bed / 1.6 + rng.normal(0, .5, n)).round(), 1, 5)
    age = rng.integers(0, 60, n).astype(float)
    garage = rng.integers(0, 4, n)
    bonus = {"Downtown": 60000, "Suburb": 25000, "Riverside": 45000, "Countryside": -10000}
    price = (40000 + 135 * area + 9000 * bed + 14000 * bath - 1100 * age + 8000 * garage
             + np.array([bonus[l] for l in loc]) + rng.normal(0, 25000, n))
    df = pd.DataFrame(dict(area=area.round(0), bedrooms=bed, bathrooms=bath, age=age,
                           garage=garage, location=loc, price=price.round(0)))
    df.loc[rng.choice(n, 45, replace=False), "age"] = np.nan  # realistic missing values
    return df


with st.sidebar:
    st.header("⚙️ Settings")
    src = st.radio("Dataset", ["Demo housing data", "Upload CSV (Kaggle)"])
    if src.startswith("Upload"):
        f = st.file_uploader("Upload e.g. Ames / House Prices train.csv", type="csv")
        if f is None:
            st.info("Upload a CSV to continue, or switch to the demo data.")
            st.stop()
        df = pd.read_csv(f)
    else:
        df = demo_data()
    nums = df.select_dtypes("number").columns.tolist()
    guess = next((c for c in ["SalePrice", "price", "Price"] if c in df.columns), nums[-1])
    target = st.selectbox("Target (price) column", nums, index=nums.index(guess))
    cand = [c for c in df.columns if c != target and c.lower() != "id"]
    good = [c for c in cand if df[c].isna().mean() <= .4
            and (pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique() <= 25)]
    feats = st.multiselect("Features", good, default=good)
    alpha = st.slider("Ridge / Lasso alpha", 0.1, 500.0, 50.0)
    st.caption("Split is fixed at 80 / 20 as required by the task.")

if not feats:
    st.warning("Select at least one feature.")
    st.stop()

data = df.dropna(subset=[target])
X, y = data[feats], data[target]
num = X.select_dtypes("number").columns.tolist()
cat = [c for c in feats if c not in num]


@st.cache_data(show_spinner="Training models…")
def train_all(X, y, num, cat, alpha):
    pre = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
    models = {"Linear Regression": LinearRegression(), "Ridge": Ridge(alpha=alpha),
              "Lasso": Lasso(alpha=alpha, max_iter=50000)}
    res = {}
    for name, m in models.items():
        pipe = Pipeline([("pre", clone(pre)), ("model", m)]).fit(Xtr, ytr)
        p = pipe.predict(Xte)
        mse = mean_squared_error(yte, p)
        res[name] = dict(pipe=pipe, pred=p, mse=mse, rmse=mse ** .5, mae=mean_absolute_error(yte, p),
                         r2=r2_score(yte, p), train_r2=pipe.score(Xtr, ytr))
    names = [n.split("__", 1)[1] for n in res["Linear Regression"]["pipe"].named_steps["pre"].get_feature_names_out()]
    return res, yte, names, len(Xtr), len(Xte)


res, yte, names, ntr, nte = train_all(X, y, num, cat, alpha)
lr = res["Linear Regression"]

tabs = st.tabs(["📊 EDA", "🧠 Feature Selection", "🤖 Model", "📈 Evaluation", "⚖️ Coefficients & Ridge/Lasso", "🔮 Predict"])

# ───────────────────────── EDA ─────────────────────────
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Rows", f"{len(df):,}")
    c[1].metric("Columns", df.shape[1])
    c[2].metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c[3].metric("Median price", f"{data[target].median():,.0f}")
    st.subheader("Preview & descriptive statistics")
    st.dataframe(df.head(10), use_container_width=True)
    st.dataframe(df.describe().T, use_container_width=True)
    a, b = st.columns(2)
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False).head(25)
    with a:
        st.subheader("Null check")
        if nulls.empty:
            st.success("No missing values 🎉")
        else:
            st.plotly_chart(px.bar(nulls, template=T, labels={"value": "missing", "index": "column"},
                                   color_discrete_sequence=["#f472b6"]).update_layout(showlegend=False),
                            use_container_width=True)
    with b:
        st.subheader(f"Distribution of target: {target}")
        st.plotly_chart(px.histogram(data, x=target, nbins=50, marginal="box", template=T,
                                     color_discrete_sequence=["#818cf8"]), use_container_width=True)
    st.caption(f"Skewness of target: **{data[target].skew():.2f}** "
               "(right-skewed prices are common; a log-transform can help, but we keep raw prices for interpretability).")

# ───────────────────────── Feature selection ─────────────────────────
with tabs[1]:
    ncols = [c for c in num] + [target]
    corr = data[ncols].corr(numeric_only=True)
    ct = corr[target].drop(target).dropna().sort_values()
    a, b = st.columns(2)
    with a:
        st.subheader("Correlation with price")
        st.plotly_chart(px.bar(ct.tail(15), orientation="h", template=T, color=ct.tail(15),
                               color_continuous_scale="RdBu", range_color=[-1, 1],
                               labels={"value": "correlation", "index": ""}).update_layout(coloraxis_showscale=False),
                        use_container_width=True)
    with b:
        st.subheader("Correlation heatmap")
        top = ct.abs().sort_values(ascending=False).head(11).index.tolist() + [target]
        st.plotly_chart(px.imshow(data[top].corr(numeric_only=True), text_auto=".2f", template=T,
                                  color_continuous_scale="RdBu", zmin=-1, zmax=1), use_container_width=True)
    best = ct.abs().sort_values(ascending=False).head(3).index.tolist()
    st.subheader("📝 Feature selection discussion")
    st.markdown(f"""
**Likely predictors and why**
- **Area / size** – larger homes cost more; usually the strongest driver.
- **Rooms (bedrooms, bathrooms)** – proxy for usable space and family capacity.
- **Location** – the classic real-estate factor; encoded with One-Hot Encoding.
- **Age** – older buildings tend to sell for less (negative relationship).
- **Garage / extras** – smaller but positive premium.

**What the data says:** the three most correlated numeric features are **{', '.join(best)}**.
Features with very low correlation add noise, and highly inter-correlated features (e.g. area vs. rooms) cause
*multicollinearity*, which makes individual coefficients unstable – this is where Ridge/Lasso help.
""")

# ───────────────────────── Model ─────────────────────────
with tabs[2]:
    st.subheader("Preprocessing + model pipeline")
    st.markdown(f"""
| Step | What happens |
|---|---|
| Missing values | numeric → **median**, categorical → **most frequent** |
| Scaling | `StandardScaler` on numeric features (makes coefficients comparable) |
| Encoding | **One-Hot Encoding** on {len(cat)} categorical feature(s): `{', '.join(cat) or 'none'}` |
| Split | **80 / 20** → {ntr:,} training rows, {nte:,} test rows |
| Model | `sklearn.linear_model.LinearRegression` |
""")
    st.success(f"Model trained on {ntr:,} rows using {len(names)} encoded features.")
    st.code("pipe = Pipeline([('pre', ColumnTransformer(...)), ('model', LinearRegression())])\npipe.fit(X_train, y_train)", "python")

# ───────────────────────── Evaluation ─────────────────────────
with tabs[3]:
    c = st.columns(5)
    c[0].metric("MSE", f"{lr['mse']:,.0f}")
    c[1].metric("RMSE", f"{lr['rmse']:,.0f}")
    c[2].metric("MAE", f"{lr['mae']:,.0f}")
    c[3].metric("R² (test)", f"{lr['r2']:.3f}")
    c[4].metric("R² (train)", f"{lr['train_r2']:.3f}")
    gap = lr["train_r2"] - lr["r2"]
    st.caption("Train/test R² gap is small → no strong overfitting." if gap < .05 else
               f"Train R² is {gap:.2f} higher than test R² → some overfitting; try Ridge/Lasso.")
    a, b = st.columns(2)
    pred = lr["pred"]
    with a:
        st.subheader("Actual vs predicted")
        fig = px.scatter(x=yte, y=pred, template=T, opacity=.6, labels={"x": "Actual", "y": "Predicted"},
                         color_discrete_sequence=["#38bdf8"])
        lo, hi = float(min(yte.min(), pred.min())), float(max(yte.max(), pred.max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect fit",
                                 line=dict(color="#f472b6", dash="dash")))
        st.plotly_chart(fig, use_container_width=True)
    with b:
        st.subheader("Residual plot")
        resid = yte.values - pred
        fig = px.scatter(x=pred, y=resid, template=T, opacity=.6, labels={"x": "Predicted", "y": "Residual"},
                         color_discrete_sequence=["#a78bfa"])
        fig.add_hline(y=0, line_color="#f472b6", line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(px.histogram(x=resid, nbins=40, template=T, title="Residual distribution",
                                 color_discrete_sequence=["#34d399"], labels={"x": "Residual"}), use_container_width=True)
    st.info("**Reading the plots:** points hugging the diagonal = accurate predictions. Residuals should look like a "
            "random cloud centred on 0. A funnel or curve means non-linearity or heteroscedasticity "
            "(e.g. errors growing with price).")

# ───────────────────────── Coefficients ─────────────────────────
with tabs[4]:
    coef = pd.DataFrame({"feature": names, **{m: res[m]["pipe"].named_steps["model"].coef_ for m in res}})
    st.subheader("Linear Regression coefficients")
    k = st.slider("Show top N features", 5, min(40, len(coef)), min(15, len(coef)))
    top = coef.reindex(coef["Linear Regression"].abs().sort_values(ascending=False).index).head(k).iloc[::-1]
    top["sign"] = np.where(top["Linear Regression"] >= 0, "Increases price", "Decreases price")
    st.plotly_chart(px.bar(top, x="Linear Regression", y="feature", orientation="h", color="sign", template=T,
                           color_discrete_map={"Increases price": "#34d399", "Decreases price": "#f87171"}),
                    use_container_width=True)
    pos = coef.sort_values("Linear Regression").iloc[-1]
    neg = coef.sort_values("Linear Regression").iloc[0]
    st.markdown(f"**Biggest positive impact:** `{pos.feature}` · **Biggest negative impact:** `{neg.feature}`  "
                "(numeric features are standardised, so a coefficient = price change per +1 standard deviation).")
    st.subheader("🎁 Bonus: Linear vs Ridge vs Lasso")
    comp = pd.DataFrame({m: {"R² test": res[m]["r2"], "RMSE": res[m]["rmse"], "MAE": res[m]["mae"],
                             "Non-zero coefs": int((res[m]["pipe"].named_steps["model"].coef_ != 0).sum())}
                         for m in res}).T
    st.dataframe(comp.style.format({"R² test": "{:.4f}", "RMSE": "{:,.0f}", "MAE": "{:,.0f}"}), use_container_width=True)
    st.plotly_chart(px.bar(comp.reset_index(), x="index", y="R² test", color="index", template=T, text_auto=".4f",
                           labels={"index": "model"}).update_layout(showlegend=False), use_container_width=True)
    st.caption("Lasso can shrink weak coefficients exactly to zero (built-in feature selection); "
               "Ridge shrinks all of them smoothly and handles multicollinearity.")

# ───────────────────────── Predict ─────────────────────────
with tabs[5]:
    st.subheader("Try your own house")
    model_name = st.selectbox("Model", list(res))
    row, cols = {}, st.columns(3)
    for i, f in enumerate(feats):
        with cols[i % 3]:
            if f in num:
                s = data[f].dropna()
                row[f] = st.number_input(f, float(s.min()), float(s.max()), float(s.median()))
            else:
                opts = sorted(data[f].dropna().astype(str).unique())
                row[f] = st.selectbox(f, opts)
    p = res[model_name]["pipe"].predict(pd.DataFrame([row]))[0]
    st.markdown(f'<div class="big">≈ {p:,.0f}</div>', unsafe_allow_html=True)
    st.caption(f"Typical error of this model (RMSE): ±{res[model_name]['rmse']:,.0f}")
