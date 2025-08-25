\
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import re

st.set_page_config(page_title="Influencer Price Calculator", page_icon="💸", layout="wide")
st.title("💸 Influencer Price Calculator")
st.caption("CSV를 업로드하고, 입력값으로 적정 단가를 예측합니다. (RandomForest 회귀)")

# -------------------- Utilities --------------------
EXPECTED = [
    "Influencer","Followers","Tier","Median of 20-View","Median of 20-Save",
    "US Viewer %","Influencer Quoted Price","Contracted Price"
]

CANON = {
    "influencer":"Influencer",
    "followers":"Followers",
    "tier":"Tier",
    "median of 20-view":"Median of 20-View",
    "median of 20-save":"Median of 20-Save",
    "us viewer %":"US Viewer %",
    "us viewer%":"US Viewer %",
    "us %":"US Viewer %",
    "influencer quoted price":"Influencer Quoted Price",
    "quoted price":"Influencer Quoted Price",
    "contracted price":"Contracted Price",
    "final price":"Contracted Price"
}

# 원하는 Tier 순서 (UI 및 정렬에 사용)
TIER_ORDER = ["Nano", "Micro", "Mid", "Macro", "Mega"]

def norm(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip())
    s = s.replace("‑","-").replace("–","-")  # weird dashes
    return s

def clean_numeric(x):
    if pd.isna(x):
        return None
    x = str(x)
    x = x.replace(",", "").replace("$", "").replace("%", "").strip()
    x = x.replace("USD","").strip()
    try:
        return float(x)
    except ValueError:
        return None

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    # Try common encodings
    for enc in ["utf-8", "cp949", "ISO-8859-1"]:
        try:
            df = pd.read_csv(file, encoding=enc)
            # normalize headers
            new_cols = []
            for c in df.columns:
                c2 = norm(c)
                key = c2.lower()
                new_cols.append(CANON.get(key, c2))
            df.columns = new_cols
            return df
        except Exception:
            file.seek(0)
            continue
    st.error("CSV 인코딩을 읽지 못했습니다. UTF-8/CP949로 저장해서 다시 시도하세요.")
    st.stop()

@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    # ensure essential columns exist
    missing = [c for c in ["Tier","Followers","Median of 20-View","Median of 20-Save","US Viewer %","Contracted Price"] if c not in df.columns]
    if missing:
        st.error(f"필수 컬럼이 없습니다: {missing}. CSV 헤더를 확인하세요. (예상 헤더 예: {EXPECTED})")
        st.stop()

    # numeric cleanup
    for col in ["Followers","Median of 20-View","Median of 20-Save","US Viewer %","Influencer Quoted Price","Contracted Price"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)

    # filter to contracted
    mask = df["Contracted Price"].notna()
    if not mask.any():
        st.error("계약 단가(Contracted Price)가 존재하는 행이 없습니다. CSV를 확인하세요.")
        st.stop()

    df_train = df.loc[mask, ["Followers","Median of 20-View","Median of 20-Save","US Viewer %","Tier","Contracted Price"]].dropna()
    if len(df_train) < 30:
        st.warning("학습 가능한 표본이 30건 미만입니다. 예측 정확도가 낮을 수 있어요.")

    X = df_train[["Followers","Median of 20-View","Median of 20-Save","US Viewer %","Tier"]]
    y = df_train["Contracted Price"]

    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), ["Tier"])],
        remainder="passthrough"
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=400, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    return model, r2, df

def predict_price(model, row: dict):
    input_df = pd.DataFrame([row])
    pred = float(model.predict(input_df)[0])
    return pred

def verdict(pred, quoted, tol=0.2):
    if quoted is None or (isinstance(quoted, float) and np.isnan(quoted)):
        return "No Quote", "제안 단가가 없습니다."
    if quoted > pred * (1 + tol):
        return "Overpriced", f"제안가(${quoted:,.0f})가 예상치(${pred:,.0f})보다 {int(tol*100)}% 이상 높음"
    elif quoted < pred * (1 - tol):
        return "Underpriced", f"제안가(${quoted:,.0f})가 예상치(${pred:,.0f})보다 {int(tol*100)}% 이상 낮음"
    else:
        return "Fair", f"제안가(${quoted:,.0f})가 예상치(${pred:,.0f}) ±{int(tol*100)}% 이내"

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("1) CSV 업로드")
    file = st.file_uploader("influencer data.csv 업로드", type=["csv"])
    tol_pct = st.slider("판정 허용 범위 (±%)", 5, 40, 20, step=5)
    st.caption("예: 20% → 예상치 기준 ±20% 이내면 Fair")

if not file:
    st.info("좌측에서 CSV를 업로드하세요. 권장 컬럼: " + ", ".join(EXPECTED))
    st.stop()

df_raw = load_csv(file)
model, r2, df_clean = train_model(df_raw)
st.success(f"모델 학습 완료 · 테스트 R² = {r2:.2f}")

# -------------------- Tabs --------------------
tab1, tab2 = st.tabs(["🔢 단건 계산기", "📦 일괄 판정"])

with tab1:
    st.subheader("단건 입력으로 예측")
    c1, c2, c3 = st.columns(3)
    with c1:
        # Tier 드롭다운: 원하는 순서 유지
        present_tiers = [t for t in TIER_ORDER if t in df_clean["Tier"].dropna().unique().tolist()]
        if not present_tiers:
            present_tiers = sorted(df_clean["Tier"].dropna().unique().tolist())
        tier = st.selectbox("Tier", options=present_tiers, index=0)

        followers = st.number_input("Followers", min_value=0, value=20000, step=100)
    with c2:
        median_view = st.number_input("Median of 20-View", min_value=0, value=3000, step=10)
        median_save = st.number_input("Median of 20-Save", min_value=0, value=20, step=1)
    with c3:
        us_pct = st.number_input("US Viewer %", min_value=0, max_value=100, value=40, step=1)
        quoted = st.number_input("Influencer Quoted Price ($)", min_value=0, value=500, step=10)

    if st.button("예측 실행", type="primary"):
        row = {
            "Followers": followers,
            "Median of 20-View": median_view,
            "Median of 20-Save": median_save,
            "US Viewer %": us_pct,
            "Tier": tier
        }
        pred = predict_price(model, row)
        v, reason = verdict(pred, quoted, tol=tol_pct/100.0)

        st.metric(label="예상 단가 (USD)", value=f"${pred:,.0f}")
        st.metric(label="제안 단가 (USD)", value=f"${quoted:,.0f}")
        st.markdown(f"**판정:** {v}")
        st.caption(reason)

with tab2:
    st.subheader("CSV 일괄 판정")
    df = df_clean.copy()

    # compute predictions for rows with enough data
    needed = ["Followers","Median of 20-View","Median of 20-Save","US Viewer %","Tier"]
    missing_mask = df[needed].isna().any(axis=1)
    usable = df[~missing_mask].copy()

    # clean quoted
    if "Influencer Quoted Price" in usable.columns:
        usable["Influencer Quoted Price"] = usable["Influencer Quoted Price"].apply(clean_numeric)
    else:
        usable["Influencer Quoted Price"] = np.nan

    if usable.empty:
        st.warning("예측에 필요한 컬럼에 결측치가 많아 일괄 판정을 수행할 수 없습니다.")
    else:
        preds, verdicts, reasons = [], [], []
        tol = tol_pct/100.0
        for _, r in usable.iterrows():
            row = {
                "Followers": r["Followers"],
                "Median of 20-View": r["Median of 20-View"],
                "Median of 20-Save": r["Median of 20-Save"],
                "US Viewer %": r["US Viewer %"],
                "Tier": r["Tier"]
            }
            pred = predict_price(model, row)
            q = r.get("Influencer Quoted Price", np.nan)
            v, reason = verdict(pred, q, tol=tol)
            preds.append(pred); verdicts.append(v); reasons.append(reason)

        usable["Predicted Price"] = np.round(preds, 2)
        usable["Verdict"] = verdicts
        usable["Reason"] = reasons

        # Tier 순서로 정렬(있을 때만)
        cat_type = pd.api.types.CategoricalDtype(categories=TIER_ORDER, ordered=True)
        if "Tier" in usable.columns:
            try:
                usable["Tier"] = usable["Tier"].astype(cat_type)
                usable = usable.sort_values(["Tier","Followers"])
            except Exception:
                pass

        st.dataframe(
            usable[["Influencer","Tier","Followers","Median of 20-View","Influencer Quoted Price","Predicted Price","Verdict","Reason"]].head(200),
            use_container_width=True
        )
        # Download
        csv_bytes = usable.to_csv(index=False).encode("utf-8")
        st.download_button("결과 CSV 다운로드", data=csv_bytes, file_name="priced_influencers.csv", mime="text/csv")

st.divider()
st.caption("※ 로컬에서 실행되는 앱입니다. 업로드한 파일은 브라우저/PC 내에서만 처리됩니다.")
