import os
import requests
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from datetime import date
from pathlib import Path

# ---------------------------- CONFIG ---------------------------------
OWNER = "BSalvatierra13"
REPO  = "credit-scoring-lc"
TAG   = "v1.0.0"

st.set_page_config(page_title="Credit Scoring — Release", layout="wide")

# Rutas de datos auxiliares (CSV con promedios y bins)
DATA_DIR = Path(__file__).parent  # carpeta donde está tu script
DF_GRADE = pd.read_csv(DATA_DIR / "grade_avg_int_rate.csv")     # cols: grade, avg_int_rate (fracción)
DF_BINS  = pd.read_csv(DATA_DIR / "interest_rate_bins.csv")     # cols: bin, left, right

# ---------------------------- MODEL LOADER ----------------------------
@st.cache_resource(show_spinner=False)
def load_model_from_github_release(owner: str, repo: str, tag: str):
    """
    1) Llama a la API pública de GitHub para obtener la Release por TAG.
    2) Busca el primer asset que termine en .joblib.
    3) Lo descarga por streaming (barra de progreso) y lo guarda en disco.
    4) Lo carga con joblib y devuelve el objeto (tu Pipeline).
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
    r = requests.get(api_url, timeout=60)
    r.raise_for_status()
    rel = r.json()

    assets = rel.get("assets", [])
    joblib_assets = [a for a in assets if a.get("name", "").lower().endswith(".joblib")]
    if not joblib_assets:
        names = [a.get("name", "(sin nombre)") for a in assets]
        raise RuntimeError(
            "No encontré ningún asset .joblib en la Release.\n"
            f"Assets disponibles: {names}"
        )

    asset = joblib_assets[0]
    url = asset.get("browser_download_url")
    fname = asset.get("name", "model.joblib")

    # Descarga con progreso
    tmp_path = fname + ".part"
    with requests.get(url, stream=True, timeout=600) as dl:
        dl.raise_for_status()
        total = int(dl.headers.get("Content-Length", 0))
        done = 0
        chunk = 1024 * 1024

        prog = st.progress(0.0)
        with open(tmp_path, "wb") as f, st.spinner(f"Descargando {fname}…"):
            for data in dl.iter_content(chunk_size=chunk):
                if not data:
                    continue
                f.write(data)
                done += len(data)
                if total:
                    prog.progress(min(done / total, 1.0))

    os.replace(tmp_path, fname)
    st.success(f"Descargado: {fname} ({done/1024/1024:.1f} MB)")
    return joblib.load(fname)

# --- NO cargar en el arranque ---
st.title("Credit Scoring — modelo cargado desde Release")

if "model" not in st.session_state:
    st.info("Click **Download & Load model** to initialize.")
    if st.button("Download & Load model", type="primary"):
        try:
            st.session_state.model = load_model_from_github_release(OWNER, REPO, TAG)
            st.success("Model loaded")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to download/load model: {e}")
            st.stop()
else:
    model = st.session_state.model
    st.success("Model ready")
# ---------------------------- FE HELPERS ------------------------------
def _parse_rate_series(s: pd.Series) -> pd.Series:
    """
    Convierte '13.56%' / '13.56' / 0.1356 -> fracción decimal.
    Si promedio > 1, asumimos que viene en porcentaje (13.56 -> 0.1356).
    """
    s_str = s.astype(str).str.strip().str.replace('%', '', regex=False).str.replace(',', '.', regex=False)
    out = pd.to_numeric(s_str, errors="coerce")
    if pd.notnull(out).any() and out.mean(skipna=True) > 1:
        out = out / 100.0
    return out

def _edges_from_left_right(df_bins: pd.DataFrame, data: pd.Series) -> list:
    left0  = pd.to_numeric(df_bins["left"], errors="coerce").min()
    rights = pd.to_numeric(df_bins["right"], errors="coerce").dropna().tolist()
    edges  = [left0] + rights
    # asegurar cobertura del rango observado
    lo, hi = float(np.nanmin(data.values)), float(np.nanmax(data.values))
    if edges[0] > lo:
        edges = [min(edges[0], lo)] + edges
    if edges[-1] < hi:
        edges = edges + [max(edges[-1], hi)]
    return edges

# ---------------------- FEATURE ENGINEERING ---------------------------
def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica las features derivadas que pediste, sin pedirlas al usuario.
    Requiere: 'grade', 'annual_inc', 'loan_amnt', 'int_rate'
    """
    df = df.copy()

    # 1) high_grade (grade > 'D' -> 1, else 0)
    df["high_grade"] = (df["grade"].astype(str) > "D").astype(int)

    # 2) loan_to_income
    den = pd.to_numeric(df["annual_inc"], errors="coerce").fillna(0.0).replace(0, 1)
    df["loan_to_income"] = pd.to_numeric(df["loan_amnt"], errors="coerce").fillna(0.0) / den

    # 3) critical_int_rate (> 13%) – normalizamos int_rate a fracción
    ir = _parse_rate_series(df["int_rate"]).fillna(0.0)
    df["critical_int_rate"] = (ir > 0.13).astype(int)

    # 4) promedio por grade desde DF_GRADE (ya en fracción)
    grade_map = dict(zip(DF_GRADE["grade"].astype(str), DF_GRADE["avg_int_rate"]))
    df["_avg_rate_grade"] = df["grade"].astype(str).map(grade_map).fillna(ir.mean())

    # 5) diferencia y ratio (ratio robusto contra 0)
    df["int_rate_diff_from_grade"] = ir - df["_avg_rate_grade"]
    safe_avg = pd.to_numeric(df["_avg_rate_grade"], errors="coerce").replace(0, np.nan)
    df["int_rate_ratio_grade"] = ir.divide(safe_avg).fillna(0.0)

    # 6) binning de tasa con left/right reales
    edges = _edges_from_left_right(DF_BINS, ir)
    df["int_rate_bin"] = pd.cut(ir, bins=edges, labels=False, include_lowest=True, right=True)

    # limpiar auxiliares si no te interesa exponerlas
    df.drop(columns=["_avg_rate_grade"], inplace=True, errors="ignore")
    return df

# ------------------------- UI MANUAL ----------------------------------
st.subheader("Carga manual de características")

with st.form("manual_form"):
    c1, c2, c3 = st.columns(3)

    # Columna 1
    loan_amnt   = c1.number_input("loan_amnt", min_value=0.0, step=100.0, format="%.2f")
    term_months = c1.number_input("term_months", min_value=0, step=6)
    int_rate    = c1.text_input("int_rate (ej: 13.5% o 0.135)", value="")

    # Columna 2
    grade                 = c2.selectbox("grade", options=list("ABCDEFG"))
    sub_grade             = c2.text_input("sub_grade (ej: B3)", value="")
    emp_length_years_txt  = c2.number_input("emp_length_years_txt (años)", min_value=0, step=1)
    home_ownership        = c2.selectbox("home_ownership", options=["RENT", "OWN", "MORTGAGE", "OTHER"])

    # Columna 3
    annual_inc        = c3.number_input("annual_inc", min_value=0.0, step=1000.0, format="%.2f")
    dti               = c3.text_input("dti (ej: 18.2% o 0.182)", value="")
    revol_util        = c3.text_input("revol_util (ej: 42% o 0.42)", value="")
    earliest_cr_line  = c3.number_input("earliest_cr_line (meses)", min_value=0, step=1)  # <<< NUMÉRICO
    open_acc          = c3.number_input("open_acc", min_value=0, step=1)

    submitted = st.form_submit_button("Calcular PD")

if submitted:
    # Construimos el DF base con los nombres EXACTOS
    base = {
        "loan_amnt": [loan_amnt],
        "term_months": [term_months],
        "int_rate": [int_rate],  # lo normaliza la FE
        "grade": [grade],
        "sub_grade": [sub_grade],
        "emp_length_years_txt": [emp_length_years_txt],  # numérico (si tu pipeline esperaba texto, cambialo)
        "home_ownership": [home_ownership],
        "annual_inc": [annual_inc],
        "dti": [dti],                 # normalizamos como tasa
        "revol_util": [revol_util],   # normalizamos como tasa
        "earliest_cr_line": [earliest_cr_line],  # NUMÉRICO (meses)
        "open_acc": [open_acc],
    }
    df_in = pd.DataFrame(base)

    # Normalizaciones adicionales de tasas que pueden venir como % o fracción
    df_in["dti"]        = _parse_rate_series(df_in["dti"]).fillna(0.0)
    df_in["revol_util"] = _parse_rate_series(df_in["revol_util"]).fillna(0.0)
    # Garantizar tipo numérico en earliest_cr_line (por si acaso)
    df_in["earliest_cr_line"] = pd.to_numeric(df_in["earliest_cr_line"], errors="coerce").fillna(0).astype(int)

    # Aplicar Feature Engineering
    df_fe = apply_feature_engineering(df_in)

    # ---------- Alinear columnas y tipar correctamente ----------
    # definimos qué columnas son categóricas y cuáles numéricas en TU app
    CATS = {"grade", "sub_grade", "emp_length_years_txt", "home_ownership"}
    # Si tu pipeline tiene otros categóricos, agrégalos a CATS.
    # Todo lo demás lo trataremos como numérico.

    X = df_fe.copy()

    # 1) Si el pipeline expone columnas esperadas, reindexamos
    expected = None
    try:
        prep = model.named_steps.get("prep", None)
        if prep is not None and hasattr(prep, "feature_names_in_"):
            expected = list(prep.feature_names_in_)
    except Exception:
        pass

    if expected is not None:
        # Creamos un df con TODAS las columnas esperadas
        tmp = {}
        for col in expected:
            if col in X.columns:
                tmp[col] = X[col]
            else:
                # columna faltante: rellenamos según tipo esperado (heurística por nombre)
                if col in CATS:
                    tmp[col] = ""  # string vacío para categóricas
                else:
                    tmp[col] = 0.0  # 0.0 para numéricas
        X = pd.DataFrame(tmp, columns=expected)
    else:
        # Sin info de expected: usamos lo que tenemos
        pass

    # 2) Cast explícito de tipos
    for col in X.columns:
        if col in CATS:
            X[col] = X[col].astype(str)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

    # ---------- Predicción ----------
    try:
        proba = float(model.predict_proba(X)[:, 1][0])
        yhat  = int(proba >= 0.5)
        st.success("Predicción realizada")
        m1, m2 = st.columns(2)
        m1.metric("PD estimada", f"{proba:.3f}")
        m2.metric("Clasificación @0.50", "Default" if yhat == 1 else "No Default")
        with st.expander("Ver fila de entrada (con FE)"):
            st.dataframe(X.T, use_container_width=True)  # muestro lo que realmente consume el modelo
    except Exception as e:
        st.error(f"Error al predecir: {e}")
