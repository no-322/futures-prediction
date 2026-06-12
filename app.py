"""Streamlit GUI for the Futures Price Direction Predictor.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALGO_DISPLAY: dict[str, str] = {
    "rf":       "Random Forest",
    "gbm":      "Gradient Boosting",
    "svm":      "SVM",
    "baseline": "Logistic Regression",
    "hmm":      "HMM-regime",
}
DISPLAY_ALGO: dict[str, str] = {v: k for k, v in ALGO_DISPLAY.items()}

# Model dropdown for both tabs (4 base classifiers + HMM-regime).
MODEL_OPTIONS = ["Random Forest", "Gradient Boosting", "SVM",
                 "Logistic Regression", "HMM-regime"]
_FEATURE_LABELS = {"v1": "v1 — 20 features", "v2": "v2 — 49 features"}
PROC_DIR = Path("data/processed")

REQUIRED_COLS = {"Date and Time", "Open", "Close", "High", "Low", "VWAP"}

MODEL_PATHS: dict[str, Path] = {
    "rf":       Path("data/processed/rf_model.joblib"),
    "gbm":      Path("data/processed/gbm_model.joblib"),
    "svm":      Path("data/processed/svm_model.joblib"),
    "baseline": Path("data/processed/baseline_model.joblib"),
}

METADATA_PATH   = Path("data/processed/training_metadata.json")
STATS_PATH      = Path("docs/statistics.md")
UPLOAD_TRAIN    = Path("data/processed/_upload_train.csv")
UPLOAD_PRED     = Path("data/processed/_upload_pred.csv")
CONFIG_PATH     = Path("config.yaml")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_metadata(algo: str | None = None) -> dict | None:
    """Return metadata for one algo, or the full dict if algo is None."""
    if not METADATA_PATH.exists():
        return None
    try:
        all_meta = json.loads(METADATA_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if algo is None:
        return all_meta
    return all_meta.get(algo)


def _render_last_run_info(algo: str) -> None:
    with st.expander("Last Training Run"):
        meta = _load_metadata(algo)
        if meta is None:
            st.info("No training record found for this model.")
            return
        col1, col2 = st.columns(2)
        col1.markdown(f"**Algorithm:** {meta['display_name']}")
        col1.markdown(f"**Data path:** {meta['data_path']}")
        col1.markdown(f"**Train size:** {meta['train_size']*100:.0f}%  |  Test rows: {meta['n_test']:,}")
        col2.markdown(f"**Timestamp:** {meta['timestamp']}")
        col2.markdown(f"**Accuracy:** {meta['metrics']['accuracy']:.4f}")
        col2.markdown(f"**Recall (Up):** {meta['metrics']['recall']:.4f}")
        st.markdown("**Hyperparameters:**")
        st.json(meta["params"])


def _confusion_df(cm: np.ndarray) -> pd.DataFrame:
    tn, fp, fn, tp = cm.ravel()
    return pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual Down", "Actual Up"],
        columns=["Predicted Down", "Predicted Up"],
    )


# --- model-variant resolution (model × feature-set × flat) ----------------

def _variant_stem(algo: str, feat: str, drop_flat: bool) -> str:
    """Artifact stem for a (base algo, feature set, flat-toggle) variant.

    v1+flat-incl → "<algo>" (production); v1+no-flat → "exp_noflat_<algo>";
    v2+flat-incl → "exp_v2_<algo>"; v2+no-flat → "exp_noflat_v2_<algo>".
    """
    if feat == "v1":
        return f"exp_noflat_{algo}" if drop_flat else algo
    return f"exp_noflat_v2_{algo}" if drop_flat else f"exp_v2_{algo}"


def _model_joblib(algo: str, feat: str, drop_flat: bool) -> Path:
    return PROC_DIR / f"{_variant_stem(algo, feat, drop_flat)}_model.joblib"


def _model_exists(algo: str, feat: str, drop_flat: bool) -> bool:
    if algo == "hmm":
        return (PROC_DIR / "exp_regime_binary_hmm.joblib").exists()
    return _model_joblib(algo, feat, drop_flat).exists()


def _model_selectors(key_prefix: str) -> tuple[str, str, bool, str]:
    """Render Model + Feature-set + drop-flat selectors; return (algo, feat, drop_flat, display)."""
    display = st.selectbox("Model", MODEL_OPTIONS, key=f"{key_prefix}_model")
    algo    = DISPLAY_ALGO[display]
    is_hmm  = algo == "hmm"
    c1, c2  = st.columns(2)
    feat = c1.radio(
        "Feature set", ["v1", "v2"],
        format_func=lambda v: _FEATURE_LABELS[v],
        horizontal=True, disabled=is_hmm, key=f"{key_prefix}_feat",
    )
    drop_flat = c2.checkbox(
        "Drop flat (Close==Open) training bars", value=False,
        disabled=is_hmm, key=f"{key_prefix}_flat",
        help="Exclude bars where Close == Open from the training set "
             "(focus on pure up/down). Test set is never filtered.",
    )
    if is_hmm:                       # HMM-regime is always v2 / no-flat
        feat, drop_flat = "v2", True
        c1.caption("HMM-regime uses the 49-feature set and drops flats by design.")
    return algo, feat, drop_flat, display


def _show_eval(y_true: np.ndarray | None, y_pred: np.ndarray) -> None:
    """Render standardised statistics (+ confusion) from predictions in the GUI."""
    import src.statistics as _stats  # noqa: E402
    n_up = int((y_pred == 1).sum())
    cols = st.columns(3)
    cols[0].metric("Total", f"{len(y_pred):,}")
    cols[1].metric("Down (0)", f"{len(y_pred) - n_up:,}")
    cols[2].metric("Up (1)", f"{n_up:,}")
    if y_true is None:
        st.info("No Open/Close ground truth in the file — showing prediction counts only.")
        return
    res = _stats.compute(np.asarray(y_true), np.asarray(y_pred))
    m = st.columns(4)
    m[0].metric("Accuracy", f"{res['accuracy']:.4f}")
    m[1].metric("Macro F1", f"{res['macro_f1']:.4f}")
    m[2].metric("MCC", f"{res['mcc']:.4f}")
    m[3].metric("Weighted F1", f"{res['weighted_f1']:.4f}")
    st.markdown("**Per-class & confusion matrix:**")
    st.markdown(_stats.format_markdown(res))


# --- SVM-inference guards (RBF predict is O(n_support_vectors × n_rows)) ----

_SVM_PREDICT_RATE = 2.3e8  # kernel-dim ops/s, calibrated: 253k SV × 49 feat × 1196 rows ≈ 64.5 s


@st.cache_resource(show_spinner=False)
def _load_svm_cached(path_str: str):
    """Load an SVM joblib once (cached across reruns — these files are large)."""
    from src.models.svm import load as svm_load  # noqa: E402
    return svm_load(Path(path_str))


def _svm_sv_count(model) -> tuple[int, int]:
    """(n_support_vectors, n_features) for a loaded SVMModel bundle."""
    sv = model["clf"].support_vectors_
    return int(sv.shape[0]), int(sv.shape[1])


def _estimate_svm_seconds(n_sv: int, n_rows: int, n_feat: int) -> float:
    """Rough single-threaded RBF-predict time estimate (seconds)."""
    return n_sv * n_rows * n_feat / _SVM_PREDICT_RATE


def _fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"~{seconds:.0f}s"
    return f"~{seconds / 60:.1f} min"


def _predict_chunked(predict_fn, features, label: str, chunk: int = 2000) -> np.ndarray:
    """Predict in row-chunks with a live progress bar (cancellable between chunks)."""
    n = len(features)
    out = np.empty(n, dtype=int)
    bar = st.progress(0.0, text=f"{label} 0/{n:,}")
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        out[start:end] = predict_fn(features.iloc[start:end])
        bar.progress(end / n, text=f"{label} {end:,}/{n:,}")
    bar.empty()
    return out


def _render_hyperparams(algo: str, current: dict) -> dict:
    """Render per-model hyperparameter widgets and return the updated params dict."""
    p: dict[str, Any] = {}

    if algo == "rf":
        p["n_estimators"] = st.number_input(
            "Number of trees", min_value=1, max_value=5000,
            value=int(current.get("n_estimators", 500)), step=50,
        )
        raw_depth = current.get("max_depth", None)
        depth_in = st.number_input(
            "Max depth (0 = unlimited)", min_value=0, max_value=100,
            value=0 if raw_depth is None else int(raw_depth), step=1,
        )
        p["max_depth"]        = None if depth_in == 0 else int(depth_in)
        p["min_samples_leaf"] = st.number_input(
            "Min samples per leaf", min_value=1, max_value=500,
            value=int(current.get("min_samples_leaf", 5)), step=1,
        )
        p["max_features"] = st.selectbox(
            "Max features", ["sqrt", "log2"],
            index=0 if current.get("max_features", "sqrt") == "sqrt" else 1,
        )
        p["class_weight"] = st.selectbox(
            "Class weight", ["balanced", "balanced_subsample"],
            index=0 if current.get("class_weight", "balanced") == "balanced" else 1,
        )
        p["oob_score"]  = bool(current.get("oob_score", True))
        p["bootstrap"]  = bool(current.get("bootstrap", True))
        p["n_jobs"]     = -1

    elif algo == "gbm":
        p["n_estimators"] = st.number_input(
            "Number of estimators", min_value=1, max_value=5000,
            value=int(current.get("n_estimators", 500)), step=50,
        )
        p["learning_rate"] = st.number_input(
            "Learning rate", min_value=0.001, max_value=1.0,
            value=float(current.get("learning_rate", 0.05)),
            step=0.005, format="%.3f",
        )
        p["max_depth"] = st.number_input(
            "Max depth", min_value=1, max_value=20,
            value=int(current.get("max_depth", 4)), step=1,
        )
        p["subsample"] = st.slider(
            "Subsample ratio", min_value=0.1, max_value=1.0,
            value=float(current.get("subsample", 0.8)), step=0.05,
        )
        p["colsample_bytree"] = st.slider(
            "Column sample per tree", min_value=0.1, max_value=1.0,
            value=float(current.get("colsample_bytree", 0.8)), step=0.05,
        )
        p["reg_lambda"]       = st.number_input(
            "L2 regularization", min_value=0.0, max_value=100.0,
            value=float(current.get("reg_lambda", 1.0)), step=0.1,
        )
        p["min_child_weight"] = int(current.get("min_child_weight", 1))
        p["objective"]        = str(current.get("objective", "binary:logistic"))
        p["eval_metric"]      = str(current.get("eval_metric", "logloss"))
        p["n_jobs"]           = -1

    elif algo == "svm":
        st.warning(
            "SVM training can take 2–4 hours on large datasets. "
            "Consider reducing the training size percentage below 20%."
        )
        p["C"] = st.number_input(
            "C (regularization)", min_value=0.001, max_value=1000.0,
            value=float(current.get("C", 1.0)), step=0.1, format="%.3f",
        )
        p["kernel"] = st.selectbox(
            "Kernel", ["rbf", "linear", "poly"],
            index=["rbf", "linear", "poly"].index(current.get("kernel", "rbf")),
        )
        p["gamma"] = st.selectbox(
            "Gamma", ["scale", "auto"],
            index=0 if current.get("gamma", "scale") == "scale" else 1,
        )
        p["class_weight"] = "balanced"
        p["cache_size"]   = st.number_input(
            "Cache size (MB)", min_value=100, max_value=8000,
            value=int(current.get("cache_size", 500)), step=100,
        )
        p["probability"] = False

    elif algo == "baseline":
        p["max_iter"] = st.number_input(
            "Max iterations", min_value=100, max_value=10000,
            value=int(current.get("max_iter", 1000)), step=100,
        )

    return p


def _write_statistics(
    preds: np.ndarray,
    y_true: np.ndarray | None,
    meta: dict,
    input_file_name: str,
    n_rows: int,
) -> None:
    from src.evaluate import (
        accuracy as _acc,
        confusion as _conf,
        recall as _rec,
    )

    n_up    = int(preds.sum())
    n_down  = int(len(preds) - n_up)
    pct_up  = n_up  / len(preds) * 100
    pct_dn  = n_down / len(preds) * 100

    lines = [
        "# Prediction Statistics\n\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        "## Model\n\n",
        f"- Algorithm: {meta['display_name']}\n",
        f"- Trained on: {meta['data_file']} ({meta['n_rows']:,} rows)\n",
        f"- Train size: {meta['train_size']*100:.0f}%\n\n",
        "## Input\n\n",
        f"- File: {input_file_name}\n",
        f"- Rows processed: {n_rows:,}\n\n",
        "## Predictions\n\n",
        "| Label | Count | Percentage |\n",
        "|-------|-------|------------|\n",
        f"| Down (0) | {n_down:,} | {pct_dn:.1f}% |\n",
        f"| Up (1) | {n_up:,} | {pct_up:.1f}% |\n\n",
    ]

    if y_true is not None:
        acc = _acc(y_true, preds)
        rec = _rec(y_true, preds)
        cm  = _conf(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        lines += [
            "## Performance (ground truth available)\n\n",
            "| Metric | Value |\n",
            "|--------|-------|\n",
            f"| Accuracy | {acc:.4f} |\n",
            f"| Recall (Up) | {rec:.4f} |\n\n",
            "**Confusion Matrix** (rows = actual, cols = predicted):\n\n",
            "|  | Predicted Down | Predicted Up |\n",
            "|--|----------------|--------------|\n",
            f"| **Actual Down** | {tn:,} | {fp:,} |\n",
            f"| **Actual Up** | {fn:,} | {tp:,} |\n",
        ]

    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text("".join(lines))


# ---------------------------------------------------------------------------
# Page config and CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Futures Price Direction Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Accent colour for expander headers */
    div[data-testid="stExpander"] summary p {
        color: #8EE3EF;
        font-weight: 600;
    }
    /* Mauve left-border on error alerts */
    div[data-testid="stAlert"][data-baseweb="notification"] {
        border-left: 4px solid #7E4E60;
    }
    /* Horizontal rule in mauve */
    hr { border: none; border-top: 2px solid #7E4E60; margin: 1.2rem 0; }
    /* Tighten metric label */
    div[data-testid="stMetricLabel"] { font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

for _k in ("train_result", "train_error", "predict_result", "predict_error"):
    if _k not in st.session_state:
        st.session_state[_k] = None

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

from src.config import load_config, model_params as _model_params  # noqa: E402

try:
    cfg = load_config(CONFIG_PATH)
except FileNotFoundError:
    st.error("config.yaml not found. Run the app from the project root directory.")
    st.stop()

# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.title("Futures Price Direction Predictor")
st.caption("Predict whether a futures contract closes up or down relative to its open price.")
st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_train, tab_predict = st.tabs(["Training", "Prediction"])

# ===========================================================================
# TRAINING TAB
# ===========================================================================

with tab_train:
    st.subheader("Train a Model")
    st.markdown(
        "Upload a raw OHLCV data file, configure the model and parameters, then train. "
        "If a saved model already exists it will be overwritten."
    )

    # --- file upload -------------------------------------------------------
    uploaded_train = st.file_uploader(
        "Upload training data (.csv)",
        type=["csv"],
        key="train_uploader",
        help="Required columns: Date and Time, Open, Close, High, Low, VWAP, Symbol.",
    )

    st.markdown("---")

    # --- model + feature-set + flat selectors ------------------------------
    algo, feat, drop_flat, model_display = _model_selectors("train")
    variant_label = ("HMM-regime" if algo == "hmm"
                     else f"{model_display} · {feat}"
                          + (" · no-flat" if drop_flat else ""))
    if _model_exists(algo, feat, drop_flat):
        st.caption(f"✓ A saved **{variant_label}** model exists (will be overwritten).")
    else:
        st.caption(f"No saved **{variant_label}** model yet.")

    # --- train size --------------------------------------------------------
    default_pct = int(cfg["data"].get("train_size", 0.5) * 100)
    train_pct = st.number_input(
        "Training size (%)",
        min_value=1, max_value=99, value=default_pct, step=5,
        help=("Percentage of uploaded rows used for training "
              "(time-ordered — first N%). The remainder is the test set."),
    )

    # --- advanced hyperparameters (HMM trains per-regime RFs → use rf knobs) -
    hp_algo = "rf" if algo == "hmm" else algo
    current_params = _model_params(cfg, hp_algo)
    with st.expander("Advanced Hyperparameters", expanded=False):
        new_params = _render_hyperparams(hp_algo, current_params)

    st.markdown("---")

    # --- train button ------------------------------------------------------
    if uploaded_train is None:
        st.info("Upload a training data file to enable training.")

    if st.button("Train Model", disabled=(uploaded_train is None), type="primary"):
        UPLOAD_TRAIN.parent.mkdir(parents=True, exist_ok=True)
        UPLOAD_TRAIN.write_bytes(uploaded_train.getvalue())

        import copy as _copy
        cfg2 = _copy.deepcopy(cfg)
        cfg2["data"]["path"] = str(UPLOAD_TRAIN)
        cfg2["data"]["train_size"] = train_pct / 100
        cfg2.setdefault("models", {})[hp_algo] = new_params

        try:
            with st.spinner(f"Training {variant_label} — this may take a while…"):
                if algo == "hmm":
                    from src.models import regime_binary  # noqa: E402
                    regime_binary.run(config=cfg2)
                    d = np.load(PROC_DIR / "exp_regime_binary_predictions.npz")
                    yt, yp = d["y_true"], d["y_pred"]
                elif feat == "v1" and not drop_flat:
                    from src import pipeline  # noqa: E402
                    r = pipeline.run(algo, data_path=UPLOAD_TRAIN,
                                     force_retrain=True, config=cfg2)
                    yt, yp = r["y_true"], r["y_pred"]
                else:
                    from src import binary_suite  # noqa: E402
                    from src.features_v2 import build_features_v2  # noqa: E402
                    stem   = _variant_stem(algo, feat, drop_flat)
                    prefix = stem[: -(len(algo) + 1)]
                    binary_suite.run(
                        config=cfg2, algos=(algo,),
                        build_features_fn=(build_features_v2 if feat == "v2" else None),
                        drop_flat=drop_flat, prefix=prefix,
                    )
                    d = np.load(PROC_DIR / f"{stem}_predictions.npz")
                    yt, yp = d["y_true"], d["y_pred"]

            st.session_state.train_result = {
                "display": variant_label, "y_true": yt, "y_pred": yp,
            }
            st.session_state.train_error = None
        except Exception as exc:  # noqa: BLE001
            st.session_state.train_error  = str(exc)
            st.session_state.train_result = None

    # --- training result ---------------------------------------------------
    if st.session_state.train_result is not None:
        res = st.session_state.train_result
        st.success(f"Training complete — {res['display']} (held-out test split).")
        _show_eval(res["y_true"], res["y_pred"])

    if st.session_state.train_error is not None:
        st.error(f"Training failed: {st.session_state.train_error}")


# ===========================================================================
# PREDICTION TAB
# ===========================================================================

with tab_predict:
    st.subheader("Run Predictions")
    st.markdown(
        "Upload an OHLCV data file. Features will be built automatically and "
        "predictions made using the selected model. If the file contains "
        "Open and Close columns, accuracy metrics are computed as well."
    )

    # --- model + feature-set + flat selectors ------------------------------
    p_algo, p_feat, p_drop_flat, p_display = _model_selectors("pred")
    p_variant = ("HMM-regime" if p_algo == "hmm"
                 else f"{p_display} · {p_feat}" + (" · no-flat" if p_drop_flat else ""))
    p_available = _model_exists(p_algo, p_feat, p_drop_flat)
    if not p_available:
        st.warning(f"No saved **{p_variant}** model — train it in the Training tab first.")

    txn_cost = st.number_input(
        "Transaction cost per bar (for the backtest; 0 = frictionless, 0.0001 = 1 bp)",
        min_value=0.0, value=0.0, step=0.0001, format="%.4f", key="pred_cost",
    )

    # --- SVM is the only slow predictor: show a time estimate + row cap -----
    svm_cap = 0
    if p_algo == "svm" and p_available:
        mpath_sel = _model_joblib(p_algo, p_feat, p_drop_flat)
        try:
            n_sv, n_feat = _svm_sv_count(_load_svm_cached(str(mpath_sel)))
        except Exception:
            n_sv, n_feat = 0, (49 if p_feat == "v2" else 20)
        per_row = _estimate_svm_seconds(n_sv, 1, n_feat) if n_sv else 0.0
        # Default cap targets ~90 s of prediction time, rounded to 500 rows.
        default_cap = (int(round(max(1000, 90 / per_row) / 500)) * 500
                       if per_row else 25000)
        st.warning(
            f"RBF-SVM inference is slow — this model has **{n_sv:,} support vectors** "
            f"(≈ {per_row * 1000:.1f} ms/row, single-threaded, not parallelisable). "
            f"E.g. {default_cap:,} rows ≈ {_fmt_duration(per_row * default_cap)}; "
            f"250,000 rows ≈ {_fmt_duration(per_row * 250_000)}."
        )
        svm_cap = int(st.number_input(
            "Max rows to predict (SVM) — 0 = no cap (predict all rows)",
            min_value=0, value=default_cap, step=500, key="svm_cap",
            help="SVM predicts only the first N rows; statistics & backtest use that "
                 "subset. Raise it or set 0 at your own time cost.",
        ))
        if svm_cap == 0:
            st.error("No cap set — predicting all rows may take many minutes.")
        else:
            st.caption(f"Selected: {svm_cap:,} rows → est {_fmt_duration(per_row * svm_cap)}.")

    # --- file upload + validation ------------------------------------------
    uploaded_pred = st.file_uploader(
        "Upload prediction data (.csv)",
        type=["csv"],
        key="pred_uploader",
        help="Required columns: Date and Time, Open, Close, High, Low, VWAP.",
    )

    cols_ok = False
    if uploaded_pred is not None:
        try:
            peek = pd.read_csv(uploaded_pred, nrows=5)
            uploaded_pred.seek(0)
            missing = REQUIRED_COLS - set(peek.columns)
            if missing:
                st.error(
                    f"Column mismatch — missing: {', '.join(sorted(missing))}. "
                    "Please select a valid OHLCV file."
                )
            else:
                st.success("File validated. Required columns present.")
                cols_ok = True
        except Exception as exc:
            st.error(f"Could not read file: {exc}")

    st.markdown("---")

    V2_EXTRA = {"Up Ticks", "Down Ticks", "Tick Count", "Volume"}
    predict_disabled = not cols_ok or not p_available
    if st.button("Run Prediction", disabled=predict_disabled, type="primary"):
        try:
            UPLOAD_PRED.parent.mkdir(parents=True, exist_ok=True)
            UPLOAD_PRED.write_bytes(uploaded_pred.getvalue())

            from src.load import load_raw          # noqa: E402
            from src.labels import build_labels    # noqa: E402

            with st.spinner("Building features…"):
                df = load_raw(UPLOAD_PRED)
                use_v2 = (p_feat == "v2") or (p_algo == "hmm")
                if use_v2 and not V2_EXTRA.issubset(df.columns):
                    raise ValueError(
                        "v2 / HMM models also need tick columns: "
                        + ", ".join(sorted(V2_EXTRA))
                    )
                if use_v2:
                    from src.features_v2 import build_features_v2  # noqa: E402
                    features = build_features_v2(df)
                else:
                    from src.features import build_features        # noqa: E402
                    features = build_features(df)

            raw_align = df.iloc[4:].reset_index(drop=True)

            if p_algo == "svm":
                # Capped + chunked (progress + cancellable) — RBF predict is slow.
                from src.models.svm import predict as svm_predict   # noqa: E402
                if svm_cap and len(features) > svm_cap:
                    st.info(f"Predicting the first {svm_cap:,} of {len(features):,} rows "
                            "(SVM cap). Statistics & backtest use this subset.")
                    features = features.iloc[:svm_cap]
                _svm = _load_svm_cached(str(_model_joblib(p_algo, p_feat, p_drop_flat)))
                preds = _predict_chunked(
                    lambda X: svm_predict(_svm, X), features,
                    label="SVM predicting", chunk=2000,
                )
            else:
                with st.spinner(f"Predicting with {p_variant}…"):
                    if p_algo == "hmm":
                        from src.models import regime_binary            # noqa: E402
                        preds = regime_binary.predict(features, regime_binary.load_bundle())
                    else:
                        mpath = _model_joblib(p_algo, p_feat, p_drop_flat)
                        from src.models import baseline, rf             # noqa: E402
                        from src.models.gbm import load as gbm_load, predict as gbm_predict  # noqa: E402
                        if p_algo == "baseline":
                            preds = baseline.predict(baseline.load(mpath), features)
                        elif p_algo == "rf":
                            preds = rf.predict(rf.load(mpath), features)
                        else:  # gbm
                            preds = gbm_predict(gbm_load(mpath), features)

            # Align the raw slice to the predictions (handles the SVM row cap).
            raw_align = raw_align.iloc[:len(preds)].reset_index(drop=True)

            # ground truth + per-bar returns (for the backtest)
            y_true = None
            if {"Open", "Close"}.issubset(raw_align.columns):
                ys = build_labels(raw_align)
                if len(ys) == len(preds):
                    y_true = ys.to_numpy()
            bar_returns = ((raw_align["Close"] - raw_align["Open"])
                           / raw_align["Open"]).to_numpy()
            timestamps = (raw_align["Date and Time"].to_numpy()
                          if "Date and Time" in raw_align.columns else None)

            st.session_state.predict_result = {
                "display": p_variant, "algo": p_algo, "feat": p_feat,
                "drop_flat": p_drop_flat, "file": uploaded_pred.name,
                "preds": np.asarray(preds), "y_true": y_true,
                "bar_returns": bar_returns, "timestamps": timestamps,
                "cost": float(txn_cost),
            }
            st.session_state.predict_error = None
        except Exception as exc:  # noqa: BLE001
            st.session_state.predict_error  = str(exc)
            st.session_state.predict_result = None

    # --- results: statistics + backtest in one place -----------------------
    if st.session_state.predict_result is not None:
        pr = st.session_state.predict_result
        import src.statistics as _stats  # noqa: E402

        st.success(f"Prediction complete — {pr['display']} on {pr['file']}.")

        st.markdown("### Statistics")
        _show_eval(pr["y_true"], pr["preds"])

        st.markdown("### Backtest — $1,000, reinvested each bar")
        bt = _stats.backtest(
            pr["preds"], pr["bar_returns"], timestamps=pr["timestamps"],
            transaction_cost=pr["cost"], name=pr["display"],
        )
        b = st.columns(4)
        b[0].metric("Final equity", f"${bt['final_equity']:,.2f}")
        b[1].metric("Total return", f"{bt['total_return']:.2%}")
        b[2].metric("Max drawdown", f"{bt['max_drawdown']:.2%}")
        b[3].metric("Ann. Sharpe", f"{bt['annualized_sharpe']:.2f}")
        b2 = st.columns(2)
        b2[0].metric("Passive final equity", f"${bt['passive_final_equity']:,.2f}")
        b2[1].metric("Strategy − Passive (return)",
                     f"{(bt['total_return'] - bt['passive_total_return']):+.2%}")

        plot_path = Path("docs/notes") / (
            f"backtest_gui_{pr['algo']}_{pr['feat']}"
            f"{'_noflat' if pr['drop_flat'] else ''}.png"
        )
        _stats.plot_equity_curve(bt, plot_path)
        st.image(str(plot_path),
                 caption="Equity curve — strategy vs passive buy & hold")

        # downloadable combined report
        report = "# Prediction Report\n\n"
        if pr["y_true"] is not None:
            report += _stats.format_markdown(
                _stats.compute(pr["y_true"], pr["preds"], name=pr["display"])
            ) + "\n\n---\n\n"
        report += _stats.format_backtest_markdown(bt)
        st.download_button("Download report (.md)", data=report,
                           file_name="prediction_report.md", mime="text/markdown")

    if st.session_state.predict_error is not None:
        st.error(f"Prediction failed: {st.session_state.predict_error}")
