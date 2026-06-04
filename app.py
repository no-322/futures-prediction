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
}
DISPLAY_ALGO: dict[str, str] = {v: k for k, v in ALGO_DISPLAY.items()}

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

    # --- model selection ---------------------------------------------------
    col_model, col_status = st.columns([4, 1])
    with col_model:
        model_display = st.selectbox(
            "Model",
            list(DISPLAY_ALGO.keys()),
            index=0,
            key="model_select",
        )
    algo = DISPLAY_ALGO[model_display]

    with col_status:
        st.markdown("<br>", unsafe_allow_html=True)
        if MODEL_PATHS[algo].exists():
            st.success("Saved model exists")
        else:
            st.warning("No saved model")

    # --- train size --------------------------------------------------------
    default_pct = int(cfg["data"].get("train_size", 0.5) * 100)
    train_pct = st.number_input(
        "Training size (%)",
        min_value=1, max_value=99,
        value=default_pct,
        step=5,
        help=(
            "Percentage of uploaded rows used for training "
            "(time-ordered — first N%). The remainder is the test set."
        ),
    )

    # --- advanced hyperparameters ------------------------------------------
    current_params = _model_params(cfg, algo)
    with st.expander("Advanced Hyperparameters", expanded=False):
        new_params = _render_hyperparams(algo, current_params)

    st.markdown("---")

    # --- train button ------------------------------------------------------
    if uploaded_train is None:
        st.info("Upload a training data file to enable training.")

    if st.button("Train Model", disabled=(uploaded_train is None), type="primary"):
        UPLOAD_TRAIN.parent.mkdir(parents=True, exist_ok=True)
        UPLOAD_TRAIN.write_bytes(uploaded_train.getvalue())

        # persist updated config
        cfg["data"]["train_size"] = train_pct / 100
        cfg["models"][algo] = new_params
        try:
            CONFIG_PATH.write_text(
                yaml.dump(cfg, default_flow_style=False, sort_keys=False, allow_unicode=True)
            )
        except Exception:
            pass  # non-fatal — params still used in-memory this run

        from src import pipeline  # noqa: E402 (deferred to avoid slow startup)

        try:
            with st.spinner(f"Training {model_display} — this may take a while…"):
                result = pipeline.run(
                    algo,
                    data_path=UPLOAD_TRAIN,
                    force_retrain=True,
                    config=cfg,
                )
            st.session_state.train_result = result
            st.session_state.train_error  = None
        except Exception as exc:
            st.session_state.train_error  = str(exc)
            st.session_state.train_result = None

    # --- training result ---------------------------------------------------
    if st.session_state.train_result is not None:
        res = st.session_state.train_result
        st.success("Training complete.")
        m = res["metrics"]
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{m['accuracy']:.4f}")
        c2.metric("Recall (Up)", f"{m['recall']:.4f}")
        st.markdown("**Confusion Matrix:**")
        st.dataframe(_confusion_df(m["confusion"]), use_container_width=False)

    if st.session_state.train_error is not None:
        st.error(f"Training failed: {st.session_state.train_error}")

    st.markdown("---")
    _render_last_run_info(algo)


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

    # --- model selector (based on saved joblobs, not metadata) -------------
    saved_models = {
        display: algo
        for display, algo in DISPLAY_ALGO.items()
        if MODEL_PATHS[algo].exists()
    }
    if not saved_models:
        st.warning(
            "No saved models found in data/processed/. "
            "Train a model in the Training tab first."
        )
        pred_algo = None
    else:
        pred_model_display = st.selectbox(
            "Model to use for prediction",
            list(saved_models.keys()),
            key="pred_model_select",
        )
        pred_algo = saved_models[pred_model_display]

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

    predict_disabled = not cols_ok or pred_algo is None
    if st.button("Run Prediction", disabled=predict_disabled, type="primary"):
        from src.features import build_features   # noqa: E402
        from src.labels import build_labels       # noqa: E402
        from src.load import load_raw             # noqa: E402
        from src.models import baseline, rf       # noqa: E402
        from src.models.gbm import load as gbm_load, predict as gbm_predict  # noqa: E402
        from src.models.svm import load as svm_load, predict as svm_predict  # noqa: E402

        # use metadata if available, otherwise fall back to minimal info
        pred_meta = _load_metadata()
        if pred_meta is None or pred_meta.get("algo") != pred_algo:
            pred_meta = {
                "algo":         pred_algo,
                "display_name": ALGO_DISPLAY[pred_algo],
                "data_file":    "unknown",
                "n_rows":       0,
                "train_size":   cfg["data"].get("train_size", 0.5),
            }

        try:
            UPLOAD_PRED.parent.mkdir(parents=True, exist_ok=True)
            UPLOAD_PRED.write_bytes(uploaded_pred.getvalue())

            with st.spinner("Building features…"):
                df = load_raw(UPLOAD_PRED)
                features = build_features(df)

            algo_p = pred_algo
            mpath  = MODEL_PATHS[algo_p]

            with st.spinner(f"Loading {ALGO_DISPLAY[algo_p]} model…"):
                if algo_p == "baseline":
                    model = baseline.load(mpath)
                    preds = baseline.predict(model, features)
                elif algo_p == "rf":
                    model = rf.load(mpath)
                    preds = rf.predict(model, features)
                elif algo_p == "gbm":
                    model = gbm_load(mpath)
                    preds = gbm_predict(model, features)
                else:  # svm
                    model = svm_load(mpath)
                    preds = svm_predict(model, features)

            # ground truth labels (if Open + Close are present)
            y_true: np.ndarray | None = None
            if {"Open", "Close"}.issubset(df.columns):
                raw_align = df.iloc[4:].reset_index(drop=True)
                y_series = build_labels(raw_align)
                if len(y_series) == len(preds):
                    y_true = y_series.to_numpy()

            _write_statistics(preds, y_true, pred_meta, uploaded_pred.name, len(features))

            st.session_state.predict_result = {
                "preds": preds,
                "y_true": y_true,
                "algo": algo_p,
            }
            st.session_state.predict_error = None

        except Exception as exc:
            st.session_state.predict_error  = str(exc)
            st.session_state.predict_result = None

    # --- prediction results ------------------------------------------------
    if st.session_state.predict_result is not None:
        pr    = st.session_state.predict_result
        preds = pr["preds"]
        y_true = pr["y_true"]

        st.success("Prediction complete.")
        n_up   = int(preds.sum())
        n_down = int(len(preds) - n_up)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total predictions", f"{len(preds):,}")
        c2.metric("Down (0)", f"{n_down:,} ({n_down/len(preds)*100:.1f}%)")
        c3.metric("Up (1)",   f"{n_up:,} ({n_up/len(preds)*100:.1f}%)")

        if y_true is not None:
            from src.evaluate import accuracy as _a, recall as _r, confusion as _c
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{_a(y_true, preds):.4f}")
            c2.metric("Recall (Up)", f"{_r(y_true, preds):.4f}")
            st.markdown("**Confusion Matrix:**")
            st.dataframe(_confusion_df(_c(y_true, preds)), use_container_width=False)

    if st.session_state.predict_error is not None:
        st.error(f"Prediction failed: {st.session_state.predict_error}")

    st.markdown("---")

    # --- statistics report -------------------------------------------------
    with st.expander("View Statistics Report"):
        if STATS_PATH.exists():
            stats_text = STATS_PATH.read_text()
            st.markdown(stats_text)
            st.download_button(
                "Download statistics.md",
                data=stats_text,
                file_name="statistics.md",
                mime="text/markdown",
            )
        else:
            st.info("No statistics report yet. Run a prediction first.")

    st.markdown("---")
    _render_last_run_info(pred_algo or "rf")
