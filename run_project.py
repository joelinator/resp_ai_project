#!/usr/bin/env python3
"""End-to-end final project pipeline for Human-AI collaboration in education."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PAPER_DIR = ROOT / "paper"
FORMATTING_DIR = ROOT / "Formatting_Instructions_For_NeurIPS_2026"

RAW_DATA_PATH = DATA_DIR / "assistments_2009_raw.csv"
PROCESSED_DATA_PATH = DATA_DIR / "assistments_2009_processed_with_gender.csv"
LOG_PATH = RESULTS_DIR / "run.log"

GDRIVE_FILE_ID = "1NNXHFRxcArrU0ZJSb9BIL56vmUt5FhlE"
SEED = 42

#deployment-cost logic:
# Action A (AI):          loss in [0,1] + cost 0.0
# Action B (Human):       loss in [0,1] + cost 0.8
# Action C (Diagnostic):  loss in [0,1] + cost 0.4
DEPLOYMENT_COST_AI = 0.0
DEPLOYMENT_COST_HUMAN = 0.8
DEPLOYMENT_COST_DIAGNOSTIC = 0.4


@dataclass
class ActionOutcomes:
    ai_pred: np.ndarray
    human_pred: np.ndarray
    diagnostic_pred: np.ndarray
    ai_loss: np.ndarray
    human_loss: np.ndarray
    diagnostic_loss: np.ndarray
    ai_total_cost: np.ndarray
    human_total_cost: np.ndarray
    diagnostic_total_cost: np.ndarray
    human_accuracy: np.ndarray


class LinUCB:
    """Simple LinUCB with Sherman-Morrison updates."""

    def __init__(self, n_actions: int, d: int, alpha: float = 0.7, reg: float = 1.0):
        self.n_actions = n_actions
        self.d = d
        self.alpha = alpha
        self.reg = reg
        self.A_inv = [np.eye(d) / reg for _ in range(n_actions)]
        self.b = [np.zeros(d) for _ in range(n_actions)]

    def select(self, x: np.ndarray, ai_penalty: float) -> int:
        scores: List[float] = []
        for a in range(self.n_actions):
            a_inv = self.A_inv[a]
            theta = a_inv @ self.b[a]
            mean = float(theta @ x)
            uncertainty = self.alpha * math.sqrt(max(1e-12, float(x @ a_inv @ x)))
            score = mean + uncertainty
            if a == 0:  # AI action
                score -= ai_penalty
            scores.append(score)
        return int(np.argmax(scores))

    def update(self, action: int, x: np.ndarray, reward: float) -> None:
        a_inv = self.A_inv[action]
        ax = a_inv @ x
        denom = 1.0 + float(x @ ax)
        self.A_inv[action] = a_inv - np.outer(ax, ax) / max(1e-9, denom)
        self.b[action] = self.b[action] + reward * x


def configure_logging() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
    )


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)


def download_from_google_drive(file_id: str, destination: Path) -> None:
    logging.info("Downloading dataset from Google Drive...")
    url = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(url, params={"id": file_id}, stream=True, timeout=120)
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        response = session.get(
            url,
            params={"id": file_id, "confirm": token},
            stream=True,
            timeout=120,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
    logging.info("Saved raw dataset to %s", destination)


def discover_existing_csv() -> Path | None:
    candidates = sorted(DATA_DIR.glob("*.csv"))
    if candidates:
        for c in candidates:
            if c.name == PROCESSED_DATA_PATH.name:
                continue
            return c
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    renamed = {}
    for col in df.columns:
        clean = col.strip().lower()
        clean = clean.replace(" ", "_").replace("-", "_")
        clean = re.sub(r"[^a-z0-9_]+", "", clean)
        renamed[col] = clean
    return df.rename(columns=renamed)


def load_raw_data(max_rows: int | None = None) -> pd.DataFrame:
    if PROCESSED_DATA_PATH.exists():
        logging.info("Using existing processed file: %s", PROCESSED_DATA_PATH)
        df = pd.read_csv(PROCESSED_DATA_PATH, low_memory=False)
        return normalize_columns(df)

    existing_csv = discover_existing_csv()
    if existing_csv is None:
        download_from_google_drive(GDRIVE_FILE_ID, RAW_DATA_PATH)
        raw_path = RAW_DATA_PATH
    else:
        raw_path = existing_csv
        logging.info("Using existing raw CSV: %s", raw_path)

    df = pd.read_csv(raw_path, low_memory=False, encoding="ISO-8859-15")
    if max_rows is not None and len(df) > max_rows:
        logging.info("Subsampling raw rows from %d to %d", len(df), max_rows)
        df = df.sample(n=max_rows, random_state=SEED).sort_index()
    return normalize_columns(df)


def preprocess_data(df: pd.DataFrame, max_interactions: int) -> pd.DataFrame:
    logging.info("Preprocessing dataset...")
    required = ["user_id", "skill_id", "correct"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    work = df.copy()
    work["user_id"] = pd.to_numeric(work["user_id"], errors="coerce")
    work["skill_id"] = work["skill_id"].astype(str)
    work["correct"] = pd.to_numeric(work["correct"], errors="coerce")
    work = work.dropna(subset=["user_id", "skill_id", "correct"]).copy()
    work["user_id"] = work["user_id"].astype(np.int64)
    work["correct"] = work["correct"].clip(0, 1).astype(np.int64)

    if "attempt_count" not in work.columns:
        work["attempt_count"] = 1.0
    work["attempt_count"] = pd.to_numeric(work["attempt_count"], errors="coerce").fillna(1.0)
    work["attempt_count"] = work["attempt_count"].clip(lower=1.0, upper=10.0)

    if "order_id" in work.columns:
        sequence = pd.to_numeric(work["order_id"], errors="coerce")
        if sequence.isna().all():
            sequence = pd.Series(np.arange(len(work)), index=work.index)
    elif "timestamp" in work.columns:
        sequence = pd.to_datetime(work["timestamp"], errors="coerce")
        sequence = sequence.view("int64") // 10**9
        sequence = sequence.fillna(pd.Series(np.arange(len(work)), index=work.index))
    else:
        sequence = pd.Series(np.arange(len(work)), index=work.index)
    default_seq = pd.Series(np.arange(len(work)), index=work.index, dtype=float)
    work["sequence_idx"] = pd.to_numeric(sequence, errors="coerce")
    work["sequence_idx"] = work["sequence_idx"].fillna(default_seq).astype(float)

    unique_users = np.array(sorted(work["user_id"].unique()))
    gender_map = {uid: (i % 2) for i, uid in enumerate(unique_users)}
    work["gender"] = work["user_id"].map(gender_map).astype(np.int64)

    work = work.sort_values(["user_id", "sequence_idx"]).reset_index(drop=True)

    recency = work.groupby("user_id")["sequence_idx"].diff()
    recency = recency.replace([np.inf, -np.inf], np.nan)
    recency_fill = float(recency.median()) if not np.isnan(recency.median()) else 0.0
    work["recency"] = recency.fillna(recency_fill).clip(lower=0.0)

    grouped = work.groupby(["user_id", "skill_id"])
    cum = grouped["correct"].cumsum() - work["correct"]
    cnt = grouped.cumcount()
    global_correct_mean = float(work["correct"].mean())
    rolling = np.where(cnt > 0, cum / cnt, global_correct_mean)
    work["rolling_skill_correct"] = np.clip(rolling, 0.0, 1.0)

    if max_interactions > 0 and len(work) > max_interactions:
        logging.info("Limiting interactions from %d to %d", len(work), max_interactions)
        work = work.sample(n=max_interactions, random_state=SEED).sort_values("sequence_idx")
        work = work.reset_index(drop=True)

    work.to_csv(PROCESSED_DATA_PATH, index=False)
    logging.info("Saved processed data to %s (rows=%d)", PROCESSED_DATA_PATH, len(work))
    return work


def make_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    base_cols = ["rolling_skill_correct", "attempt_count", "recency"]
    X_num = df[base_cols].astype(np.float32).copy()
    X_num["recency"] = np.log1p(X_num["recency"])

    skill_dummies = pd.get_dummies(df["skill_id"], prefix="skill", dtype=np.float32)
    X = pd.concat([X_num, skill_dummies], axis=1).astype(np.float32)
    y = df["correct"].to_numpy(dtype=np.int64)
    return X, y


def build_strata(df: pd.DataFrame) -> pd.Series:
    user_bucket = (df["user_id"] % 10).astype(str)
    strata = df["skill_id"].astype(str) + "_" + user_bucket
    counts = strata.value_counts()
    rare = strata.map(counts) < 2
    strata.loc[rare] = df.loc[rare, "skill_id"].astype(str)
    counts = strata.value_counts()
    strata = strata.where(strata.map(counts) >= 2, "rare")
    return strata


def simulate_action_outcomes(
    y_true: np.ndarray,
    p_correct: np.ndarray,
    gender: np.ndarray,
    skill_id: np.ndarray,
) -> ActionOutcomes:
    rng = np.random.default_rng(SEED)
    ai_pred = (p_correct >= 0.5).astype(np.int64)
    ai_loss = (ai_pred != y_true).astype(float)

    skill_codes, _ = pd.factorize(pd.Series(skill_id).astype(str), sort=True)
    hard_skill_penalty = np.where(skill_codes % 7 == 0, -0.08, 0.0)
    gender_effect = np.where(gender == 1, -0.18, 0.05)
    human_accuracy = np.clip(
        0.88 + gender_effect + hard_skill_penalty + rng.normal(0.0, 0.03, size=len(y_true)),
        0.60,
        0.98,
    )
    human_correct = rng.binomial(1, human_accuracy).astype(np.int64)
    human_pred = np.where(human_correct == 1, y_true, 1 - y_true).astype(np.int64)
    human_loss = (human_pred != y_true).astype(float)

    ai_conf = np.maximum(p_correct, 1.0 - p_correct)
    diagnostic_accuracy = np.clip(0.66 + 0.20 * ai_conf + 0.10 * human_accuracy, 0.70, 0.99)
    diagnostic_correct = rng.binomial(1, diagnostic_accuracy).astype(np.int64)
    diagnostic_pred = np.where(diagnostic_correct == 1, y_true, 1 - y_true).astype(np.int64)
    # Loss is 1.0 iff student outcome is wrong, otherwise 0.0.
    diagnostic_loss = (diagnostic_pred != y_true).astype(float)

    # Total expected cost = wrong-answer loss + action deployment cost.
    ai_total_cost = ai_loss + DEPLOYMENT_COST_AI
    human_total_cost = human_loss + DEPLOYMENT_COST_HUMAN
    diagnostic_total_cost = diagnostic_loss + DEPLOYMENT_COST_DIAGNOSTIC

    return ActionOutcomes(
        ai_pred=ai_pred,
        human_pred=human_pred,
        diagnostic_pred=diagnostic_pred,
        ai_loss=ai_loss,
        human_loss=human_loss,
        diagnostic_loss=diagnostic_loss,
        ai_total_cost=ai_total_cost,
        human_total_cost=human_total_cost,
        diagnostic_total_cost=diagnostic_total_cost,
        human_accuracy=human_accuracy,
    )


def compute_fairness_rates(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray) -> Dict[str, float]:
    def safe_rate(num: float, den: float) -> float:
        return float(num / den) if den > 0 else 0.0

    out: Dict[str, float] = {}
    for g in [0, 1]:
        mask = group == g
        yt = y_true[mask]
        yp = y_pred[mask]
        tp = float(np.sum((yt == 1) & (yp == 1)))
        fn = float(np.sum((yt == 1) & (yp == 0)))
        fp = float(np.sum((yt == 0) & (yp == 1)))
        tn = float(np.sum((yt == 0) & (yp == 0)))
        out[f"tpr_{g}"] = safe_rate(tp, tp + fn)
        out[f"fpr_{g}"] = safe_rate(fp, fp + tn)
        out[f"fnr_{g}"] = safe_rate(fn, tp + fn)
    out["delta_eo_fnr"] = abs(out["fnr_0"] - out["fnr_1"])
    out["equalized_odds_diff"] = max(abs(out["tpr_0"] - out["tpr_1"]), abs(out["fpr_0"] - out["fpr_1"]))
    return out


def summarize_policy(logs: pd.DataFrame) -> Dict[str, float]:
    y_true = logs["y_true"].to_numpy()
    y_pred = logs["y_pred"].to_numpy()
    gender = logs["gender"].to_numpy()
    fairness = compute_fairness_rates(y_true, y_pred, gender)

    male_mask = logs["gender"] == 0
    female_mask = logs["gender"] == 1
    escalation_male = float((logs.loc[male_mask, "action"] == "HUMAN").mean()) if male_mask.any() else 0.0
    escalation_female = (
        float((logs.loc[female_mask, "action"] == "HUMAN").mean()) if female_mask.any() else 0.0
    )
    risk_male = float(logs.loc[male_mask, "loss"].mean()) if male_mask.any() else 0.0
    risk_female = float(logs.loc[female_mask, "loss"].mean()) if female_mask.any() else 0.0

    summary = {
        "coverage_ai": float((logs["action"] == "AI").mean()),
        "escalation_rate": float((logs["action"] == "HUMAN").mean()),
        "diagnostic_rate": float((logs["action"] == "DIAGNOSTIC").mean()),
        "average_loss": float(logs["loss"].mean()),
        "average_total_cost": float(logs["total_cost"].mean()),
        "accuracy": float((logs["y_true"] == logs["y_pred"]).mean()),
        "teacher_workload": float((logs["action"] == "HUMAN").mean()),
        "demographic_parity_escalation_diff": abs(escalation_male - escalation_female),
        "risk_gap": abs(risk_male - risk_female),
        **fairness,
    }
    return summary


def run_baseline_policy(
    confidence: np.ndarray,
    threshold: float,
    y_true: np.ndarray,
    gender: np.ndarray,
    outcomes: ActionOutcomes,
) -> pd.DataFrame:
    action = np.where(confidence >= threshold, "AI", "HUMAN")
    y_pred = np.where(action == "AI", outcomes.ai_pred, outcomes.human_pred)
    loss = np.where(action == "AI", outcomes.ai_loss, outcomes.human_loss)
    total_cost = np.where(action == "AI", outcomes.ai_total_cost, outcomes.human_total_cost)
    df = pd.DataFrame(
        {
            "action": action,
            "y_true": y_true,
            "y_pred": y_pred,
            "gender": gender,
            "loss": loss,
            "total_cost": total_cost,
            "confidence": confidence,
        }
    )
    return df


def make_bandit_context(test_df: pd.DataFrame, p_correct: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    attempt = np.log1p(test_df["attempt_count"].to_numpy(dtype=float))
    recency = np.log1p(test_df["recency"].to_numpy(dtype=float))
    rolling = test_df["rolling_skill_correct"].to_numpy(dtype=float)
    gender = test_df["gender"].to_numpy(dtype=float)

    skill_codes, _ = pd.factorize(test_df["skill_id"].astype(str), sort=True)
    n_buckets = 8
    buckets = skill_codes % n_buckets
    bucket_one_hot = np.eye(n_buckets)[buckets]

    context = np.column_stack(
        [
            np.ones(len(test_df)),
            p_correct,
            confidence,
            rolling,
            attempt / max(1e-6, attempt.std() + 1e-9),
            recency / max(1e-6, recency.std() + 1e-9),
            gender,
            bucket_one_hot,
        ]
    )
    return context.astype(float)


def _action_arrays(action_name: str, outcomes: ActionOutcomes) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if action_name == "AI":
        return outcomes.ai_pred, outcomes.ai_loss, outcomes.ai_total_cost
    if action_name == "HUMAN":
        return outcomes.human_pred, outcomes.human_loss, outcomes.human_total_cost
    return outcomes.diagnostic_pred, outcomes.diagnostic_loss, outcomes.diagnostic_total_cost


def run_proposed_policy(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    confidence: np.ndarray,
    outcomes: ActionOutcomes,
    alpha: float = 0.7,
    guardrail_batch: int = 500,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    context = make_bandit_context(test_df, p_correct=test_df["p_correct"].to_numpy(), confidence=confidence)
    bandit = LinUCB(n_actions=3, d=context.shape[1], alpha=alpha, reg=1.0)
    action_names = ["AI", "HUMAN", "DIAGNOSTIC"]

    order = np.argsort(test_df["sequence_idx"].to_numpy())
    penalties = {0: 0.0, 1: 0.0}
    logs = []
    guardrail_rows = []

    for t, idx in enumerate(order):
        g = int(test_df.iloc[idx]["gender"])
        x = context[idx]
        action_idx = bandit.select(x, penalties[g])
        action_name = action_names[action_idx]
        pred_arr, loss_arr, total_arr = _action_arrays(action_name, outcomes)
        reward = -float(total_arr[idx])
        bandit.update(action_idx, x, reward)

        logs.append(
            {
                "action": action_name,
                "y_true": int(y_true[idx]),
                "y_pred": int(pred_arr[idx]),
                "gender": g,
                "loss": float(loss_arr[idx]),
                "total_cost": float(total_arr[idx]),
                "confidence": float(confidence[idx]),
                "step": int(t + 1),
                "ai_penalty_gender0": penalties[0],
                "ai_penalty_gender1": penalties[1],
            }
        )

        if (t + 1) % guardrail_batch == 0:
            partial = pd.DataFrame(logs)
            fairness = compute_fairness_rates(
                partial["y_true"].to_numpy(),
                partial["y_pred"].to_numpy(),
                partial["gender"].to_numpy(),
            )
            delta = fairness["delta_eo_fnr"]
            disadvantaged_gender = 0 if fairness["fnr_0"] > fairness["fnr_1"] else 1
            if delta > 0.05:
                penalties[disadvantaged_gender] = min(2.5, penalties[disadvantaged_gender] + 0.05)

            guardrail_rows.append(
                {
                    "step": int(t + 1),
                    "fnr_male": fairness["fnr_0"],
                    "fnr_female": fairness["fnr_1"],
                    "delta_eo_fnr": delta,
                    "disadvantaged_gender": disadvantaged_gender,
                    "penalty_gender0": penalties[0],
                    "penalty_gender1": penalties[1],
                }
            )

    return pd.DataFrame(logs), pd.DataFrame(guardrail_rows)


def selective_risk_curve(
    confidence: np.ndarray,
    ai_loss: np.ndarray,
    fallback_loss: np.ndarray,
    coverages: np.ndarray,
) -> pd.DataFrame:
    n = len(confidence)
    order = np.argsort(-confidence)
    rows = []
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        mask_ai = np.zeros(n, dtype=bool)
        mask_ai[order[:k]] = True
        mixed_loss = np.where(mask_ai, ai_loss, fallback_loss)
        rows.append({"coverage": float(mask_ai.mean()), "risk": float(mixed_loss.mean())})
    return pd.DataFrame(rows)


def plot_risk_coverage(
    baseline_curve: pd.DataFrame,
    proposed_curve: pd.DataFrame,
    baseline_point: Tuple[float, float],
    proposed_point: Tuple[float, float],
    save_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(baseline_curve["coverage"], baseline_curve["risk"], label="Baseline curve", lw=2)
    plt.plot(proposed_curve["coverage"], proposed_curve["risk"], label="Proposed curve", lw=2)
    plt.scatter([baseline_point[0]], [baseline_point[1]], label="Baseline operating point", marker="o", s=60)
    plt.scatter([proposed_point[0]], [proposed_point[1]], label="Proposed operating point", marker="x", s=70)
    plt.xlabel("Coverage (AI autonomous rate)")
    plt.ylabel("Risk (average loss)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_shap_artifacts(
    rf_model: RandomForestClassifier,
    X_test: pd.DataFrame,
    proposed_logs: pd.DataFrame,
    save_dir: Path,
) -> None:
    if not SHAP_AVAILABLE:
        logging.warning("SHAP not available; skipping transparency artifacts.")
        return

    logging.info("Generating SHAP artifacts...")
    sample_n = min(1500, len(X_test))
    X_sample = X_test.sample(n=sample_n, random_state=SEED)
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values
        if sv.ndim == 3:
            sv = sv[:, :, 1]

    plt.figure(figsize=(8, 5))
    shap.summary_plot(sv, X_sample, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(save_dir / "shap_global_importance.png", dpi=200)
    plt.close()

    escalated_idx = proposed_logs.index[proposed_logs["action"] == "HUMAN"].tolist()
    if escalated_idx:
        row_idx = int(escalated_idx[0]) % len(X_sample)
    else:
        row_idx = 0

    vals = sv[row_idx]
    feat_names = np.array(X_sample.columns)
    top_idx = np.argsort(np.abs(vals))[-12:]
    top_names = feat_names[top_idx]
    top_vals = vals[top_idx]

    plt.figure(figsize=(8, 5))
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in top_vals]
    plt.barh(range(len(top_names)), top_vals, color=colors)
    plt.yticks(range(len(top_names)), top_names)
    plt.axvline(0, color="black", lw=1)
    plt.title("Local explanation for one escalated case")
    plt.xlabel("SHAP value (impact on prediction)")
    plt.tight_layout()
    plt.savefig(save_dir / "shap_local_case.png", dpi=200)
    plt.close()


def build_report_payload(
    baseline_summary: Dict[str, float],
    proposed_summary: Dict[str, float],
    model_metrics: Dict[str, float],
) -> Dict[str, object]:
    cost_drop = baseline_summary["average_total_cost"] - proposed_summary["average_total_cost"]
    cost_drop_pct = 100.0 * cost_drop / max(1e-9, baseline_summary["average_total_cost"])
    eo_gap_drop = baseline_summary["delta_eo_fnr"] - proposed_summary["delta_eo_fnr"]

    payload = {
        "baseline": baseline_summary,
        "proposed": proposed_summary,
        "model": model_metrics,
        "headline": {
            "cost_drop_abs": cost_drop,
            "cost_drop_pct": cost_drop_pct,
            "eo_gap_drop": eo_gap_drop,
        },
    }
    return payload


def write_text_analysis(payload: Dict[str, object]) -> None:
    baseline = payload["baseline"]
    proposed = payload["proposed"]
    headline = payload["headline"]
    lines = [
        "# Experimental Analysis",
        "",
        f"- Proposed policy average total cost: {proposed['average_total_cost']:.4f}",
        f"- Baseline average total cost: {baseline['average_total_cost']:.4f}",
        f"- Cost reduction: {headline['cost_drop_abs']:.4f} ({headline['cost_drop_pct']:.2f}%)",
        f"- Proposed coverage: {proposed['coverage_ai']:.3f}, escalation: {proposed['escalation_rate']:.3f}, diagnostic: {proposed['diagnostic_rate']:.3f}",
        f"- Baseline coverage: {baseline['coverage_ai']:.3f}, escalation: {baseline['escalation_rate']:.3f}",
        f"- Proposed Delta EO (FNR): {proposed['delta_eo_fnr']:.4f}",
        f"- Baseline Delta EO (FNR): {baseline['delta_eo_fnr']:.4f}",
        "",
        "Interpretation:",
        "The adaptive routing policy learns when to allocate human effort under cost constraints while",
        "the fairness guardrail explicitly penalizes AI autonomy for the disadvantaged group when FNR gaps widen.",
        "This often trades some coverage for better subgroup parity and lower deployment risk.",
    ]
    (RESULTS_DIR / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def copy_neurips_assets() -> None:
    style_src = FORMATTING_DIR / "neurips_2026.sty"
    checklist_src = FORMATTING_DIR / "checklist.tex"
    style_dst = PAPER_DIR / "neurips_2026.sty"
    checklist_dst = PAPER_DIR / "checklist.tex"
    if style_src.exists():
        shutil.copy2(style_src, style_dst)
    if checklist_src.exists():
        shutil.copy2(checklist_src, checklist_dst)


def run_pipeline(max_interactions: int, max_raw_rows: int | None) -> None:
    configure_logging()
    ensure_dirs()
    logging.info("Starting pipeline...")

    raw_df = load_raw_data(max_rows=max_raw_rows)
    df = preprocess_data(raw_df, max_interactions=max_interactions)
    X, y = make_feature_matrix(df)

    strata = build_strata(df)
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.3,
        random_state=SEED,
        stratify=strata,
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    test_df = df.iloc[test_idx].reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    logging.info("Training calibrated KT model...")
    base_estimator = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=SEED,
        n_jobs=-1,
    )
    calibrated = CalibratedClassifierCV(
        estimator=base_estimator,
        cv=5,
        method="sigmoid",
    )
    calibrated.fit(X_train, y_train)

    # Separate RF for transparent SHAP explanations.
    rf_for_shap = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=SEED,
        n_jobs=-1,
    )
    rf_for_shap.fit(X_train, y_train)

    p_train = calibrated.predict_proba(X_train)[:, 1]
    p_test = calibrated.predict_proba(X_test)[:, 1]
    conf_train = np.maximum(p_train, 1.0 - p_train)
    conf_test = np.maximum(p_test, 1.0 - p_test)

    tau = float(np.quantile(conf_train, 0.20))  # ~80% coverage
    model_metrics = {
        "auroc": float(roc_auc_score(y_test, p_test)),
        "brier": float(brier_score_loss(y_test, p_test)),
        "confidence_threshold_tau": tau,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    outcomes = simulate_action_outcomes(
        y_true=y_test,
        p_correct=p_test,
        gender=test_df["gender"].to_numpy(dtype=np.int64),
        skill_id=test_df["skill_id"].to_numpy(),
    )

    baseline_logs = run_baseline_policy(
        confidence=conf_test,
        threshold=tau,
        y_true=y_test,
        gender=test_df["gender"].to_numpy(dtype=np.int64),
        outcomes=outcomes,
    )

    test_df = test_df.copy()
    test_df["p_correct"] = p_test
    proposed_logs, guardrail = run_proposed_policy(
        test_df=test_df,
        y_true=y_test,
        confidence=conf_test,
        outcomes=outcomes,
        alpha=0.7,
        guardrail_batch=500,
    )

    baseline_summary = summarize_policy(baseline_logs)
    proposed_summary = summarize_policy(proposed_logs)

    coverages = np.linspace(0.10, 1.00, 19)
    baseline_curve = selective_risk_curve(conf_test, outcomes.ai_loss, outcomes.human_loss, coverages)
    proposed_curve = selective_risk_curve(
        conf_test,
        outcomes.ai_loss,
        np.minimum(outcomes.human_loss, outcomes.diagnostic_loss),
        coverages,
    )

    plot_risk_coverage(
        baseline_curve=baseline_curve,
        proposed_curve=proposed_curve,
        baseline_point=(baseline_summary["coverage_ai"], baseline_summary["average_loss"]),
        proposed_point=(proposed_summary["coverage_ai"], proposed_summary["average_loss"]),
        save_path=RESULTS_DIR / "risk_coverage_overall.png",
        title="Risk-Coverage Curves (Overall)",
    )

    male_mask = test_df["gender"].to_numpy() == 0
    female_mask = test_df["gender"].to_numpy() == 1

    male_baseline_curve = selective_risk_curve(
        conf_test[male_mask],
        outcomes.ai_loss[male_mask],
        outcomes.human_loss[male_mask],
        coverages,
    )
    male_proposed_curve = selective_risk_curve(
        conf_test[male_mask],
        outcomes.ai_loss[male_mask],
        np.minimum(outcomes.human_loss[male_mask], outcomes.diagnostic_loss[male_mask]),
        coverages,
    )
    female_baseline_curve = selective_risk_curve(
        conf_test[female_mask],
        outcomes.ai_loss[female_mask],
        outcomes.human_loss[female_mask],
        coverages,
    )
    female_proposed_curve = selective_risk_curve(
        conf_test[female_mask],
        outcomes.ai_loss[female_mask],
        np.minimum(outcomes.human_loss[female_mask], outcomes.diagnostic_loss[female_mask]),
        coverages,
    )

    plot_risk_coverage(
        baseline_curve=male_baseline_curve,
        proposed_curve=male_proposed_curve,
        baseline_point=(
            float((baseline_logs.loc[male_mask, "action"] == "AI").mean()),
            float(baseline_logs.loc[male_mask, "loss"].mean()),
        ),
        proposed_point=(
            float((proposed_logs.loc[male_mask, "action"] == "AI").mean()),
            float(proposed_logs.loc[male_mask, "loss"].mean()),
        ),
        save_path=RESULTS_DIR / "risk_coverage_male.png",
        title="Risk-Coverage Curves (Male students)",
    )

    plot_risk_coverage(
        baseline_curve=female_baseline_curve,
        proposed_curve=female_proposed_curve,
        baseline_point=(
            float((baseline_logs.loc[female_mask, "action"] == "AI").mean()),
            float(baseline_logs.loc[female_mask, "loss"].mean()),
        ),
        proposed_point=(
            float((proposed_logs.loc[female_mask, "action"] == "AI").mean()),
            float(proposed_logs.loc[female_mask, "loss"].mean()),
        ),
        save_path=RESULTS_DIR / "risk_coverage_female.png",
        title="Risk-Coverage Curves (Female students)",
    )

    # Transparency artifacts.
    save_shap_artifacts(rf_for_shap, X_test, proposed_logs, RESULTS_DIR)

    baseline_logs.to_csv(RESULTS_DIR / "baseline_policy_logs.csv", index=False)
    proposed_logs.to_csv(RESULTS_DIR / "proposed_policy_logs.csv", index=False)
    guardrail.to_csv(RESULTS_DIR / "guardrail_history.csv", index=False)

    summary_table = pd.DataFrame(
        [
            {"policy": "baseline", **baseline_summary},
            {"policy": "proposed", **proposed_summary},
        ]
    )
    summary_table.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)

    fairness_table = pd.DataFrame(
        [
            {
                "policy": "baseline",
                "delta_eo_fnr": baseline_summary["delta_eo_fnr"],
                "equalized_odds_diff": baseline_summary["equalized_odds_diff"],
                "demographic_parity_escalation_diff": baseline_summary["demographic_parity_escalation_diff"],
                "risk_gap": baseline_summary["risk_gap"],
            },
            {
                "policy": "proposed",
                "delta_eo_fnr": proposed_summary["delta_eo_fnr"],
                "equalized_odds_diff": proposed_summary["equalized_odds_diff"],
                "demographic_parity_escalation_diff": proposed_summary["demographic_parity_escalation_diff"],
                "risk_gap": proposed_summary["risk_gap"],
            },
        ]
    )
    fairness_table.to_csv(RESULTS_DIR / "fairness_metrics.csv", index=False)

    with (RESULTS_DIR / "model_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(model_metrics, f, indent=2)

    payload = build_report_payload(baseline_summary, proposed_summary, model_metrics)
    with (RESULTS_DIR / "report_payload.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    write_text_analysis(payload)
    copy_neurips_assets()
    logging.info("Pipeline complete. Artifacts saved to %s", RESULTS_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full final project pipeline.")
    parser.add_argument(
        "--max-interactions",
        type=int,
        default=120000,
        help="Cap interactions after preprocessing for runtime control.",
    )
    parser.add_argument(
        "--max-raw-rows",
        type=int,
        default=None,
        help="Optional cap when first reading raw CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(max_interactions=args.max_interactions, max_raw_rows=args.max_raw_rows)
