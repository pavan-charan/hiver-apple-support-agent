"""Evaluation suite measuring Intent Classification, Escalation Routing, and Reply Quality."""
import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    CONFUSION_MATRIX_PATH,
    GOLDEN_200_CSV_PATH,
    INTENT_CLASSES,
    METRICS_JSON_PATH,
    PREDICTIONS_CSV_PATH,
    RESULTS_DIR,
    ensure_directories_exist,
)
from src.inference import AppleSupportAgent
from src.judge import LLMJudge
from src.utils import logger, save_json, timed_execution


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list,
    output_path: Path = CONFUSION_MATRIX_PATH,
) -> None:
    """Plot and save a publication-quality normalized confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    # Row normalize to get true recall per class (handle division by zero if empty)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Class Recall Rate (Normalized)", rotation=270, labelpad=15, fontsize=11)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        title="Intent Classification Confusion Matrix (Normalized Recall & Counts)",
        ylabel="True Intent",
        xlabel="Predicted Intent",
    )

    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor", fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)

    # Annotate with recall % and raw count
    for i in range(len(classes)):
        for j in range(len(classes)):
            val_pct = cm_norm[i, j]
            val_cnt = cm[i, j]
            txt = f"{val_pct * 100:.1f}%\n({val_cnt})" if val_cnt > 0 else "0%"
            text_color = "white" if val_pct > 0.45 else "black"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.5,
                fontweight="bold" if i == j else "normal",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved normalized confusion matrix plot to {output_path}")


@timed_execution
def evaluate_agent(
    dataset_path: Path = GOLDEN_200_CSV_PATH,
    metrics_path: Path = METRICS_JSON_PATH,
    predictions_path: Path = PREDICTIONS_CSV_PATH,
    run_judge: bool = False,
    agent: Optional[AppleSupportAgent] = None,
) -> Dict[str, Any]:
    """Evaluate AppleSupportAgent across intent, escalation, and reply quality on any dataset."""
    ensure_directories_exist()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {dataset_path}")

    logger.info(f"Loading evaluation dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    logger.info(f"Dataset contains {len(df):,} instances for evaluation.")

    # Initialize agent if not provided
    if agent is None:
        agent = AppleSupportAgent()

    # 1. Run batch inference through AppleSupportAgent
    logger.info("Executing inference through AppleSupportAgent...")
    predictions = agent.predict_batch(df["tweet"].tolist())
    pred_df = pd.DataFrame(predictions)

    # Attach ground truth
    pred_df["ground_truth_intent"] = df["intent"]
    pred_df["ground_truth_escalate"] = df.get("escalate", "AUTO")
    pred_df["gold_reply"] = df.get("gold_reply", "")

    # Export predictions CSV
    output_cols = [
        "tweet",
        "intent",
        "confidence",
        "generated_reply",
        "decision",
        "reason",
        "ground_truth_intent",
        "ground_truth_escalate",
    ]
    pred_df[output_cols].to_csv(predictions_path, index=False)
    logger.info(f"Saved pipeline predictions to {predictions_path}")

    # 2. Evaluate Intent Classification
    y_true_intent = pred_df["ground_truth_intent"].tolist()
    y_pred_intent = pred_df["intent"].tolist()

    intent_acc = accuracy_score(y_true_intent, y_pred_intent)
    intent_prec_macro = precision_score(y_true_intent, y_pred_intent, average="macro", zero_division=0)
    intent_rec_macro = recall_score(y_true_intent, y_pred_intent, average="macro", zero_division=0)
    intent_f1_macro = f1_score(y_true_intent, y_pred_intent, average="macro", zero_division=0)

    # Plot confusion matrix
    present_classes = sorted(list(set(y_true_intent + y_pred_intent)))
    plot_confusion_matrix(y_true_intent, y_pred_intent, present_classes)

    # 3. Evaluate Escalation Routing
    y_true_esc = (pred_df["ground_truth_escalate"] == "ESCALATE").astype(int)
    y_pred_esc = (pred_df["decision"] == "ESCALATE").astype(int)

    esc_acc = accuracy_score(y_true_esc, y_pred_esc)
    esc_prec = precision_score(y_true_esc, y_pred_esc, zero_division=0)
    esc_rec = recall_score(y_true_esc, y_pred_esc, zero_division=0)
    esc_f1 = f1_score(y_true_esc, y_pred_esc, zero_division=0)

    try:
        from src.config import PROJECT_ROOT
        clean_dataset_name = str(Path(dataset_path).resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except Exception:
        clean_dataset_name = Path(dataset_path).as_posix()

    metrics: Dict[str, Any] = {
        "dataset": clean_dataset_name,
        "intent_classification": {
            "accuracy": round(float(intent_acc), 4),
            "precision_macro": round(float(intent_prec_macro), 4),
            "recall_macro": round(float(intent_rec_macro), 4),
            "macro_f1": round(float(intent_f1_macro), 4),
            "eval_samples": len(df),
        },
        "escalation_decision": {
            "accuracy": round(float(esc_acc), 4),
            "precision": round(float(esc_prec), 4),
            "recall": round(float(esc_rec), 4),
            "f1_score": round(float(esc_f1), 4),
            "eval_samples": len(df),
        },
    }

    # 4. Optional Reply Quality & LLM Judge
    if run_judge:
        logger.info("Running LLM-as-a-judge evaluation...")
        judge = LLMJudge()
        judge_metrics = judge.evaluate_dataset(pred_df)
        human_agreement = judge.compute_human_agreement()
        metrics["reply_quality_judge"] = {
            **judge_metrics,
            "human_agreement": human_agreement,
        }
        logger.info(f"LLM Judge Composite Score: {judge_metrics['mean_composite_score']}/5.0")
    else:
        metrics["reply_quality_judge"] = {
            "status": "Skipped. Run with `--judge` to trigger LLM-as-a-judge.",
        }

    # Save metrics JSON
    save_json(metrics, metrics_path)

    logger.info(f"=== Evaluation Summary ({dataset_path.name}) ===")
    logger.info(f"Intent Accuracy:  {intent_acc:.4f} | Macro F1: {intent_f1_macro:.4f}")
    logger.info(f"Escalation F1:    {esc_f1:.4f} | Precision: {esc_prec:.4f} | Recall: {esc_rec:.4f}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Apple Support AI Agent on Dataset.")
    parser.add_argument("--dataset-path", type=str, default=str(GOLDEN_200_CSV_PATH), help="Path to input dataset CSV")
    parser.add_argument("--output-metrics", type=str, default=str(METRICS_JSON_PATH), help="Path to output metrics JSON")
    parser.add_argument("--output-predictions", type=str, default=str(PREDICTIONS_CSV_PATH), help="Path to output predictions CSV")
    parser.add_argument("--judge", action="store_true", help="Run LLM-as-a-judge reply evaluation")
    args = parser.parse_args()

    evaluate_agent(
        dataset_path=Path(args.dataset_path),
        metrics_path=Path(args.output_metrics),
        predictions_path=Path(args.output_predictions),
        run_judge=args.judge,
    )
