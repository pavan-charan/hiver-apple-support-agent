"""Baseline Models Implementation and Comparative Evaluation.

Implements and evaluates:
1. Baseline 1 (Trivial): Majority class intent predictor & zero-escalation rule.
2. Baseline 2 (Simple): TF-IDF n-gram vectorizer (1-2 grams) + Linear SVM intent classifier.
3. Compares against the primary Apple Support AI Agent pipeline.
"""
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.config import (
    TRAIN_CSV_PATH,
    GOLDEN_200_CSV_PATH,
    TEST_CSV_PATH,
    RANDOM_SEED,
)
from src.inference import AppleSupportAgent
from src.utils import logger, seed_everything


def evaluate_trivial_baseline(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate Baseline 1: Majority class predictor."""
    dummy_intent = DummyClassifier(strategy="most_frequent")
    dummy_intent.fit(train_df["tweet"], train_df["intent"])
    
    t0 = time.perf_counter()
    pred_intents = dummy_intent.predict(eval_df["tweet"])
    latency_ms = ((time.perf_counter() - t0) / len(eval_df)) * 1000.0

    acc = accuracy_score(eval_df["intent"], pred_intents)
    p, r, f1, _ = precision_recall_fscore_support(
        eval_df["intent"], pred_intents, average="macro", zero_division=0
    )

    y_true_esc = (eval_df["escalate"] == "ESCALATE").astype(int)
    y_pred_esc = np.zeros(len(eval_df), dtype=int)
    
    esc_p, esc_r, esc_f1, _ = precision_recall_fscore_support(
        y_true_esc, y_pred_esc, average="binary", zero_division=0
    )

    return {
        "model": "Baseline 1: Trivial (Majority Class)",
        "intent_accuracy": acc,
        "intent_macro_f1": f1,
        "escalation_recall": esc_r,
        "escalation_f1": esc_f1,
        "latency_ms": latency_ms,
    }


def evaluate_simple_baseline(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate Baseline 2: TF-IDF (1-2 grams) + Linear SVM."""
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), lowercase=True)
    X_train = tfidf.fit_transform(train_df["tweet"])
    y_train = train_df["intent"]

    svm = LinearSVC(C=1.0, random_state=RANDOM_SEED, max_iter=2000)
    svm.fit(X_train, y_train)

    t0 = time.perf_counter()
    X_eval = tfidf.transform(eval_df["tweet"])
    pred_intents = svm.predict(X_eval)
    latency_ms = ((time.perf_counter() - t0) / len(eval_df)) * 1000.0

    acc = accuracy_score(eval_df["intent"], pred_intents)
    p, r, f1, _ = precision_recall_fscore_support(
        eval_df["intent"], pred_intents, average="macro", zero_division=0
    )

    high_risk_intents = {"account_security", "billing_refund"}
    high_risk_keywords = {"hack", "stolen", "fraud", "unauthorized", "lost"}
    
    pred_esc = []
    for tweet, intent in zip(eval_df["tweet"], pred_intents):
        t_low = str(tweet).lower()
        if intent in high_risk_intents or any(kw in t_low for kw in high_risk_keywords):
            pred_esc.append("ESCALATE")
        else:
            pred_esc.append("AUTO")

    y_true_esc = (eval_df["escalate"] == "ESCALATE").astype(int)
    y_pred_esc = (pd.Series(pred_esc) == "ESCALATE").astype(int)

    esc_p, esc_r, esc_f1, _ = precision_recall_fscore_support(
        y_true_esc, y_pred_esc, average="binary", zero_division=0
    )

    return {
        "model": "Baseline 2: Simple (TF-IDF + Linear SVM)",
        "intent_accuracy": acc,
        "intent_macro_f1": f1,
        "escalation_recall": esc_r,
        "escalation_f1": esc_f1,
        "latency_ms": latency_ms,
    }


def evaluate_our_agent(eval_df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate primary production AppleSupportAgent."""
    agent = AppleSupportAgent()

    t0 = time.perf_counter()
    predictions = [agent.predict(t) for t in eval_df["tweet"]]
    latency_ms = ((time.perf_counter() - t0) / len(eval_df)) * 1000.0

    pred_intents = [p["intent"] for p in predictions]
    pred_esc = [p["decision"] for p in predictions]

    acc = accuracy_score(eval_df["intent"], pred_intents)
    p, r, f1, _ = precision_recall_fscore_support(
        eval_df["intent"], pred_intents, average="macro", zero_division=0
    )

    y_true_esc = (eval_df["escalate"] == "ESCALATE").astype(int)
    y_pred_esc = (pd.Series(pred_esc) == "ESCALATE").astype(int)

    esc_p, esc_r, esc_f1, _ = precision_recall_fscore_support(
        y_true_esc, y_pred_esc, average="binary", zero_division=0
    )

    return {
        "model": "Our System (MiniLM + LogReg + FAISS + Rules)",
        "intent_accuracy": acc,
        "intent_macro_f1": f1,
        "escalation_recall": esc_r,
        "escalation_f1": esc_f1,
        "latency_ms": latency_ms,
    }


def run_baselines_comparison(eval_path: str = str(GOLDEN_200_CSV_PATH)):
    """Run and display side-by-side comparison across all models."""
    seed_everything(RANDOM_SEED)

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    eval_df = pd.read_csv(eval_path)

    logger.info(f"Running baseline comparisons on: {eval_path} ({len(eval_df):,} samples)")

    res_trivial = evaluate_trivial_baseline(train_df, eval_df)
    res_simple = evaluate_simple_baseline(train_df, eval_df)
    res_ours = evaluate_our_agent(eval_df)

    results = [res_trivial, res_simple, res_ours]
    
    print("\n" + "=" * 95)
    print("                      BASELINE COMPARISON EVALUATION SUMMARY")
    print("=" * 95)
    print(f"{'Model / Architecture':<45} | {'Acc':<7} | {'Macro F1':<8} | {'Esc Rec':<7} | {'Esc F1':<7} | {'Latency'}")
    print("-" * 95)
    for r in results:
        print(
            f"{r['model']:<45} | "
            f"{r['intent_accuracy']*100:>5.2f}% | "
            f"{r['intent_macro_f1']:>8.4f} | "
            f"{r['escalation_recall']*100:>6.2f}% | "
            f"{r['escalation_f1']:>7.4f} | "
            f"{r['latency_ms']:>5.2f} ms/q"
        )
    print("=" * 95 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline models vs primary AI Agent.")
    parser.add_argument("--eval-dataset", type=str, default=str(GOLDEN_200_CSV_PATH), help="Path to evaluation CSV")
    args = parser.parse_args()
    run_baselines_comparison(eval_path=args.eval_dataset)
