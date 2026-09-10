# Golden Evaluation Set (200 Hand-Verified Samples)

This directory contains the curated golden benchmark dataset and human calibration review dataset for evaluating the Apple Support AI Agent.

---

## Dataset Files

1. **[`golden_200.csv`](file:///./golden_200.csv)**: 200 distinct, hand-verified genuine customer ↔ `@AppleSupport` conversation threads extracted from Kaggle's *Customer Support on Twitter* (`twcs.csv`).
2. **[`manual_review_30.csv`](file:///./manual_review_30.csv)**: 30 representative samples scored across 4 rubric dimensions by human domain review for calibrating the LLM-as-a-Judge.

---

## Sampling & Labelling Methodology Note

### 1. Inbound & Outbound Thread Reconstruction
- Parsed the raw 2.81M-tweet dataset (`twcs.csv`), extracting customer inbound tweets where `@AppleSupport` authored the official response.
- Cleaned and normalized text (removed URL slugs, handle mentions, unescaped HTML entities).
- Applied quality filters: minimum length ≥ 25 characters, high ASCII ratio (predominantly English).

### 2. Strict Holdout & Zero Data Contamination
- Cross-referenced against all 8,000 training samples (`data/processed/train.csv`) and 2,000 validation samples (`data/processed/validation.csv`).
- **100% of the golden dataset consists of untouched holdout threads** to guarantee zero training leakage.

### 3. Stratified Intent Distribution
- Sampled 22–23 distinct genuine customer threads across all **9 canonical intent classes**:
  1. `apple_id_login` (Account authentication, 2FA, password resets)
  2. `icloud_sync` (iCloud storage limits, Photo/iMessage synchronization)
  3. `app_store` (App downloads, installation errors, store verification)
  4. `battery_issue` (Battery drain, maximum capacity health, overheating)
  5. `charging_issue` (Lightning/USB-C cables, charging ports, MagSafe)
  6. `software_update` (iOS update failures, recovery loops, system glitches)
  7. `device_setup` (Quick Start data migration, AirPods/Bluetooth pairing)
  8. `billing_refund` (Subscription cancellations, unauthorized charges, refunds)
  9. `account_security` (Compromised accounts, phishing alerts, Lost Mode, theft)

### 4. Ground-Truth Escalation Labelling
- Annotated ground-truth escalation routing (`AUTO` vs `ESCALATE`) along with specific human-readable reasons:
  - **Always Escalate**: Critical intents (`account_security`, `billing_refund`) and high-risk keywords (`lost`, `stolen`, `fraud`, `unauthorized`, `hacked`, `breach`, `scam`).
  - **Auto-Handle**: Standard technical troubleshooting queries with established self-service resolution paths.

### 5. Human Review Calibration Subset (30 Samples)
- Selected 30 representative threads and scored them across 4 rubric dimensions (1–5 scale):
  - `Correctness` (Technical accuracy of steps)
  - `Helpfulness` (Actionable guidance and links)
  - `Brand Tone` (Empathy, conciseness ≤ 70 words, calm voice)
  - `Groundedness` (Factual fidelity to Apple Support standards)
- Used to benchmark the LLM-as-a-Judge, demonstrating **100.0% agreement within ±1 point** and a **Mean Absolute Error (MAE) of 0.300**.
