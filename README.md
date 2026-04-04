# Context-Aware Notification Management via Semantic LinUCB

An end-to-end Reinforcement Learning pipeline simulating a personalized notification scheduler. This project enhances standard Contextual Bandits (LinUCB) by integrating **Zero-Shot Large Language Model (LLM) embeddings** to capture the real-time "Semantic Intent" of incoming messages, resulting in superior sample efficiency and lower cumulative regret.

---

## The Problem
Traditional notification schedulers rely on rigid, tabular metadata (Time of Day, App Source) or static human-crafted rules to decide whether to send, delay, or mute an alert. This fails to capture the fluid, psychological context of the actual message content.

## The Solution
By passing notification text through a Sentence Transformer (`all-MiniLM-L6-v2`) and compressing the dense embeddings via PCA, we provide a LinUCB agent with a mathematical **"vibe check."** This allows the algorithm to pivot dynamically — learning, for example, that an "urgent deadline" email supersedes a "studying" context penalty without requiring hardcoded rules.

---

## Literature & Foundations
This architecture and methodology were inspired by standard industry implementations and recent research in sequential decision-making:

* **Contextual Bandits in Production:** Influenced by architectures detailed in AWS machine learning whitepapers (2025).
* **Exploration vs. Exploitation:** Built on foundational frameworks from Tewari & Murphy.
* **Semantic Feature Spaces:** Aligning with modern NLP integration techniques discussed at the KDD 2025 Workshops.

---

## Project Architecture
The pipeline is split into modular stages to simulate a full production environment:

1.  **Stage 1: Synthetic Data Generation (`stage1_generate_dataset.py`)** — Synthesizes a 10,000-row notification stream perfectly calibrated to a real-world 33-person survey of college students. Establishes ground-truth optimal actions (NOW, BATCH, MUTE) based on contextual modifiers and base importance scores.
2.  **Stage 2: Semantic Pipeline (`stage2_embeddings.py`)** — Implements Zero-Shot NLP using `sentence-transformers`. Applies Principal Component Analysis (PCA) to compress 384-dimensional dense vectors into 32 dimensions, solving the Curse of Dimensionality and allowing the Bandit to invert matrices in milliseconds.
3.  **Stage 3: The Bandit Arena (`stage3_agents.py`)** — Pits 5 distinct agents against each other in a simulated environment: ShowAll (Baseline), MuteAll (Baseline), RuleBased (Static logic from survey majority), LinUCB_Tabular (Standard industry bandit), and LinUCB_Semantic (Novel contribution).
4.  **Stage 5: Evaluation & Plotting (`stage5_plots.py`)** — Generates publication-ready figures mapping Cumulative Regret, Rolling Accuracy, and Feature Affinity Heatmaps.

---

## Empirical Results
Over a 10,000-step simulation, the Semantic LinUCB agent definitively outperformed both human intuition and standard ML baselines:

| Agent | Final Accuracy | Total Regret |
| :--- | :--- | :--- |
| **LinUCB_Semantic (Novel)** | **79.1%** | **2,088** |
| LinUCB_Tabular (Baseline) | 76.6% | 2,338 |
| Rule-Based (Human Logic) | 60.1% | 3,985 |
| Show-All (Dumb Baseline) | 33.4% | 6,655 |

> [!TIP]
> See the `/results` directory for high-resolution Cumulative Regret curves and the LinUCB Weight Heatmap demonstrating organic feature-learning.

---

## How to Run

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn sentence-transformers matplotlib
```

### 2. Execute the Pipeline
```bash
python stage1_generate_dataset.py  # Generates the 10k CSV
python stage2_embeddings.py        # Adds the PCA semantic vectors
python stage3_agents.py            # Runs the simulation
python stage5_plots.py             # Generates the charts
```

---

## Tech Stack
* **Core ML:** NumPy, scikit-learn (PCA, StandardScaler).
* **NLP:** sentence-transformers (`all-MiniLM-L6-v2`).
* **Data & Viz:** Pandas, Matplotlib.
* **Algorithm:** Disjoint Linear Upper Confidence Bound (LinUCB) with Sherman-Morrison updates for $O(d^2)$ efficiency.
