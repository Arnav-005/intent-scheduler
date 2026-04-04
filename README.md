# Context-Aware Notification Management via Semantic LinUCB

An end-to-end Reinforcement Learning pipeline simulating a personalized notification scheduler. This project enhances standard Contextual Bandits (LinUCB) by integrating **Zero-Shot Large Language Model (LLM) embeddings** to capture the real-time "Semantic Intent" of incoming messages, resulting in superior sample efficiency and lower cumulative regret.

---

## The Problem
Traditional notification schedulers rely on rigid, tabular metadata (Time of Day, App Source) or static human-crafted rules to decide whether to send, delay, or mute an alert. This fails to capture the fluid, psychological context of the actual message content.

## The Solution
By passing notification text through a Sentence Transformer (`all-MiniLM-L6-v2`) and compressing the dense embeddings via PCA, we provide a LinUCB agent with a mathematical **"vibe check."** This allows the algorithm to pivot dynamically — learning, for example, that an "urgent deadline" email supersedes a "studying" context penalty without requiring hardcoded rules.

---
### Literature & Foundations
This architecture and methodology were inspired by standard industry implementations and recent research in sequential decision-making:

*   **Contextual Bandits in Production:** Influenced by architectures detailed in *Fostering Responsibility in Email Marketing: A Contextual Restless Bandit Framework* (El Mimouni & Avrachenkov).
*   **Exploration vs. Exploitation:** Built on foundational frameworks from *From Ads to Interventions: Contextual Bandits in Mobile Health* (Tewari & Murphy).
*   **Semantic Feature Spaces:** Aligning with modern NLP integration techniques discussed in *Investigating the Relationship Between Physical Activity and Tailored Behavior Change Messaging: Connecting Contextual Bandit with Large Language Models* (Song et al., KDD 2025 Workshops).

---

### Updated Project Architecture
The pipeline is split into modular stages to simulate a full production environment:

1.  **Stage 1: Synthetic Data Generation (`stage1_generate_dataset.py`)** — Synthesizes a 10,000-row notification stream perfectly calibrated to a real-world 33-person survey of college students. Establishes ground-truth optimal actions (NOW, BATCH, MUTE) based on contextual modifiers and base importance scores.
2.  **Stage 2: Semantic Pipeline (`stage2_embeddings.py`)** — Implements Zero-Shot NLP using `sentence-transformers`. Applies Principal Component Analysis (PCA) to compress 384-dimensional dense vectors into 32 dimensions, solving the **Curse of Dimensionality** and allowing the Bandit to invert matrices in milliseconds.
3.  **Stage 3: The Bandit Arena (`stage3_agents.py`)** — Pits 5 distinct agents against each other in a simulated environment, including **LinUCB_Tabular** (Standard industry bandit) and **LinUCB_Semantic** (Novel contribution).
4.  **Stage 5: Evaluation & Plotting (`stage5_plots.py`)** — Generates publication-ready figures mapping **Cumulative Regret**, **Rolling Accuracy**, and **Feature Affinity Heatmaps**.

---

### Performance Overview
We migrated from strict classification accuracy to a **Risk-Aware Utility Model**. By using an asymmetric reward matrix, we penalize high-friction interruptions (e.g., disturbing a user during 'In Class' or 'Sleeping' states) more heavily than missed notifications.
| Figure | Description |
| :--- | :--- |
| ![Cumulative Regret](<Stage 5 Paper Figures/results/fig1_cumulative_regret.png>) | **Cumulative Regret:** Convergence toward the optimal utility policy. |
| ![Rolling Utility](<Stage 5 Paper Figures/results/fig2_rolling_accuracy.png>) | **Rolling Average Reward:** Utility gains via semantic cues. |
| ![Final Utility](<Stage 5 Paper Figures/results/fig3_final_accuracy.png>) | **Final Utility Comparison:** LinUCB-Semantic vs Baselines. |

### What the Bandit Learned
![Feature Affinity Heatmap](<Stage 5 Paper Figures/results/fig4_weight_heatmap.png>)

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
