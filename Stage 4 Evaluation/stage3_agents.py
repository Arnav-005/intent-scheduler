"""
Stage 3 — Bandit Agents
=======================
Implements all 5 agents and runs the simulation loop over the 10,000-row
notification stream produced by Stages 1 & 2.

Agents
------
1. ShowAll      — dumb baseline: always picks NOW
2. MuteAll      — dumb baseline: always picks MUTE
3. RuleBased    — hand-crafted if/else from survey majority rules
4. LinUCB_Tab   — LinUCB with one-hot tabular context (app + sender + time + ctx)
5. LinUCB_Sem   — LinUCB with tabular ⊕ 32-dim PCA semantic context

Outputs
-------
• results/simulation_log.csv   — per-step decisions & rewards for every agent
• results/agent_summary.csv    — final accuracy + total regret per agent
"""

import numpy as np
import pandas as pd
import os, time

# ── Config ────────────────────────────────────────────────────────────────────
CSV_IN      = "notifications_10k_with_embeddings.csv"
RESULTS_DIR = "results"
ALPHA       = 0.3          # LinUCB exploration parameter (tune if needed)
SEED        = 42
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(RESULTS_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(CSV_IN)
N  = len(df)
print(f"  {N} rows loaded\n")

ARMS      = ["NOW", "MUTE", "BATCH"]
ARM_INDEX = {a: i for i, a in enumerate(ARMS)}
sem_cols  = [f"sem_{i:02d}" for i in range(32)]

# ── 2. Feature engineering ────────────────────────────────────────────────────
print("Building feature matrices...")

CAT_COLS = {
    "app_source":   sorted(df["app_source"].unique()),
    "sender_type":  sorted(df["sender_type"].unique()),
    "time_slot":    sorted(df["time_slot"].unique()),
    "user_context": sorted(df["user_context"].unique()),
}

def one_hot(df: pd.DataFrame, col_maps: dict) -> np.ndarray:
    """Stack one-hot vectors for all categorical columns."""
    parts = []
    for col, vocab in col_maps.items():
        idx = pd.Categorical(df[col], categories=vocab).codes
        oh  = np.eye(len(vocab), dtype=np.float32)[idx]
        parts.append(oh)
    return np.hstack(parts)   # shape (N, D_tab)

X_tab = one_hot(df, CAT_COLS)
X_sem = df[sem_cols].values.astype(np.float32)
X_sem_norm = X_sem / (np.linalg.norm(X_sem, axis=1, keepdims=True) + 1e-9)

# Tabular + semantic concatenation  (x_normal ⊕ x_semantic)
X_full = np.hstack([X_tab, X_sem_norm])   # shape (N, D_tab + 32)

D_tab  = X_tab.shape[1]
D_full = X_full.shape[1]
print(f"  Tabular feature dim : {D_tab}")
print(f"  Full feature dim    : {D_full}  (tabular + 32 semantic)")

true_actions = df["true_optimal_action"].values   # str labels

# ── 3. Agent definitions ──────────────────────────────────────────────────────

class ShowAllAgent:
    """Dumb baseline — always show immediately."""
    name = "ShowAll"
    def choose(self, *args): return "NOW"
    def update(self, *args): pass


class MuteAllAgent:
    """Dumb baseline — always mute."""
    name = "MuteAll"
    def choose(self, *args): return "MUTE"
    def update(self, *args): pass


class RuleBasedAgent:
    """
    Hand-crafted rules derived directly from the survey majority answers.

    Survey-grounded rules (majority preference shown):
      • Studying + social_media / instagram / unknown → MUTE  (57.6 %)
      • In class                                      → MUTE  (majority)
      • Late night / sleeping                         → MUTE  (78.8 %)
      • Spam / phishing / promo content               → MUTE
      • Deadline reminder / assignment                → NOW   (78.8 % show for urgent email)
      • WhatsApp from known friend while with family  → BATCH (63.6 %)
      • Unknown sender course email during study      → MUTE  (51.5 %)
      • Waking up                                     → BATCH (batch until ready)
      • Everything else                               → BATCH (default safe choice)
    """
    name = "RuleBased"

    MUTE_CONTENT  = {"spam", "phishing", "promo_offer", "discount_offer",
                     "newsletter", "app_update"}
    MUTE_SENDERS  = {"spam", "promotional", "unknown_sender"}
    URGENT_CONTENT = {"deadline_reminder", "deadline", "assignment",
                      "grade_update", "grade", "registration", "system_alert"}

    def choose(self, row: dict, *args) -> str:
        ctx     = row["user_context"]
        slot    = row["time_slot"]
        app     = row["app_source"]
        sender  = row["sender_type"]
        content = row["content_type"]

        # ── Hard mutes ─────────────────────────────────────────────────────
        if content in self.MUTE_CONTENT:
            return "MUTE"
        if sender in self.MUTE_SENDERS:
            return "MUTE"
        if ctx in {"sleeping", "in_class"}:
            return "MUTE"
        if slot == "late_night" and content not in self.URGENT_CONTENT:
            return "MUTE"
        if ctx == "studying" and app in {"instagram", "news_promo", "unknown"}:
            return "MUTE"
        if ctx == "studying" and sender in {"unknown", "unknown_sender",
                                            "rare_contact", "promotional"}:
            return "MUTE"

        # ── Urgent escalate ────────────────────────────────────────────────
        if content in self.URGENT_CONTENT:
            return "NOW"
        if sender == "professor":
            return "NOW"
        if app == "academic" and content in {"announcement", "meeting_reminder"}:
            return "NOW"

        # ── With family — WhatsApp from friend → batch ─────────────────────
        if ctx == "with_family" and app == "whatsapp":
            return "BATCH"

        # ── Morning / waking up — batch non-urgents ────────────────────────
        if ctx == "waking_up" or slot == "morning":
            return "BATCH"

        # ── Social media while relaxing or with friends → batch ────────────
        if app in {"instagram", "news_promo"} and ctx in {"relaxing", "with_friends"}:
            return "BATCH"

        # ── Default ────────────────────────────────────────────────────────
        return "BATCH"

    def update(self, *args):
        pass


class LinUCBAgent:
    """
    Disjoint LinUCB (one independent (A, b) pair per arm).

    Parameters
    ----------
    name  : display name
    dim   : context dimensionality
    alpha : exploration / exploitation trade-off
    """
    def __init__(self, name: str, dim: int, alpha: float = ALPHA):
        self.name  = name
        self.dim   = dim
        self.alpha = alpha
        self.A  = {a: np.eye(dim, dtype=np.float64)   for a in ARMS}
        self.b  = {a: np.zeros(dim, dtype=np.float64) for a in ARMS}
        self._A_inv = {a: np.eye(dim, dtype=np.float64) for a in ARMS}  # cached

    def _ucb(self, x: np.ndarray, arm: str) -> float:
        A_inv = self._A_inv[arm]
        theta = A_inv @ self.b[arm]
        return float(theta @ x + self.alpha * np.sqrt(x @ A_inv @ x))

    def choose(self, x: np.ndarray, *args) -> str:
        scores = {a: self._ucb(x, a) for a in ARMS}
        return max(scores, key=scores.__getitem__)

    def update(self, x: np.ndarray, arm: str, reward: float):
        self.A[arm]    += np.outer(x, x)
        self.b[arm]    += reward * x
        # Sherman-Morrison rank-1 update avoids full re-inversion each step
        A_inv = self._A_inv[arm]
        Ax    = A_inv @ x
        denom = 1.0 + x @ Ax
        self._A_inv[arm] = A_inv - np.outer(Ax, Ax) / denom


# ── 4. Instantiate agents ─────────────────────────────────────────────────────
agents = [
    ShowAllAgent(),
    MuteAllAgent(),
    RuleBasedAgent(),
    LinUCBAgent("LinUCB_Tab",  dim=D_tab),
    LinUCBAgent("LinUCB_Sem",  dim=D_full),
]
print(f"\nAgents ready: {[a.name for a in agents]}\n")

# ── 5. Simulation loop ────────────────────────────────────────────────────────
print("Running simulation...")
t0 = time.time()

# Per-agent tracking
cumulative_reward  = {a.name: 0 for a in agents}
cumulative_regret  = {a.name: 0 for a in agents}
reward_history     = {a.name: [] for a in agents}
regret_history     = {a.name: [] for a in agents}
choice_history     = {a.name: [] for a in agents}

for t in range(N):
    row         = df.iloc[t]
    true_action = true_actions[t]
    row_dict    = row.to_dict()

    x_tab  = X_tab[t].astype(np.float64)
    x_full = X_full[t].astype(np.float64)

    for agent in agents:
        # Select context based on agent type
        if isinstance(agent, LinUCBAgent):
            x = x_full if "Sem" in agent.name else x_tab
            chosen = agent.choose(x)
        else:
            chosen = agent.choose(row_dict)

        reward = 1 if chosen == true_action else 0
        regret = 0 if chosen == true_action else 1

        cumulative_reward[agent.name] += reward
        cumulative_regret[agent.name] += regret
        reward_history[agent.name].append(cumulative_reward[agent.name])
        regret_history[agent.name].append(cumulative_regret[agent.name])
        choice_history[agent.name].append(chosen)

        # Update bandit state
        if isinstance(agent, LinUCBAgent):
            agent.update(x, chosen, reward)

    if (t + 1) % 1000 == 0:
        elapsed = time.time() - t0
        print(f"  Step {t+1:>5} / {N}  ({elapsed:.1f}s) — "
              f"LinUCB_Sem acc: {reward_history['LinUCB_Sem'][-1]/(t+1):.3f}")

print(f"\nSimulation complete in {time.time()-t0:.1f}s\n")

# ── 6. Save per-step log ──────────────────────────────────────────────────────
print("Saving simulation log...")
log_data = {"step": np.arange(1, N + 1), "true_action": true_actions}
for agent in agents:
    log_data[f"{agent.name}_choice"]  = choice_history[agent.name]
    log_data[f"{agent.name}_cum_rew"] = reward_history[agent.name]
    log_data[f"{agent.name}_cum_reg"] = regret_history[agent.name]

log_df = pd.DataFrame(log_data)
log_df.to_csv(f"{RESULTS_DIR}/simulation_log.csv", index=False)
print(f"  ✓ {RESULTS_DIR}/simulation_log.csv  ({log_df.shape})")

# ── 7. Summary table ──────────────────────────────────────────────────────────
print("\n── Final Results ────────────────────────────────────────────────────────")
summary_rows = []
for agent in agents:
    final_reward = cumulative_reward[agent.name]
    final_regret = cumulative_regret[agent.name]
    accuracy     = final_reward / N
    summary_rows.append({
        "agent":          agent.name,
        "total_correct":  final_reward,
        "total_regret":   final_regret,
        "accuracy_%":     round(accuracy * 100, 2),
    })
    print(f"  {agent.name:<15}  acc={accuracy:.3f}   regret={final_regret}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{RESULTS_DIR}/agent_summary.csv", index=False)
print(f"\n  ✓ {RESULTS_DIR}/agent_summary.csv")

# ── 8. LinUCB weight inspection (for paper) ───────────────────────────────────
print("\n── LinUCB_Sem Top Feature Weights ───────────────────────────────────────")
sem_agent = next(a for a in agents if a.name == "LinUCB_Sem")
for arm in ARMS:
    A_inv = sem_agent._A_inv[arm]
    theta = A_inv @ sem_agent.b[arm]

    # Semantic dims are the last 32 entries
    sem_theta = theta[-32:]
    tab_theta = theta[:D_tab]

    print(f"\n  ARM: {arm}")
    print(f"    Semantic weight  — max={sem_theta.max():.4f}  min={sem_theta.min():.4f}  "
          f"mean={sem_theta.mean():.4f}")
    print(f"    Tabular weight   — max={tab_theta.max():.4f}  min={tab_theta.min():.4f}  "
          f"mean={tab_theta.mean():.4f}")

    # Top 3 tabular features driving this arm
    feature_names = []
    for col, vocab in CAT_COLS.items():
        for v in vocab:
            feature_names.append(f"{col}={v}")
    top_idx  = np.argsort(tab_theta)[-3:][::-1]
    bot_idx  = np.argsort(tab_theta)[:3]
    print(f"    Top-3 promoting : {[feature_names[i] for i in top_idx]}")
    print(f"    Top-3 inhibiting: {[feature_names[i] for i in bot_idx]}")

print("\n── All done. Outputs ready for Stage 4 (plotting) ──────────────────────")
print(f"  {RESULTS_DIR}/simulation_log.csv")
print(f"  {RESULTS_DIR}/agent_summary.csv")
