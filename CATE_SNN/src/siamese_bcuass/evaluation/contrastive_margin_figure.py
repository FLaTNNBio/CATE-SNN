import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Workaround per un bug su Windows/Mac

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# Impostazioni grafiche per una resa professionale
matplotlib.use('TkAgg')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'CMU Serif', 'Palatino'],
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 120
})


# ==============================================================================
# 1. SIMULAZIONE DEI DATI: EFFETTO FARMACI PSICHIATRICI
# ==============================================================================

# We'll simulate the dataset using your logic and compute clear, presentation-ready summaries:
# 1) Dimensione del campione per coorte e trattamento (conteggi e percentuali)
# 2) Tasso di miglioramento (outcome) per coorte e trattamento
# 3) Statistiche descrittive per le covariate numeriche (per coorte)
# 4) Quote (rate) per le variabili binarie (per coorte e trattamento)
# 5) Riepilogo "info" del dataset

import numpy as np
import pandas as pd

def simulate_psych_data(N=200_000, seed=42):
    np.random.seed(seed)
    severity = np.clip(np.random.normal(15, 5, N), 0, 27)
    age = np.clip(np.random.normal(40, 15, N), 18, 80)
    substance_abuse_prob = 0.1 + 0.4 * (severity / 27)
    substance_abuse = np.random.binomial(1, substance_abuse_prob)
    access_to_care = np.random.binomial(1, 0.6, N)

    t_rct = np.random.binomial(1, 0.5, N)

    logit_obs = -2.5 + 4.0 * (severity / 27) + 1.5 * (age / 80) + 1.0 * access_to_care
    prob_obs = 1 / (1 + np.exp(-logit_obs))
    t_obs = np.random.binomial(1, prob_obs)

    def simulate_outcome(T, severity, age, abuse):
        logit_y = -1.0 + 1.5 * T - 2.5 * (severity / 27) + 0.5 * (age / 80) - 0.8 * abuse
        prob_y = 1 / (1 + np.exp(-logit_y))
        return np.random.binomial(1, prob_y)

    Y_rct = simulate_outcome(t_rct, severity, age, substance_abuse)
    Y_obs = simulate_outcome(t_obs, severity, age, substance_abuse)

    features = {'X1_Severity': severity, 'X2_Age': age, 'X3_Abuse': substance_abuse, 'X4_Access': access_to_care}
    df_rct = pd.DataFrame({**features, 'T': t_rct, 'Y_Improvement': Y_rct, 'cohort': 'RCT'})
    df_obs = pd.DataFrame({**features, 'T': t_obs, 'Y_Improvement': Y_obs, 'cohort': 'OBS'})

    return pd.concat([df_rct, df_obs], ignore_index=True)

# Generate data
df = simulate_psych_data()

# 1) Sample size by cohort & treatment (counts and within-cohort %)
ct_counts = (
    df.groupby(['cohort', 'T'])
      .size()
      .rename('count')
      .reset_index()
)
# Add percentages within each cohort
ct_counts['pct_within_cohort'] = (
    ct_counts.groupby('cohort')['count'].apply(lambda x: x / x.sum() * 100).values
)
# Map T to labels
ct_counts['treatment'] = ct_counts['T'].map({0: 'Control', 1: 'Treated'})
ct_counts = ct_counts[['cohort', 'treatment', 'count', 'pct_within_cohort']].sort_values(['cohort','treatment'])

# 2) Outcome rate by cohort & treatment
outcome_rate = (
    df.groupby(['cohort','T'])['Y_Improvement']
      .mean()
      .rename('improvement_rate')
      .reset_index()
)
outcome_rate['treatment'] = outcome_rate['T'].map({0: 'Control', 1: 'Treated'})
outcome_rate['improvement_rate'] = (outcome_rate['improvement_rate'] * 100).round(2)
outcome_rate = outcome_rate[['cohort','treatment','improvement_rate']].sort_values(['cohort','treatment'])

# 3) Numeric covariates summary by cohort
num_cols = ['X1_Severity','X2_Age']
num_summary = (
    df.groupby('cohort')[num_cols]
      .agg(['mean','std','min','median','max'])
      .round(2)
)
# Flatten columns for readability
num_summary.columns = ['_'.join(col).strip() for col in num_summary.columns.values]
num_summary = num_summary.reset_index()

# 4) Binary variable rates by cohort & treatment
bin_cols = ['X3_Abuse','X4_Access']
bin_rates = (
    df.groupby(['cohort','T'])[bin_cols]
      .mean()
      .mul(100)
      .round(2)
      .reset_index()
)
bin_rates['treatment'] = bin_rates['T'].map({0: 'Control', 1: 'Treated'})
bin_rates = bin_rates.drop(columns=['T']).rename(columns={'X3_Abuse':'Abuse_%','X4_Access':'Access_%'})
bin_rates = bin_rates[['cohort','treatment','Abuse_%','Access_%']].sort_values(['cohort','treatment'])

# 5) Quick dataset info
info_df = pd.DataFrame({
    'n_rows': [len(df)],
    'n_columns': [df.shape[1]],
    'columns': [', '.join(df.columns)],
    'missing_values_total': [int(df.isna().sum().sum())]
})



# Also print a short textual key takeaway
# Compute headline numbers for a simple narrative
totals = df['cohort'].value_counts().to_dict()
treated_counts = df.groupby('cohort')['T'].sum().to_dict()
control_counts = {k: totals[k] - treated_counts.get(k, 0) for k in totals.keys()}

print("=== Riepilogo veloce ===")
for coh in ['RCT','OBS']:
    if coh in totals:
        n = totals[coh]
        t = int(treated_counts.get(coh, 0))
        c = int(control_counts.get(coh, 0))
        print(f"{coh}: N={n:,} -> Trattati={t:,} ({t/n*100:.1f}%), Controllo={c:,} ({c/n*100:.1f}%)")
        # Outcome by treatment
        for label, val in [('Control', 0), ('Treated', 1)]:
            rate = df[(df['cohort']==coh) & (df['T']==val)]['Y_Improvement'].mean()*100
            print(f"  - {label}: miglioramento medio = {rate:.2f}%")



# ==============================================================================
# 2. PLOT DELLE DISTRIBUZIONI (STILE FIGURA 4)
# ==============================================================================

def plot_psych_panels(df, cohort_name):
    """Disegna i pannelli delle distribuzioni per il cohort specificato."""
    cohort_df = df[df['cohort'] == cohort_name].copy()

    # Rinomina T e Y per la legenda
    cohort_df['Treatment'] = cohort_df['T'].map({0: 'No Drug', 1: 'Drug'})

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"Distributions in the Scenario {cohort_name}", fontsize=20)  # Rimosso y=...

    # Grafici per le covariate (invariati)
    sns.kdeplot(data=cohort_df, x='X1_Severity', hue='Treatment', fill=True, ax=axes[0, 0], palette='viridis')
    axes[0, 0].set_title('X1: Depression Severity')

    sns.kdeplot(data=cohort_df, x='X2_Age', hue='Treatment', fill=True, ax=axes[0, 1], palette='viridis')
    axes[0, 1].set_title('X2: Age')

    sns.countplot(data=cohort_df, x='X3_Abuse', hue='Treatment', ax=axes[0, 2], palette='viridis')
    axes[0, 2].set_title('X3: Substance Abuse')
    axes[0, 2].set_xticklabels(['No', 'Yes'])

    sns.countplot(data=cohort_df, x='X4_Access', hue='Treatment', ax=axes[1, 0], palette='viridis')
    axes[1, 0].set_title('X4: Access to Care')
    axes[1, 0].set_xticklabels(['Low', 'High'])

    # Grafico per l'outcome (invariato)
    sns.countplot(data=cohort_df, x='Y_Improvement', hue='Treatment', ax=axes[1, 1], palette='viridis')
    axes[1, 1].set_title('Y: Symptom Improvement')
    axes[1, 1].set_xticklabels(['No', 'Yes'])

    # Nascondi l'ultimo pannello vuoto
    axes[1, 2].axis('off')

    # --- QUESTA È LA CORREZIONE ---
    # Usa il parametro 'rect' per lasciare il 5% di spazio in alto per il titolo
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



def plot_propensity_comparison(seed=42):
    """
    Illustra la differenza tra una funzione di propensione "step" (stile BCE)
    e una "smussata" (stile BCAUSS).
    """
    np.random.seed(seed)
    n_samples = 2000

    # Usiamo la gravità della depressione come confounder principale
    severity = np.random.uniform(0, 27, n_samples)
    # Assegnazione del trattamento "reale" basata su una soglia
    T_true = (severity > 15).astype(int)

    # Funzione (A) - Stile BCE: approssima bene la realtà ma crea sbilanciamento
    def g_A_step(sev):
        return (sev > 15).astype(float)

    # Funzione (B) - Stile BCAUSS: meno accurata ma bilancia meglio
    def g_B_smooth(sev):
        p = np.zeros_like(sev, dtype=float)
        mask1 = (sev > 10) & (sev <= 15)
        p[mask1] = (sev[mask1] - 10) / 5
        mask2 = (sev > 15) & (sev <= 20)
        p[mask2] = 1 - (sev[mask2] - 15) / 5
        p[sev > 20] = 1.0
        return p

    pA = g_A_step(severity)
    pB = g_B_smooth(severity)

    # --- Inizio Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey='col')

    # --- Pannelli per la Funzione A (Step) ---
    ax = axes[0, 0]
    sns.histplot(severity, bins=30, stat='density', color='lightgrey', ax=ax)
    ax2 = ax.twinx()
    ax2.plot(np.sort(severity), g_A_step(np.sort(severity)), color='#e41a1c', lw=2.5)
    ax.set_title('(A) Propensity function (BCE style)')
    ax.set_xlabel('Depression Severity')
    ax.set_ylabel('Patient Density')
    ax2.set_ylabel('P(Treatment)')

    ax = axes[0, 1]
    sns.histplot(pA[T_true == 0], label='No Drug', stat='density', color='#377eb8', ax=ax, alpha=0.7)
    sns.histplot(pA[T_true == 1], label='Drug', stat='density', color='#ff7f0e', ax=ax, alpha=0.7)
    ax.set_xlabel('Estimated Propensity Score')
    ax.legend()

    # --- Pannelli per la Funzione B (Smussata) ---
    ax = axes[1, 0]
    sns.histplot(severity, bins=30, stat='density', color='lightgrey', ax=ax)
    ax2 = ax.twinx()
    ax2.plot(np.sort(severity), g_B_smooth(np.sort(severity)), color='#4daf4a', lw=2.5)
    ax.set_title('(B) Balancing Function (BCAUSS Style)')
    ax.set_xlabel('Depression Severity')
    ax.set_ylabel('Patient Density')
    ax2.set_ylabel('P(Treatment)')

    ax = axes[1, 1]
    sns.histplot(pB[T_true == 0], label='No Drug', stat='density', color='#377eb8', ax=ax, alpha=0.7)
    sns.histplot(pB[T_true == 1], label='Drug', stat='density', color='#ff7f0e', ax=ax, alpha=0.7)
    ax.set_xlabel('Estimated Propensity Score')
    ax.legend()

    plt.tight_layout()
    plt.show()
# ==============================================================================
# 3. PLOT DELLE FUNZIONI DI PROPENSIONE (STILE FIGURA 3)
# ==============================================================================

def plot_propensity_comparison(seed=42):
    """
    Illustra la differenza tra una funzione di propensione "step" (stile BCE)
    e una "smussata" (stile BCAUSS).
    """
    np.random.seed(seed)
    n_samples = 2000

    # Usiamo la gravità della depressione come confounder principale
    severity = np.random.uniform(0, 27, n_samples)
    # Assegnazione del trattamento "reale" basata su una soglia
    T_true = (severity > 15).astype(int)

    # Funzione (A) - Stile BCE: approssima bene la realtà ma crea sbilanciamento
    def g_A_step(sev):
        return (sev > 15).astype(float)

    # Funzione (B) - Stile BCAUSS: meno accurata ma bilancia meglio
    def g_B_smooth(sev):
        p = np.zeros_like(sev, dtype=float)
        mask1 = (sev > 10) & (sev <= 15)
        p[mask1] = (sev[mask1] - 10) / 5
        mask2 = (sev > 15) & (sev <= 20)
        p[mask2] = 1 - (sev[mask2] - 15) / 5
        p[sev > 20] = 1.0
        return p

    pA = g_A_step(severity)
    pB = g_B_smooth(severity)

    # --- Inizio Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey='col')

    # --- Pannelli per la Funzione A (Step) ---
    ax = axes[0, 0]
    sns.histplot(severity, bins=30, stat='density', color='lightgrey', ax=ax)
    ax2 = ax.twinx()
    ax2.plot(np.sort(severity), g_A_step(np.sort(severity)), color='#e41a1c', lw=2.5)
    ax.set_title('(A) Propensity function (BCE style)')
    ax.set_xlabel('Depression Severity')
    ax.set_ylabel('Patient Density')
    ax2.set_ylabel('P(Treatment)')

    ax = axes[0, 1]
    sns.histplot(pA[T_true == 0], label='No Drug', stat='density', color='#377eb8', ax=ax, alpha=0.7)
    sns.histplot(pA[T_true == 1], label='Drug', stat='density', color='#ff7f0e', ax=ax, alpha=0.7)
    ax.set_title('(A) Unbalanced Distributions')
    ax.set_xlabel('Estimated Propensity Score')
    ax.legend()

    # --- Pannelli per la Funzione B (Smussata) ---
    ax = axes[1, 0]
    sns.histplot(severity, bins=30, stat='density', color='lightgrey', ax=ax)
    ax2 = ax.twinx()
    ax2.plot(np.sort(severity), g_B_smooth(np.sort(severity)), color='#4daf4a', lw=2.5)
    ax.set_title('(B) Balancing Function (BCAUSS Style)')
    ax.set_xlabel('Depression Severity')
    ax.set_ylabel('Patient Density')
    ax2.set_ylabel('P(Treatment)')

    ax = axes[1, 1]
    sns.histplot(pB[T_true == 0], label='No Drug', stat='density', color='#377eb8', ax=ax, alpha=0.7)
    sns.histplot(pB[T_true == 1], label='Drug', stat='density', color='#ff7f0e', ax=ax, alpha=0.7)
    ax.set_title('(B) More Balanced Distributions')
    ax.set_xlabel('Estimated Propensity Score')
    ax.legend()

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 4. ESECUZIONE DELLO SCRIPT
# ==============================================================================

if __name__ == '__main__':
    # Simula i dati per il nuovo scenario
    df_psych = simulate_psych_data()

    # Mostra le distribuzioni per lo scenario RCT (bilanciato)
    print("\n--- Analisi dello scenario RCT (trattamento randomizzato) ---")
    plot_psych_panels(df_psych, 'RCT')

    # Mostra le distribuzioni per lo scenario Osservazionale (sbilanciato)
    print("\n--- Analisi dello scenario Osservazionale (con confondimento) ---")
    plot_psych_panels(df_psych, 'OBS')

    # Mostra il confronto tra le due strategie di stima della propensione
    print("\n--- Confronto tra funzioni di propensione (Stile BCE vs. BCAUSS) ---")
    plot_propensity_comparison()

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# Backend e stile grafico professionale
matplotlib.use('TkAgg')
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'CMU Serif', 'Palatino'],
    'axes.titlesize': 16,
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150
})


def plot_publication_quality_margin(n_points=50, seed=42):
    """
    Versione finale del grafico, con legenda posizionata professionalmente
    al di fuori dell'area dei dati.
    """
    np.random.seed(seed)

    # --- Palette Colori ---
    C_SIMILI = '#377eb8'
    C_DISSIMILI = '#e41a1c'
    C_POSITIVO = '#4daf4a'
    C_ANCHOR = '#ff7f0e'
    C_NEUTRO = '#555555'

    # --- Generazione Dati ---
    cluster_A = np.random.randn(n_points, 2) + np.array([0, 0])
    cluster_B = np.random.randn(n_points, 2) + np.array([2.8, 2.8])

    idx_A = np.arange(n_points)
    idx_B = np.arange(n_points, n_points * 2)
    anchor_idx, positive_idx = np.random.choice(idx_A, 2, replace=False)
    negative_idx = np.random.choice(idx_B)

    points = np.vstack([cluster_A, cluster_B])
    anchor = points[anchor_idx]
    positive = points[positive_idx]
    negative = points[negative_idx]

    d_pos = np.linalg.norm(anchor - positive)
    d_neg = np.linalg.norm(anchor - negative)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(7, 7))

    # Punti dei cluster
    ax.scatter(cluster_A[:, 0], cluster_A[:, 1], c=C_SIMILI, s=30, alpha=0.5)
    ax.scatter(cluster_B[:, 0], cluster_B[:, 1], c=C_DISSIMILI, marker='X', s=30, alpha=0.5)

    # Evidenzia i punti chiave
    ax.scatter(anchor[0], anchor[1], c=C_ANCHOR, marker='o', s=150, edgecolors='black', linewidth=1.5, zorder=10)
    ax.scatter(positive[0], positive[1], c=C_SIMILI, s=70, edgecolors='black', linewidth=0.5, zorder=10)
    ax.scatter(negative[0], negative[1], c=C_DISSIMILI, marker='X', s=70, edgecolors='black', linewidth=0.5, zorder=10)

    # Margine (Anello e Cerchi)
    annulus = patches.Annulus(anchor, r=d_neg, width=d_neg - d_pos, facecolor=C_NEUTRO, alpha=0.1)
    circle_p = patches.Circle(anchor, radius=d_pos, fill=False, edgecolor=C_POSITIVO, linewidth=1.5, linestyle='-')
    circle_n = patches.Circle(anchor, radius=d_neg, fill=False, edgecolor=C_DISSIMILI, linewidth=1.5, linestyle='--')
    ax.add_patch(annulus)
    ax.add_patch(circle_p)
    ax.add_patch(circle_n)

    # Annotazioni
    arrow_kw = dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color=C_NEUTRO, lw=1)
    bbox_kw = dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7)
    ax.annotate('Anchor', xy=anchor, xytext=(-2.5, 1.8), arrowprops=arrow_kw, bbox=bbox_kw)
    ax.annotate(f'Positive\n$d={d_pos:.2f}$', xy=positive, xytext=(1.5, -1.8), ha='center', arrowprops=arrow_kw,
                bbox=bbox_kw)
    ax.annotate(f'Negative\n$d={d_neg:.2f}$', xy=negative, xytext=(1.5, 4.5), ha='center', arrowprops=arrow_kw,
                bbox=bbox_kw)

    # --- Rifiniture Finali ---

    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_aspect('equal', 'box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)

    # --- MODIFICA CHIAVE PER LA LEGENDA ---
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', label='Similar Points (similar ITE)', markerfacecolor=C_SIMILI,
               markersize=8),
        Line2D([0], [0], marker='X', color='w', label='Dissimilar Points (different ITE)', markerfacecolor=C_DISSIMILI,
               markersize=8),
        patches.Patch(facecolor=C_NEUTRO, alpha=0.3, label='Valid Margin Area ($d_{pos} < m < d_{neg}$)')
    ]
    # Posiziona la legenda più in alto per distanziarla dal titolo
    ax.legend(handles=legend_elems,
              loc='lower center',
              bbox_to_anchor=(0.5, 1.08),  # Aumentato il valore di y per più spazio
              frameon=False,
              ncol=3)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_publication_quality_margin()
