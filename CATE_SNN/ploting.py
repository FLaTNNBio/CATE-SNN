import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')       # o 'Qt5Agg', oppure 'Agg' se vuoi solo salvare

np.random.seed(0)
N = 200_000

# Variabili comuni (X1-X4)
gender = np.random.binomial(1, 0.55, N)  # 1 = Female, 0 = Male
age = np.clip(np.random.normal(45, 25, N), 18, 70)
CCI = np.clip(np.random.normal(3.5, 2.5, N), 0, 7)

# Smoking: probabilità dipendente da età e CCI (come nel paper)
smoke_prob = np.clip(0.1 + 0.5 * age / 71 + 0.3 * CCI / 7, 0, 1)
smoking = np.random.binomial(1, smoke_prob)

# 🧪 RCT: trattamento assegnato a caso
t_rct = np.random.binomial(1, 0.5, N)

# 🧪 OBS: trattamento dipende da età, CCI e fumo
logit = -0.01 + 3.75 * age / 71 + 3.4 * CCI / 7 + 0.5 * smoking
prob_obs = 1 / (1 + np.exp(-logit))
t_obs = np.random.binomial(1, prob_obs)

# Output Y (ospedalizzazione/morte entro 3 anni)
def simulate_outcome(A, age, CCI, smoke):
    mu = 0.09 - 0.11*A - 0.03*A*(1 - smoke) + 0.25 * age / 71 + 0.3 * CCI / 7 + 0.3 * smoke
    return np.random.binomial(1, mu)

# Simuliamo Y per entrambe le versioni
Y_rct = simulate_outcome(t_rct, age, CCI, smoking)
Y_obs = simulate_outcome(t_obs, age, CCI, smoking)

# Costruiamo i dataframe
df_rct = pd.DataFrame({'X1': gender, 'X2': age, 'X3': CCI, 'X4': smoking, 'T': t_rct, 'Y': Y_rct, 'cohort': 'RCT'})
df_obs = pd.DataFrame({'X1': gender, 'X2': age, 'X3': CCI, 'X4': smoking, 'T': t_obs, 'Y': Y_obs, 'cohort': 'OBS'})
df = pd.concat([df_rct, df_obs])

# 🖼️ PASSO 3: Plot (figura in stile paper)
def plot_panel(df, cohort_name):
    cohort_df = df[df['cohort'] == cohort_name]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    sns.countplot(x='X1', hue='T', data=cohort_df, ax=axes[0, 0])
    axes[0, 0].set_title('X1 (gender)')
    axes[0, 0].set_xticklabels(['Male', 'Female'])

    sns.kdeplot(x='X2', hue='T', data=cohort_df, fill=True, ax=axes[0, 1])
    axes[0, 1].set_title('X2 (age)')

    sns.kdeplot(x='X3', hue='T', data=cohort_df, fill=True, ax=axes[0, 2])
    axes[0, 2].set_title('X3 (CCI)')

    sns.countplot(x='X4', hue='T', data=cohort_df, ax=axes[1, 0])
    axes[1, 0].set_title('X4 (smoking)')
    axes[1, 0].set_xticklabels(['Non-smoker', 'Smoker'])

    sns.countplot(x='Y', hue='T', data=cohort_df, ax=axes[1, 1])
    axes[1, 1].set_title('Y (Outcome)')
    axes[1, 1].set_xticklabels(['No Event', 'Event'])

    fig.suptitle(cohort_name, fontsize=16)
    plt.tight_layout()
    plt.show()

# 🔍 Visualizziamo i pannelli
plot_panel(df, 'RCT')
plot_panel(df, 'OBS')
