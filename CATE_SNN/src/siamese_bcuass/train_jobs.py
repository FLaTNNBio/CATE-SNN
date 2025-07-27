import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.models.bcauss import BCAUSS
from src.siamese_bcuass.siamese import SiameseBCAUSS
from src.contrastive import DynamicContrastiveCausalDS

# -------------------
# CONFIGURAZIONE
# -------------------
NPZ_TRAIN_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../data/jobs_DW_bin.new.10.train.npz')
)
NPZ_TEST_PATH  = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../data/jobs_DW_bin.new.10.test.npz')
)
REPLICA_INDEX = 0  # replica (0-9)
OUTPUT_DIR    = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../outputs_jobs')
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------
# IPERPARAMETRI
# -------------------
# Warm-up BCAUSS
WARMUP_EPOCHS_BASE = 5
LR_BASE            = 1e-3
WEIGHT_DECAY_BASE  = 1e-5
USE_BCE            = True
SCALE_PREDS        = False

# Siamese-BCAUSS
MARGIN               = 1.0
LAMBDA_CTR           = 1.0
BATCH_SIZE_SIAMESE   = 128
LR_SIAMESE           = 1e-4
EPOCHS_SIAMESE       = 50
CLIP_NORM            = 1.0
USE_AMP              = False
VAL_SPLIT_SIAMESE    = 0.2
SIAMESE_PATIENCE     = 5
UPDATE_ITE_FREQ      = 1
VERBOSE              = True

# -------------------
# CARICAMENTO DATI
# -------------------
print(f"Carico TRAIN da: {NPZ_TRAIN_PATH}")
train = np.load(NPZ_TRAIN_PATH)
X_train = train['x'][REPLICA_INDEX]
T_train = train['t'][REPLICA_INDEX]
Y_train = train['yf'][REPLICA_INDEX]

print(f"Carico TEST da:  {NPZ_TEST_PATH}")
test = np.load(NPZ_TEST_PATH)
X_test  = test['x'][REPLICA_INDEX]
T_test  = test['t'][REPLICA_INDEX]
Y_test  = test['yf'][REPLICA_INDEX]

# -------------------
# PREPROCESSING
# -------------------
print("Standardizzo covariate...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# -------------------
# WARM-UP MANUALE BCAUSS
# -------------------
print("Warm-up base BCAUSS (manuale)...")
base = BCAUSS(
    input_dim=X_train.shape[1],
    learning_rate=LR_BASE,
    reg_l2=WEIGHT_DECAY_BASE,
    use_bce=USE_BCE,
    scale_preds=SCALE_PREDS,
    verbose=VERBOSE
)
base.fit(
    X_train,
    T_train,
    Y_train.reshape(-1,1),
    epochs=WARMUP_EPOCHS_BASE
)
print("Fine warm-up base BCAUSS.")

# -------------------
# TRAINING SIAMESE-BCAUSS
# -------------------
siamese_params = {
    'ds_class':          DynamicContrastiveCausalDS,
    'margin':            MARGIN,
    'lambda_ctr':        LAMBDA_CTR,
    'batch_size':        BATCH_SIZE_SIAMESE,
    'lr':                LR_SIAMESE,
    'epochs':            EPOCHS_SIAMESE,
    'clip_norm':         CLIP_NORM,
    'use_amp':           USE_AMP,
    'val_split':         VAL_SPLIT_SIAMESE,
    'patience':          SIAMESE_PATIENCE,
    'update_ite_freq':   UPDATE_ITE_FREQ,
    'warmup_epochs_base':0,        # disabilita warm-up interno
    'verbose':           VERBOSE
}

print("Inizio training di SiameseBCAUSS…")
model_siamese = SiameseBCAUSS(base_model=base, **siamese_params)
best_path = os.path.join(OUTPUT_DIR, 'best_siamese_bcauss_jobs.pth')
model_siamese.fit(
    X_train,
    T_train,
    Y_train.reshape(-1,1),
    best_model_path=best_path
)

# -------------------
# VALUTAZIONE E METRICHE
# -------------------
print("Valuto su test set…")
ite_test = model_siamese.predict_ite(X_test)

df_metrics = pd.DataFrame({
    'ITE_mean_test': [np.mean(ite_test)],
    'ITE_std_test':  [np.std(ite_test)]
})
if 'e' in test and 'I' in test:
    mask = test['e'][REPLICA_INDEX].astype(bool)
    df_metrics['ATT_pred_rct'] = np.mean(ite_test[mask])
    df_metrics['R_policy_rct'] = (
        np.sum((ite_test[mask] > 0) == test['I'][REPLICA_INDEX]) / mask.sum()
    )

metrics_path = os.path.join(OUTPUT_DIR, 'metrics_jobs.csv')
df_metrics.to_csv(metrics_path, index=False)
print(f"Metriche salvate in: {metrics_path}")
