#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Global experiment variables
DATASET="ciciomt2024"
SEEDS="[42,52,62]"
GPU="0"
BATCH_SIZE=256
TRADES_BETA=6.0

# SOTA MLP Parameters
MLP_EPOCHS=50
MLP_PATIENCE=5
MLP_LR=1e-3
MLP_WD=1e-4
MLP_DIMS="[64,32]"

# SOTA RF Parameters
RF_TREES=100
RF_MAX_FEATURES="null" # Maps to 'None' in Python

echo "================================================================="
echo " Starting SOTA NIDS Experiments on $DATASET"
echo " Seeds: $SEEDS"
echo "================================================================="

# -------------------------------------------------------------------------
# 1. MLP EXPERIMENTS (PyTorch Target)
# -------------------------------------------------------------------------
echo ">>> [1/6] Training MLP: Clear Defense Mode (Baseline)"
python main.py action=train_nids \
    model=mlp \
    dataset=$DATASET \
    model.training.epochs=$MLP_EPOCHS \
    model.training.batch_size=$BATCH_SIZE \
    model.training.lr=$MLP_LR \
    model.training.weight_decay=$MLP_WD \
    model.training.early_stopping_patience=$MLP_PATIENCE \
    model.params.hidden_dims=$MLP_DIMS \
    model.training.defense_mode=none \
    seeds=$SEEDS \
    gpu=$GPU

echo ">>> [2/6] Training MLP: Adversarial Training (TRADES)"
python main.py action=train_nids \
    model=mlp \
    dataset=$DATASET \
    model/training=trades_at \
    model.training.epochs=$MLP_EPOCHS \
    model.training.batch_size=$BATCH_SIZE \
    model.training.lr=$MLP_LR \
    model.training.weight_decay=$MLP_WD \
    model.training.early_stopping_patience=$MLP_PATIENCE \
    model.params.hidden_dims=$MLP_DIMS \
    model.training.trades_beta=$TRADES_BETA \
    seeds=$SEEDS \
    gpu=$GPU

echo ">>> [3/6] Training MLP: Adversarial Training (TRADES) + Mahalanobis Detector"
python main.py action=train_nids \
    model=mlp \
    dataset=$DATASET \
    model/training=trades_at \
    model.training.epochs=$MLP_EPOCHS \
    model.training.batch_size=$BATCH_SIZE \
    model.training.lr=$MLP_LR \
    model.training.weight_decay=$MLP_WD \
    model.training.early_stopping_patience=$MLP_PATIENCE \
    model.params.hidden_dims=$MLP_DIMS \
    model.training.trades_beta=$TRADES_BETA \
    detector.enabled=true \
    detector.name=mahalanobis \
    seeds=$SEEDS \
    gpu=$GPU


# -------------------------------------------------------------------------
# 2. RANDOM FOREST EXPERIMENTS (Scikit-Learn Target)
# -------------------------------------------------------------------------
echo ">>> [4/6] Training Random Forest: Clear Defense Mode (Baseline)"
python main.py action=train_nids \
    model=random_forest \
    dataset=$DATASET \
    model.params.n_estimators=$RF_TREES \
    model.params.max_features=$RF_MAX_FEATURES \
    model.training.defense_mode=none \
    seeds=$SEEDS \
    gpu=$GPU

echo ">>> [5/6] Training Random Forest: Empirical AT (TRADES Surrogate)"
# Note: EAT uses a differentiable surrogate to craft the TRADES boundary samples.
# The surrogate mirrors the exact same MLP architecture size and early stopping.
python main.py action=train_nids \
    model=random_forest \
    dataset=$DATASET \
    model.params.n_estimators=$RF_TREES \
    model.params.max_features=$RF_MAX_FEATURES \
    model/training=empirical_at \
    model.training.empirical_at.surrogate.training.epochs=$MLP_EPOCHS \
    model.training.empirical_at.surrogate.training.early_stopping_patience=$MLP_PATIENCE \
    model.training.empirical_at.surrogate.params.hidden_dims=$MLP_DIMS \
    model.training.empirical_at.attack.trades_beta=$TRADES_BETA \
    seeds=$SEEDS \
    gpu=$GPU

echo ">>> [6/6] Training Random Forest: Empirical AT (TRADES) + Mahalanobis Detector"
python main.py action=train_nids \
    model=random_forest \
    dataset=$DATASET \
    model.params.n_estimators=$RF_TREES \
    model.params.max_features=$RF_MAX_FEATURES \
    model/training=empirical_at \
    model.training.empirical_at.surrogate.training.epochs=$MLP_EPOCHS \
    model.training.empirical_at.surrogate.training.early_stopping_patience=$MLP_PATIENCE \
    model.training.empirical_at.surrogate.params.hidden_dims=$MLP_DIMS \
    model.training.empirical_at.attack.trades_beta=$TRADES_BETA \
    detector.enabled=true \
    detector.name=mahalanobis \
    seeds=$SEEDS \
    gpu=$GPU

echo "================================================================="
echo " All 6 training runs completed successfully!"
echo "================================================================="