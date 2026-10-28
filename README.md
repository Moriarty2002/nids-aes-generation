# NIDS Adversarial Benchmark Framework

A modular, extensible PyTorch benchmarking framework for evaluating state-of-the-art (SOTA) adversarial evasion attacks against machine learning and deep learning-based Network Intrusion Detection Systems (NIDS).

---

## Table of Contents
- [NIDS Adversarial Benchmark Framework](#nids-adversarial-benchmark-framework)
  - [Table of Contents](#table-of-contents)
  - [1. Motivation \& Design Philosophy](#1-motivation--design-philosophy)
    - [Framework Goal](#framework-goal)
    - [Execution Mechanics](#execution-mechanics)
  - [2. System Architecture](#2-system-architecture)
    - [2.1 End-to-End Pipeline Flow](#21-end-to-end-pipeline-flow)
    - [2.2 Core Subsystems](#22-core-subsystems)
      - [1. Configuration \& Run Isolation (`main.py`, `configs/`)](#1-configuration--run-isolation-mainpy-configs)
      - [2. Multi-Process Orchestrator (`orchestrator/`)](#2-multi-process-orchestrator-orchestrator)
      - [3. Polymorphic Model Engine \& Sklearn Support (`models/`)](#3-polymorphic-model-engine--sklearn-support-models)
      - [4. Dual-Mode Task Engine: AD vs. AC (`data/`, `workflow.py`)](#4-dual-mode-task-engine-ad-vs-ac-data-workflowpy)
      - [5. Threat Models \& Attack Engines (`attacks/`)](#5-threat-models--attack-engines-attacks)
      - [6. Universal Constraint Gatekeeper (`constraints/`)](#6-universal-constraint-gatekeeper-constraints)
      - [7. Evaluation Engine \& Bounded Memory Management (`evaluator/`)](#7-evaluation-engine--bounded-memory-management-evaluator)
      - [8. Crash-Resilient Reporting Engine (`evaluator/logger.py`)](#8-crash-resilient-reporting-engine-evaluatorloggerpy)
    - [2.3 Framework Directory Layout](#23-framework-directory-layout)
  - [3. Installation \& Setup](#3-installation--setup)
  - [4. CLI \& Configuration Reference](#4-cli--configuration-reference)
    - [Core Global Parameters](#core-global-parameters)
    - [Task Modes (AD vs. AC)](#task-modes-ad-vs-ac)
    - [Component Configuration](#component-configuration)
    - [Defense \& Training Flags (`model.training.*`)](#defense--training-flags-modeltraining)
    - [Inference-Time Detector Flags (`detector.*`)](#inference-time-detector-flags-detector)
    - [Evaluation \& Memory Safety Flags (`evaluation.*`)](#evaluation--memory-safety-flags-evaluation)
  - [5. Operational Workflows](#5-operational-workflows)
    - [5.1 Target NIDS Training (Clean, AT, TRADES, \& Empirical AT)](#51-target-nids-training-clean-at-trades--empirical-at)
      - [Standard Training (Clean Baseline)](#standard-training-clean-baseline)
      - [PyTorch Adversarial Training (PGD-AT \& TRADES)](#pytorch-adversarial-training-pgd-at--trades)
      - [Empirical Adversarial Training for Sklearn Targets (EAT)](#empirical-adversarial-training-for-sklearn-targets-eat)
      - [Inference-Time Detector Fitting](#inference-time-detector-fitting)
    - [5.2 Attack Model Training (Generative / Surrogates)](#52-attack-model-training-generative--surrogates)
    - [5.3 Adversarial Evaluation \& Detection](#53-adversarial-evaluation--detection)
      - [Evaluating Evasion \& Enabling the Detector](#evaluating-evasion--enabling-the-detector)
      - [Large-Dataset Memory Safety \& Subsampling](#large-dataset-memory-safety--subsampling)
    - [5.4 Fast Smoke Testing](#54-fast-smoke-testing)
  - [6. Evaluation Metrics \& Statistical Protocol](#6-evaluation-metrics--statistical-protocol)
    - [Imbalance-Aware Defense Metrics](#imbalance-aware-defense-metrics)
    - [Attacker Performance Metrics](#attacker-performance-metrics)
    - [Statistical Protocol](#statistical-protocol)
  - [7. Extending the Framework (Developer Guide)](#7-extending-the-framework-developer-guide)
    - [Adding a New Attack](#adding-a-new-attack)
    - [Adding a Custom Constraint](#adding-a-custom-constraint)
    - [Adding a New Inference-Time Detector](#adding-a-new-inference-time-detector)
  - [8. Credits \& Acknowledgements](#8-credits--acknowledgements)

---

## 1. Motivation & Design Philosophy

### Framework Goal
A significant portion of adversarial machine learning literature suffers from reproducibility and comparability issues: researchers evaluate attacks against different target models, non-standardized feature spaces, or single-seed runs without physical protocol constraint checks or defensive baselines.

This framework addresses these shortcomings by providing a standardized, multi-seed evaluation harness. It decouples dataset handling, model wrappers, threat models, physical protocol gatekeeping, and metric reporting to allow seamless plug-and-play evaluation. **Crucially, it implements a Dual Authoritative Representation architecture**, mathematically separating unassailable physical constraints (`float64`) from normalized continuous model inference (`float32`) to resolve representation drift.

### Execution Mechanics

The framework employs a two-tier execution lifecycle (parent orchestrator $\rightarrow$ worker seeds):

1. [`main.py`](./main.py) resolves Hydra configurations, initializes logging and run directories, discovers registered plugins, and manages seed execution.
2. [`orchestrator/runner.py`](./orchestrator/runner.py) spawns isolated worker processes, assigning a deterministic random seed and target GPU/CPU device to each worker.
3. [`workflow.py`](./workflow.py) executes worker-level routines: `train_nids_seed`, `train_attack_seed`, `evaluate_attack_seed`, and `preprocess_seed`.
4. Workers handle data loading, normalization schema matching, model/attack instantiation, strict constraint checks, and seed-local artifact serialization.
5. `main.py` aggregates numeric metrics across all completed seed workers to compute cross-seed statistics (Mean $\pm$ Std) and Wilcoxon significance tests.

---

## 2. System Architecture

### 2.1 End-to-End Pipeline Flow

The execution graph strictly separates adversarial generation, physical protocol validation, and victim model scoring:

```text
[Hydra CLI / Config]
        │
        ▼
[main.py Orchestrator] ── (Resolves run_id & output paths, saves resolved config)
        │
        ▼
[ProcessPoolExecutor (spawn)] ── Distributes Seeds & GPU/CPU Assignments
        │
        ├── Worker: Seed 1 (Device cuda:0) ── logs/seed_1.log
        ├── Worker: Seed N ...
                │
                ▼
        [Dataset Loading & DAR Extraction] ── Extracts absolute physical baseline (float64)
                │                             before normalizing for the model (float32)
                ▼
        [Clean Baseline Inference] ── Establishes baseline MCC, FAR, DR, Accuracy
                │
                ▼
        [Attack Optimization] ── Targeted optimization toward Benign (Label 0)
          ├── Continuous Generation (float32): PGD, TRADES, C&W, Square, TDTCE (Diffusion)
          ├── Physical Projection (float64): Discretization (optional) & immutable restorations
          ├── Feature-Space: PGD / TdTce / etc. on normalized vectors
          └── Problem-Space: PCAP mutation -> SOTA Feature Extractor
                │
                ▼
        [Universal Constraint Gatekeeper] ── Realizability referee in strict physical domain
          ├── Bitwise Sync Check: Verifies float64 physical encodes exactly to float32 model input
          ├── Checks (on float64): Exact bounds, discrete protocol rules, victim immutability
          └── Telemetry: Realizability Rate (RR) & per-rule violation counters
                │
                ▼
        [Target Model Scoring] ── Bounded batch scoring (prunes unrealizable samples if strict)
                │
                ▼
        [Per-Seed Reporting] ── Writes predictions (.npz), text reports, confusion matrices
                │
                ▼
[Post-Run Aggregator (main.py)] ── Computes Mean ± Std, Wilcoxon p-values, generates mean CMs & multi-seed learning curves
```

---

### 2.2 Core Subsystems

#### 1. Configuration & Run Isolation (`main.py`, `configs/`)
* **Hydra I/O Control:** Default Hydra directory generation is suppressed (`output_subdir: null`). Output directories are deterministically generated by the framework.
* **Deterministic Single-Evaluation Path:** Output directories follow a unified convention:
  `runs/<action>/<dataset>/<model>/<attack_variant>/<timestamp>_<hash>/`

#### 2. Multi-Process Orchestrator (`orchestrator/`)
* **Process Isolation:** Worker seeds execute in isolated Python processes via `spawn` to prevent CUDA memory fragmentation and global state bleeding.
* **Device Partitioning (`orchestrator/device.py`):** Automatically assigns seeds across physical GPUs (`gpu_partitions: [cuda:0, cuda:1]`) or CPU cores.
* **Telemetry Decoupling (`orchestrator/runner.py`):** The parent process manages the main progress bar (`tqdm`), while individual workers log to dedicated seed-specific log files.

#### 3. Polymorphic Model Engine & Sklearn Support (`models/`)
* **Unified Model Abstract (`ModelWrapper`):** Both PyTorch neural backbones (`PyTorchModelWrapper`) and scikit-learn estimators (`SklearnModelWrapper`, e.g., Random Forest, XGBoost) share a unified interface.
* **Empirical Adversarial Training (EAT):** Non-differentiable target models (such as decision tree ensembles) cannot compute input gradients $\nabla_x \mathcal{L}$. For these targets, `defense_mode: empirical_at` fits a differentiable PyTorch surrogate (e.g., `jmila_surrogate`), generates gatekeeper-filtered realizable adversarial samples (via PGD or TRADES inner-loop optimization), and merges them into the training set prior to calling `estimator.fit()`.
* **Dynamic Tensor Reshaping:** Automatically reshapes inputs based on architecture type:
  * Dense models (MLP, Random Forest, XGBoost) receive 2-D flattened vectors `(N, T*F)`.
  * Spatial/Recurrent models (1D-CNN, LSTM, Transformer) receive 3-D sequence tensors `(N, T, F)`.

#### 4. Dual-Mode Task Engine: AD vs. AC (`data/`, `workflow.py`)
* **Anomaly Detection (AD):** Activates when binary numeric labels (`0`, `1`) are detected (`0 = Benign`, `1 = Malicious`).
* **Attack Classification (AC):** Activates when multi-class labels are present. The framework maps `benign_string` to class index `0` and attack classes to `1..K`.
* **Strict Targeted Evasion:** Confusing one attack type for another still triggers defensive alerts. Attacks optimize explicitly toward the Benign class (`target_label = 0`), and Attack Success Rate (ASR) measures forced predictions to `0`.
* **Physical Source Authority:** Datasets (e.g., `SequenceParquetDataset`) natively extract padding and sequence attributes in `float64` to preserve pristine physical metrics before canonical model normalization. 

#### 5. Threat Models & Attack Engines (`attacks/`)
* **Feature-Space Pipeline (`attacks/feature_space/`):** Implements gradient-based (PGD, BIM, C&W, TRADES), gradient-free (Square Attack), and generative attacks (TDTCE - Latent Diffusion).
* **TRADES as an Attack Plugin (`TRADESAttack`):** Uses TRADES' inner KL-divergence maximization loop to explore prediction boundary instability. This creates transfer-capable adversarial samples for robustifying non differentiable models during EAT.
* **Autograd Isolation:** Iterative gradient attacks enforce strict tensor detaching (`logits.detach()`, `loss.detach()`, `grad.detach()`), zeroing, and `del` calls per step to prevent autograd graph memory leaks. Non-gradient forward passes execute inside `with torch.no_grad():`.
* **Dual Representation Intercept:** `FeatureSpaceAttack` wrappers intercept continuous `float32` adversarial outputs, project them into a `float64` workspace to snap discrete features (PL, TTL, WIN) to perfect integers, and flawlessly restore immutable victim bytes. 
* **Problem-Space Pipeline (`attacks/problem_space/`):** Modifies raw PCAP packets/bytes and re-extracts features dynamically using `BasePreprocessor`.

#### 6. Universal Constraint Gatekeeper (`constraints/`)
* **Decoupled Evaluation:** Sits strictly downstream of attack generation. Attacks do not have access to gatekeeper rules during optimization.
* **Absolute Physical Truth:** Rejects drift-tolerance hacks. Validates exact constraints (e.g., `< 0.0`) on `float64` tensors where zero is mathematically exact.
* **Bitwise Sync Verification:** Before executing any physical rules, it strictly verifies that the `float64` physical tensor precisely maps to the target NIDS `float32` memory layout, guaranteeing the oracle and NIDS evaluate the exact same payload. 
* **Realizability Telemetry:** Computes the **Realizability Rate** ($RR = \frac{N_{\text{valid}}}{N_{\text{total}}}$). When `strict_constraints: true`, unrealizable flows are discarded before scoring.

#### 7. Evaluation Engine & Bounded Memory Management (`evaluator/`)
* **Bounded Batch Processing:** All prediction passes, detector scoring, and distance calculations execute in configurable batch chunks to prevent OOM errors (e.g., `evaluation.batch_size=256`).
* **Float64 Lifecycle Management:** To prevent memory bloat, `workflow.py` aggressively manages massive `float64` datasets, actively destroying arrays via `del` immediately after EAT or DAR projections conclude.
* **Dataset Subsampling (`evaluation.max_samples`):** Enables deterministic pre-normalization subsampling for evaluating giant datasets efficiently.

#### 8. Crash-Resilient Reporting Engine (`evaluator/logger.py`)
* **Atomic Telemetry:** `FlushFileHandler` calls `os.fsync()` after writes to preserve logs in the event of job interruption.
* **Prediction Archives:** Saves ground-truth labels, clean predictions, adversarial predictions, and detector scores to compressed archives (`seed_<seed>_preds.npz`).
* **Cross-Seed Aggregation:** Aggregates multi-seed CSVs into `summary.json`, text summaries (Mean $\pm$ Std), confusion matrix plots, and composite learning curves (`epoch_loss.pdf`, `epoch_metrics.pdf`).

---

### 2.3 Framework Directory Layout

```text
nids-adv-benchmark/
├── configs/                          # Hierarchical Hydra configurations
│   ├── config.yaml                   # Core runtime options and output paths
│   ├── attack/                       # Attack parameters (pgd.yaml, square.yaml, trades.yaml)
│   ├── dataset/                      # Dataset configurations and label schemas
│   ├── model/                        # Model architectures and training parameters
│   └── preprocessing/                # PCAP feature extractor configurations
│
├── core/
│   ├── dual_representation.py        # Canonical float32 model/float64 physical codecs
│   ├── exceptions.py                 # Domain exceptions (ConfigurationError, etc.)
│   └── registry.py                   # ComponentRegistry supporting decorator registration
│
├── data/
│   ├── base.py                       # BaseDataset interface and NormalizationStats
│   └── ciciomt2024.py                # SequenceParquetDataset with float64 preservation
│
├── defenses/
│   ├── base.py                       # BaseDetector interface (fit, score, state_dict)
│   └── detectors.py                  # Mahalanobis and reactive detector plugins
│
├── preprocessing/
│   ├── base.py                       # Preprocessor interface contract
│   └── sota_extractor.py             # PCAP-to-vector extraction engine
│
├── constraints/
│   ├── rules.py                      # Physical and strict float64 RFC constraint definitions
│   └── gatekeeper.py                 # UniversalConstraintGatekeeper dual-mode referee
│
├── models/
│   ├── architectures/                # PyTorch models (MLP, CNN, LSTM, Transformer)
│   ├── trainer.py                    # PyTorch training loops with live CSV logging
│   └── wrapper.py                    # ModelWrapper for PyTorch and Sklearn models
│
├── attacks/
│   ├── base.py                       # BaseAttack, FeatureSpaceAttack, ProblemSpaceAttack
│   ├── feature_space/                # Feature attacks (PGD, Square, TdTce, C&W, etc.)
│   └── problem_space/                # PCAP mutators and flow injectors
│
├── evaluator/
│   ├── engine.py                     # EvaluationEngine coordinating target scoring
│   ├── logger.py                     # ExperimentReporter, FlushFileHandler, crash hooks
│   ├── metrics.py                    # Multi-class MCC, strict ASR, AdvEOC, and distances
│   └── statistical.py                # Cross-seed aggregation and Wilcoxon testing
│
├── orchestrator/
│   ├── device.py                     # CUDA pinning and fallback management
│   └── runner.py                     # Multiprocessing seed executor (spawn)
│
├── workflow.py                       # Worker entry points with strict memory lifecycle hooks
└── main.py                           # CLI entry point and post-run aggregator
```

---

## 3. Installation & Setup

**Prerequisites:** Linux (Ubuntu 20.04+), Python 3.10+, PyTorch 2.0+. CUDA is optional and used when `gpu` or `gpu_partitions` selects a CUDA device.

```bash
# Clone repository
git clone [https://github.com/Moriarty2002/nids-aes-generation.git](https://github.com/Moriarty2002/nids-aes-generation.git)
cd nids-aes-generation

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Verify setup and available plugins:

```bash
python main.py --help
```

---

## 4. CLI & Configuration Reference

### Core Global Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `action` | `str` | *Required* | Workflow to execute: `preprocess`, `train_nids`, `train_attack`, `evaluate_attack`. |
| `gpu` | `str` | `cpu` | Target GPU/CPU device (`0`, `cuda:0`, or `cpu`). |
| `seeds` | `list[int]` | `[1,2,3,4,5]`| Random seeds for execution. |
| `parallel_workers` | `int` | `5` | Maximum number of concurrent worker processes. |

### Task Modes (AD vs. AC)
* **Anomaly Detection (Binary):** Configured when `label_column` contains binary values (`0` and `1`).
* **Attack Classification (Multi-Class):** Configured when `label_column` contains multi-class strings or integers. The framework assigns `0` to `benign_string` and treats non-zero values as attack classes.

### Component Configuration
| Group | Description |
| :--- | :--- |
| `dataset` | Dataset adapter, file path, sequence features, and label column. |
| `model` | Target architecture, training strategy, hyperparameters, and checkpoint paths. |
| `attack` | Evasion attack implementation, step size, perturbation budget, and targeted options. |
| `preprocessing` | Raw PCAP feature extraction settings. |
| `detector` | Inference-time detection plugin (e.g., Mahalanobis). |

### Defense & Training Flags (`model.training.*`)
* `defense_mode` (`str`, default `"none"`): Defense strategy. Options: `"none"`, `"pgd"`, `"trades"`, or `"empirical_at"`.
* `strategy` (`str`, default `"auto"`): Execution strategy (`"auto"`, `"pytorch"`, or `"sklearn"`).
* `pgd_epsilon`, `pgd_alpha`, `pgd_steps`: Inner-loop adversarial parameters for PGD training.
* `trades_beta` (`float`, default `6.0`): Regularization weight ($\beta$) for TRADES training.
* `empirical_at.*`: Configuration block for Empirical Adversarial Training on sklearn targets (defines surrogate model, augmentation ratio, and realizability filtering).

### Inference-Time Detector Flags (`detector.*`)
* `detector.enabled` (`bool`, default `false`): Enables fitting/evaluating an inference-time detector.
* `detector.name` (`str`, default `"mahalanobis"`): Detector plugin name.

### Evaluation & Memory Safety Flags (`evaluation.*`)
* `evaluation.batch_size` (`int`, default `256`): Sample batch size for attack generation, target inference, and constraint checks.
* `evaluation.max_samples` (`int` or `null`, default `null`): Subsamples the dataset to `max_samples` before running normalization and evaluation.
* `evaluation.strict_constraints` (`bool`, default `false`): Prunes unrealizable adversarial samples from target model scoring when `true`.

---

## 5. Operational Workflows

### 5.1 Target NIDS Training (Clean, AT, TRADES, & Empirical AT)

#### Standard Training (Clean Baseline)
```bash
python main.py \
  action=train_nids \
  model=transformer \
  dataset=ciciomt2024 \
  model.training.epochs=30 \
  model.training.batch_size=512 \
  seeds=[1,2,3,4,5] \
  gpu=0
```

#### PyTorch Adversarial Training (PGD-AT & TRADES)
For differentiable PyTorch models, use `defense_mode=pgd` or `defense_mode=trades`:

```bash
python main.py \
  action=train_nids \
  model=mlp \
  dataset=ciciomt2024 \
  model.training.defense_mode=trades \
  model.training.pgd_epsilon=0.04 \
  model.training.pgd_steps=10 \
  model.training.trades_beta=6.0 \
  seeds=[1,2,3,4,5] \
  gpu=0
```

#### Empirical Adversarial Training for Sklearn Targets (EAT)
For non-differentiable scikit-learn models (e.g., Random Forest), use `defense_mode=empirical_at` with a differentiable surrogate model (`jmila_surrogate`):

```bash
python main.py \
  action=train_nids \
  model=random_forest \
  dataset=ciciomt2024 \
  model.training.defense_mode=empirical_at \
  model.training.empirical_at.surrogate.name=jmila_surrogate \
  model.training.empirical_at.augmentation.ratio=0.5 \
  model.training.empirical_at.augmentation.filter_realizable=true \
  seeds=[1,2,3,4,5] \
  gpu=0
```

#### Inference-Time Detector Fitting
Fit a Mahalanobis detector on clean training features after model training:

```bash
python main.py \
  action=train_nids \
  model=mlp \
  dataset=ciciomt2024 \
  detector.enabled=true \
  detector.name=mahalanobis \
  seeds=[1,2,3,4,5] \
  gpu=0
```

---

### 5.2 Attack Model Training (Generative / Surrogates)
For attacks requiring pre-trained generative models (like TDTCE) or substitute generators:

```bash
python main.py \
  action=train_attack \
  attack=tdtce \
  attack.params.epochs=30 \
  seeds=[1,2,3,4,5] \
  gpu=0
```

---

### 5.3 Adversarial Evaluation & Detection

#### Evaluating Evasion & Enabling the Detector
Evaluate a trained NIDS against an evasion attack (e.g., PGD) and score the inference-time detector:

```bash
python main.py \
  action=evaluate_attack \
  gpu=0 \
  attack=pgd \
  attack.epsilon=0.04 \
  attack.steps=20 \
  model=mlp \
  model.checkpoint="../runs/train_nids/ciciomt2024/mlp/baseline/<TIMESTAMP>/checkpoints/mlp_seed{seed}/nids.pt" \
  dataset=ciciomt2024 \
  evaluation.batch_size=256 \
  detector.enabled=true \
  seeds=[1,2,3,4,5]
```

#### Large-Dataset Memory Safety & Subsampling
For large datasets, use `evaluation.max_samples` to bound evaluation memory:

```bash
python main.py \
  action=evaluate_attack \
  gpu=0 \
  attack=pgd \
  dataset=ciciomt2024 \
  evaluation.max_samples=10000 \
  evaluation.batch_size=256 \
  model=mlp \
  model.checkpoint="<PATH_TO_CHECKPOINT>" \
  seeds=[1]
```

---

### 5.4 Fast Smoke Testing
Run a quick CPU-based smoke test to verify execution paths:

```bash
# 1. Train 1 epoch on CPU
python main.py action=train_nids model=mlp gpu=cpu seeds=[1] model.training.epochs=1 dataset.params.path="data/tiny.parquet"

# 2. Evaluate 2 steps on CPU
python main.py action=evaluate_attack attack=pgd attack.steps=2 model=mlp gpu=cpu seeds=[1] dataset.params.path="data/tiny.parquet" model.checkpoint="<PATH_TO_GENERATED_NIDS_PT>"
```

---

## 6. Evaluation Metrics & Statistical Protocol

### Imbalance-Aware Defense Metrics
* **Matthews Correlation Coefficient (MCC):** Evaluates classification accuracy on imbalanced datasets.
* **False Alarm Rate (FAR) & Detection Rate (DR):** Measures false positive and true positive rates for attack detection.

### Attacker Performance Metrics
* **Attack Success Rate (ASR):** Percentage of correctly identified clean attack samples forced into the Benign class.
* **Realizability Rate (RR):** Fraction of generated adversarial samples satisfying all domain protocol constraints ($RR = \frac{N_{\text{valid}}}{N_{\text{total}}}$).
* **Detection AUROC:** Receiver Operating Characteristic area under the curve for inference-time detectors (`detection_auroc` and `detection_auroc_realizable`).

### Statistical Protocol
For evaluations across $N \ge 5$ seeds, the framework executes paired **Wilcoxon Signed-Rank Tests** comparing clean baseline metrics against adversarial results to produce statistical $p$-values.

---

## 7. Extending the Framework (Developer Guide)

Add new components using decorator registration without modifying core workflow code.

### Adding a New Attack
Create a file under `attacks/feature_space/my_attack.py`:

```python
from attacks.base import FeatureSpaceAttack, AttackResult
from core.registry import register_attack
import torch

@register_attack("my_custom_attack")
class MyCustomAttack(FeatureSpaceAttack):
    def generate(self, x, y, **kwargs) -> AttackResult:
        # Implementation logic...
        # Ensure intermediate autograd tensors are detached
        return AttackResult(adversarial=x_adv.detach())
```

### Adding a Custom Constraint
Add custom protocol validation rules to `constraints/rules.py`:

```python
from constraints.rules import ConstraintRule, RuleViolation
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class CustomWindowRule(ConstraintRule):
    name: str = "custom_window_check"
    def check(self, x: np.ndarray, feature_index: dict) -> RuleViolation | None:
        # Physical domain checks...
        return None
```

### Adding a New Inference-Time Detector
Create a detector plugin under `defenses/my_detector.py`:

```python
from defenses.base import BaseDetector
from core.registry import register_detector
from models.wrapper import ModelWrapper
from torch.utils.data import DataLoader
import numpy as np

@register_detector("my_custom_detector")
class CustomDetector(BaseDetector):
    def fit(self, dataloader: DataLoader, wrapper: ModelWrapper) -> "CustomDetector":
        # Extract features and fit statistics
        return self

    def score(self, x: np.ndarray, wrapper: ModelWrapper) -> np.ndarray:
        # Return anomaly scores (higher score = more anomalous)
        return np.zeros(len(x))
        
    def state_dict(self) -> dict:
        return {"fitted_params": ...}
        
    def load_state_dict(self, state: dict) -> None:
        pass
```

---

## 8. Credits & Acknowledgements

This framework builds upon concepts and foundational implementations from open-source research in adversarial machine learning:

* **Literature Reference:**
  * **Paper:** *Adversarial Robustness of Deep Learning-Based Intrusion Detection Systems: A Systematic Literature Review* - Winanto et al., IEEE Access 2026
* **TDTCE (Time-Conditioned Diffusion):**
  * **Paper:** *TDTCE: generating adversarial instances against NIDS via diffusion model* - Sui et al., The Computer Journal 2026
  * **Repository:** [https://github.com/HHLinlife/TDTCE](https://github.com/HHLinlife/TDTCE)
* **AMD-GAN**
  * **Paper:** *Adversarial red teaming and imbalance correction in heterogeneous NIDS: A class-specific adaptive GAN framework* - Lara et al., Journal of Information Security and Applications 2026
  * **Repository:** [https://github.com/antoniol00/AMD-GAN](https://github.com/antoniol00/AMD-GAN)
* **TRADES Adversarial Training:**
  * **Paper:** *Theoretically Principled Trade-off between Robustness and Accuracy* - Zhang et al., ICML 2019
  * **Repository:** [https://github.com/yaodongyu/TRADES](https://github.com/yaodongyu/TRADES)
* **Deep Mahalanobis Detector:**
  * **Paper:** *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks* - Lee et al., NeurIPS 2018
  * **Repository:** [https://github.com/pokaxpoka/deep_Mahalanobis_detector](https://github.com/pokaxpoka/deep_Mahalanobis_detector)