# NIDS Adversarial Benchmark Framework

A modular, extensible PyTorch benchmarking framework for evaluating state-of-the-art (SOTA) adversarial evasion attacks against machine learning and deep learning-based Network Intrusion Detection Systems (NIDS).

---

## Table of Contents
- [NIDS Adversarial Benchmark Framework](#nids-adversarial-benchmark-framework)
  - [Table of Contents](#table-of-contents)
  - [1. Motivation \& Design Philosophy](#1-motivation--design-philosophy)
    - [Framework goal](#framework-goal)
    - [Reading the Framework](#reading-the-framework)
  - [2. System Architecture](#2-system-architecture)
    - [2.1 End-to-End Pipeline Flow](#21-end-to-end-pipeline-flow)
    - [2.2 Core Subsystems](#22-core-subsystems)
      - [1. Configuration \& Run Isolation (`main.py`, `configs/`)](#1-configuration--run-isolation-mainpy-configs)
      - [2. Multi-Process Orchestrator (`orchestrator/`)](#2-multi-process-orchestrator-orchestrator)
      - [3. Dynamic Contract Adapter (`workflow.py`)](#3-dynamic-contract-adapter-workflowpy)
      - [4. Dual-Mode Task Engine: AD vs. AC (`data/`, `workflow.py`)](#4-dual-mode-task-engine-ad-vs-ac-data-workflowpy)
      - [5. Threat Models \& Attack Engines (`attacks/`)](#5-threat-models--attack-engines-attacks)
      - [6. Universal Constraint Gatekeeper (`constraints/`)](#6-universal-constraint-gatekeeper-constraints)
      - [7. Evaluation \& Statistical Metrics (`evaluator/`)](#7-evaluation--statistical-metrics-evaluator)
      - [8. Crash-Resilient Reporting Engine (`evaluator/logger.py`)](#8-crash-resilient-reporting-engine-evaluatorloggerpy)
    - [2.3 Framework Directory Layout](#23-framework-directory-layout)
  - [3. Installation \& Setup](#3-installation--setup)
  - [4. CLI \& Configuration Reference](#4-cli--configuration-reference)
    - [Core Global Parameters](#core-global-parameters)
    - [Task Modes (AD vs. AC)](#task-modes-ad-vs-ac)
    - [Component Configuration](#component-configuration)
    - [Defense \& Training Specific Flags (`model.training.*`)](#defense--training-specific-flags-modeltraining)
    - [Inference-Time Detector Flags (`detector.*`)](#inference-time-detector-flags-detector)
  - [5. Operational Workflows](#5-operational-workflows)
    - [5.1 Target NIDS Training (Clean \& Adversarial)](#51-target-nids-training-clean--adversarial)
      - [Standard Training (Clean Baseline)](#standard-training-clean-baseline)
      - [Adversarial Training (PGD-AT \& TRADES)](#adversarial-training-pgd-at--trades)
      - [Inference-Time Detector Fitting](#inference-time-detector-fitting)
    - [5.2 Attack Model Training (Generative / Surrogates)](#52-attack-model-training-generative--surrogates)
    - [5.3 Adversarial Evaluation \& Detection](#53-adversarial-evaluation--detection)
      - [Evaluating Evasion \& Enabling the Detector](#evaluating-evasion--enabling-the-detector)
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

### Framework goal
Mosto of the existing adversarial machine learning attack paper's results are not comparable due to the use of different dataset, different target NIDS and really often they have not been tested against any defence mechanism and / or use only one seed, that could bias results for that specific seed.

The aim of this framework is to simplify the evaluation of different evasion attacks allowing to easiliy swap them, the target nids, the dataset and the defence mechanism in order to obtain standardized results, averaging the score of different seeds run, and understand better what is working and what not.

### Reading the Framework

The framework has one parent process and one worker process per seed:

1. [`main.py`](./main.py) resolves Hydra configuration, creates the run directory,
   discovers plugins, and submits seed jobs.
2. [`orchestrator/runner.py`](./orchestrator/runner.py) starts isolated `spawn`
   workers and assigns each worker a seed and device.
3. [`workflow.py`](./workflow.py) contains the worker entry points:
   `train_nids_seed`, `train_attack_seed`, `evaluate_attack_seed`, and
   `preprocess_seed`.
4. A worker loads a dataset, applies the checkpoint's feature schema and
   normalization, constructs the configured model/attack, and writes seed-local
   artifacts.
5. The parent aggregates the returned numeric metrics after all workers finish.

Most components are plugins. A class is registered with a decorator in
[`core/registry.py`](./core/registry.py); `discover_all()` imports the plugin
packages before Hydra names such as `dataset=ciciomt2024` or
`attack=pgd` are resolved. This is why a new plugin normally needs only a
registered class and a matching configuration group.

 toolkits (e.g., ART, Foolbox) are derived from computer vision paradigms. When applied to network traffic, they can suffer from critical methodological flaws:

---

## 2. System Architecture

The framework is architected as an automated, multi-process benchmarking pipeline designed to isolate adversarial optimization, protocol constraint verification, and model evaluation into strictly decoupled stages. 

---

### 2.1 End-to-End Pipeline Flow

The execution graph keeps evaluation metrics and gatekeeper rejection rules out
of attack optimization. The evaluation batch's labels are passed to attacks
because targeted and untargeted objectives require them; however, the attack
does not receive the target model's post-attack score or the constraint
decision:

```text
[Hydra CLI / Config]
        │
        ▼
[main.py Orchestrator] ── (Resolves run_id & path once, dumps config.yaml)
        │
        ▼
[ProcessPoolExecutor (spawn)] ── Distributes Seeds & GPU Partitions
        │
        ├── Worker: Seed 1 (Device cuda:0) ── logs/seed_1.log
        ├── Worker: Seed 2 (Device cuda:1) ── logs/seed_2.log
        └── Worker: Seed N ...
                │
                ▼
        [Dynamic Tensor Shaping] ── Infers 2D flat vs. 3D (N, T, F) layout
                │
                ▼
        [Label Encoder (AD/AC)] ── Maps Benign -> 0; Attacks -> 1..K
                │
                ▼
        [Clean Baseline Inference] ── Establishes clean MCC, FAR, DR
                │
                ▼
        [Attack Optimization] ── Minimizes loss toward Target (Benign = 0)
          ├── Feature-Space: Continuous/discrete perturbation on normalized x
          └── Problem-Space: PCAP mutation -> SOTA Feature Extraction
                │
                ▼
        [Universal Constraint Gatekeeper] ── RFC & physical domain validation
          ├── Checks: Non-negativity, discrete integers, TCP flag coherence, bounds
          └── Computes: Realizability Rate (RR) & violation breakdown
                │
                ▼
        [Target Model Scoring] ── Strict mode filters rejected samples
                │
                ▼
        [Per-Seed Reporting] ── Saves .npz arrays, text reports, seed CMs
                │
                ▼
[Post-Run Aggregator (main.py)] ── Computes Mean ± Std, Wilcoxon p-values, 
                                   generates mean CM & multi-seed curves
```

---

### 2.2 Core Subsystems

#### 1. Configuration & Run Isolation (`main.py`, `configs/`)
* **Hydra I/O Decoupling:** Native Hydra directory generation and log capture are completely bypassed (`output_subdir: null`, `dir: .`). The framework manages file I/O directly in Python to eliminate empty log files and duplicate folders.
* **Deterministic Single-Evaluation Path:** A dynamic resolver assigns a unified path based on workflow, dataset, target model, attack variant, timestamp, and a short unique hash:
  `runs/<action>/<dataset>/<model>/<attack_variant>/<timestamp>_<hash>/`
  The path is resolved and cached once at launch, guaranteeing that spawned child workers, logs, checkpoints, and summary files share the exact same root directory.

#### 2. Multi-Process Orchestrator (`orchestrator/`)
* **Process Isolation:** Multi-seed sweeps are executed via `ProcessPoolExecutor` using Python's `spawn` context. Worker processes operate in complete isolation to prevent CUDA context collisions, GPU memory leaks, and global state contamination.
* **Device Partitioning (`orchestrator/device.py`):** Automatically distributes worker seeds across designated physical GPUs (`gpu_partitions: [cuda:0, cuda:1]`) using round-robin allocation.
* **Telemetry Decoupling (`orchestrator/runner.py`):** The parent process
  owns the single unified progress bar (`tqdm`). Each worker has a dedicated
  file logger and also emits normal log records to its console handler; the
  per-seed log file is the durable worker record.

#### 3. Dynamic Contract Adapter (`workflow.py`)
* **Automatic Tensor Reshaping:** Sequence datasets store flows as 3-D tensors `(N, T, F)` (Batches, Packets, Features). The framework dynamically inspects the chosen architecture:
  * Dense models (MLP, Random Forest, XGBoost) trigger automatic flattening to 2-D vectors `(N, T*F)`.
  * Recurrent/Spatial models (1D-CNN, LSTM, Transformer) preserve the structured 3-D sequence tensor `(N, T, F)`.
* **Surrogate-Target Contract Validation:** When surrogate-based black-box transfers are configured, the pipeline verifies that both target and surrogate models accept compatible input dimensions before executing optimizations.

#### 4. Dual-Mode Task Engine: AD vs. AC (`data/`, `workflow.py`)
* **Anomaly Detection (AD):** If binary numeric labels (`0`, `1`) are detected, the system operates as a classic anomaly detector where `0 = Benign` and `1 = Malicious`.
* **Attack Classification (AC):** If multi-class or string labels (e.g., `"Benign"`, `"DDoS"`, `"PortScan"`) are detected, the dataset manager deterministically maps the class matching `benign_string` to index `0`, while attack classes are mapped to indices `1..K`.
* **Strict Evasion Constraint:** In network security, confusing one attack class with another (e.g., DDoS misclassified as PortScan) is an attacker failure because the defensive alert still fires. The framework enforces this semantic rule:
  * Optimization targets the Benign class directly (`target_label = 0`).
  * Attack Success Rate (ASR) strictly counts samples where the adversarial prediction equals `0`.

#### 5. Threat Models & Attack Engines (`attacks/`)
* **Feature-Space Pipeline (`attacks/feature_space/`):** Implements
  gradient-driven (eg. PGD) and gradient-free (eg. Square Attack) perturbation
  generators operating on normalized feature vectors. The workflow supplies
  `target_label=0` when a configured attack supports it.
* **Problem-Space Pipeline (`attacks/problem_space/`):** Accepts raw packet captures (`.pcap`) or packet objects. Adversarial traffic is modified at the byte/packet level, then transformed back into model feature vectors using an in-memory instance of `BasePreprocessor` (`preprocessing/sota_extractor.py`).
* **Architectural Shielding:** The attack strategy receives the configured
  attack model: a surrogate for transfer attacks, or the target wrapper for
  white-box attacks. The target model's authoritative post-attack scoring is
  performed later by `EvaluationEngine`.
* **Defense Tooling:** Training-time PGD/TRADES and inference-time detectors
  such as Mahalanobis are separate stages. Detector fitting uses bounded
  batches and two streaming passes, so all penultimate features are not kept in
  memory at once.

#### 6. Universal Constraint Gatekeeper (`constraints/`)
* **Strict Boundary Placement:** Sits downstream of attack generation and upstream of target model evaluation. The gatekeeper is never exposed to the attack algorithm, preventing adversarial optimization from "gaming" constraint checks.
* **Denormalized Domain Checking:** Normalization statistics (mean, std) are
  inverted so all constraint rules evaluate real-world physical values:
  * Non-negativity checks (packet lengths, inter-arrival times $\ge 0$).
  * Discrete protocol integrity (TCP flags must evaluate to valid bitmask permutations).
  * Feature monotonicity and packet/byte ratio laws.
* **Telemetry Output:** Calculates the formal **Realizability Rate**
  ($RR = \frac{N_{\text{valid}}}{N_{\text{total}}}$) and logs granular
  rejection tallies per rule. When `strict_constraints: true`, invalid flows
  are removed from target-model scoring; when it is false, they remain in the
  scored batch while the rejection telemetry is still reported.

#### 7. Evaluation & Statistical Metrics (`evaluator/`)
* **Authoritative Evasion Gating:** Attack success is evaluated only on flows that the target model correctly classified in clean baseline conditions. Flows already misclassified before attack injection are pruned from ASR calculations.
* **Imbalance & Economic Scoring:** Generates Matthews Correlation Coefficient (MCC), Detection Rate (DR), False Alarm Rate (FAR), balanced accuracy, and Adversarial Expected Operating Cost (AdvEOC) parameterized by alert fatigue ($c_{FP}$) and breach impact ($c_{FN}$).
* **Significance Testing (`evaluator/statistical.py`):** For $N \ge 5$ seeds, executes paired non-parametric Wilcoxon signed-rank tests between clean and adversarial test vectors to output formal $p$-values.

#### 8. Crash-Resilient Reporting Engine (`evaluator/logger.py`)
* **Atomic Telemetry & Crash Hooks:** Uses a custom `FlushFileHandler` that calls `os.fsync()` after every write, alongside a `sys.excepthook` crash hook. If a server process experiences an Out-Of-Memory (OOM) error or termination signal, logs and learning curves are preserved up to the exact line of failure.
* **Raw Array Serialization:** Dumps ground truth labels, clean baseline predictions, adversarial predictions, and logits to compressed NumPy archives (`seed_<seed>_preds.npz`) for offline verification.
* **Confusion Matrices:** The current evaluation workflow writes
  `cm_seed_<seed>_adversarial.csv` and `.pdf` for each seed, plus
  `cm_mean.csv` and `.pdf` when aggregation has matrices to combine.
  Successful-evasion counts are available in the metrics; a separate
  successful-evasion matrix is not currently generated by the workflow.
* **Post-Run Multi-Seed Aggregation:** The main process compiles per-seed metric CSVs into `summary.json`, writes human-readable `report_aggregated.txt` summaries (Mean $\pm$ Std), and produces composite training curves (`epoch_loss.pdf`, `epoch_metrics.pdf`) displaying individual seed runs alongside bold cross-seed means.

---

### 2.3 Framework Directory Layout

```text
nids-adv-benchmark/
├── configs/                          # Hierarchical Hydra configurations
│   ├── config.yaml                   # Core runtime options and output directory rules
│   ├── attack/                       # Attack parameters (pgd.yaml, square.yaml, etc.)
│   ├── dataset/                      # Dataset parameters, paths, and label schemas
│   ├── model/                        # Model architectures and training hyperparameters
│   └── preprocessing/                # PCAP feature extractor configurations
│
├── core/
│   ├── exceptions.py                 # Custom domain exceptions (ConstraintViolationError, etc.)
│   └── registry.py                   # ComponentRegistry supporting decorator registration
│
├── data/
│   ├── base.py                       # BaseDataset interface and NormalizationStats dataclass
│   └── ciciomt2024.py                # SequenceParquetDataset with dynamic label encoder
│
├── defenses/
│   ├── base.py                       # BaseDetector interface (fit, score, state_dict)
│   └── detectors.py                  # Inference-time detectors (eg. Mahalanobis)
│
├── preprocessing/
│   ├── base.py                       # Preprocessor interface (fit/transform contracts)
│   └── sota_extractor.py             # PCAP-to-vector flow extraction engine
│
├── constraints/
│   ├── rules.py                      # Physical and RFC constraint definitions
│   └── gatekeeper.py                 # UniversalConstraintGatekeeper (Realizability referee)
│
├── models/
│   ├── architectures/                # PyTorch neural network backbones (MLP, CNN, LSTM, Transformer)
│   ├── trainer.py                    # Training loops with live atomic CSV curve updates
│   └── wrapper.py                    # ModelWrapper abstracting PyTorch and sklearn models
│
├── attacks/
│   ├── base.py                       # BaseAttack, FeatureSpaceAttack, ProblemSpaceAttack
│   ├── feature_space/                # Feature-space attacks (eg. PGD, Square)
│   └── problem_space/                # Problem-space PCAP mutators and flow injectors
│
├── evaluator/
│   ├── engine.py                     # EvaluationEngine coordinating target scoring
│   ├── logger.py                     # ExperimentReporter, FlushFileHandler, crash handlers
│   ├── metrics.py                    # Multi-class MCC, strict ASR, AdvEOC, and distances
│   ├── statistical.py                # Multi-seed cross-aggregation and Wilcoxon testing
│
├── orchestrator/
│   ├── device.py                     # CUDA pinning and device fallback management
│   └── runner.py                     # Multiprocessing seed worker executor using spawn
│
├── workflow.py                       # Worker target routines (train_nids, evaluate_attack, etc.)
└── main.py                           # CLI entrypoint, argument normalizer, and run aggregator
```

---

## 3. Installation & Setup

**Prerequisites:** Linux (Ubuntu 20.04+), Python 3.13, and the dependencies in
`requirements.txt`. CPU execution is supported; CUDA is optional and is used
when `gpu` or `gpu_partitions` selects a CUDA device.

```bash
# Clone the repository
git clone [https://github.com/Moriarty2002/nids-aes-generation.git]
cd nids-aes-generation

# Create and activate a clean virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Check that the entry point and configured plugins are available before starting
a long run:

```bash
python main.py --help
```

---

## 4. CLI & Configuration Reference
The framework uses **Hydra** for configuration. We completely bypass Hydra's default output directory creation to maintain a clean, collision-free workspace.

### Core Global Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `action` | `str` | *Required* | Workflow to execute: `preprocess`, `train_nids`, `train_attack`, `evaluate_attack`. |
| `gpu` | `str` | `cpu` | Primary device index (`0`, `cuda:0`, or `cpu`). |
| `seeds` | `list[int]` | `[1,2,3,4,5]`| Random seeds for deterministic execution. |
| `parallel_workers` | `int` | `5` | Maximum number of concurrent worker processes spawned. |
*(Note: Command line arguments can be passed without prefixes, e.g., `action=train_nids model=mlp`)*

### Task Modes (AD vs. AC)
The framework infers the task mode dynamically from your dataset configuration in `configs/dataset/`:
* **Anomaly Detection (Binary):** Point `label_column` to a column containing only `0` and `1`.
* **Attack Classification (Multi-Class):** Point `label_column` to a string-based column (e.g., `LABEL`). The framework will automatically assign `0` to the class matching `benign_string` (default: `"Benign"`) and enforce targeted evasions toward this class.

### Component Configuration

Hydra composes one configuration group for each major plugin type:

| Group | Selects | Important fields |
| --- | --- | --- |
| `dataset` | Dataset adapter | `name`, `params.path`, feature fields, label column, sequence padding |
| `model` | NIDS architecture | `name`, `checkpoint`, `params`, `training.*` |
| `attack` | Attack implementation | `name`, perturbation/query budget, targeted label, immutable features |
| `preprocessing` | Raw-traffic feature extraction | `name`, `input`, `output`, normalization parameters |
| `detector` | Inference-time detector | `enabled`, `name`, `params`, optional checkpoint |

The selected model and dataset determine the tensor contract. Sequence datasets
produce either `(N, T, F)` tensors for sequence models or `(N, T*F)` vectors
for flattening models such as the MLP. Checkpoints also store feature names and
normalization statistics; evaluation verifies that the loaded checkpoint schema
matches the current dataset before scoring.

### Defense & Training Specific Flags (`model.training.*`)
* `defense_mode` (`str`, default `"none"`): Selects the robust training method. Options include `"none"`, `"pgd"`, and `"trades"`. 
* `pgd_epsilon`, `pgd_alpha`, `pgd_steps`: Hyperparameters for the inner-loop adversarial generation during robust training.
* `trades_beta` (`float`, default `6.0`): The regularization trade-off parameter for TRADES.

### Inference-Time Detector Flags (`detector.*`)
* `detector.enabled` (`bool`, default `false`): Fits an inference-time detector after the classifier finishes training.
* `detector.name` (`str`, default `"mahalanobis"`): The registered detector plugin to use.


---

## 5. Operational Workflows

### 5.1 Target NIDS Training (Clean & Adversarial)
Trains a target detector across multiple seeds. The framework dynamically flattens the dataset if you select an MLP, and preserves `(N, T, F)` sequence dimensions if you select a Transformer.

#### Standard Training (Clean Baseline)
```bash
python main.py \
  action=train_nids \
  model=transformer \
  dataset=ciciot2023 \
  model.training.epochs=30 \
  model.training.batch_size=512 \
  seeds=[1,2,3,4,5] \
  gpu=0
```

#### Adversarial Training (PGD-AT & TRADES)
Proactively robustifies the model against evasion attacks by optimizing perturbations within the training loop. The framework supports standard PGD-AT (`defense_mode=pgd`) and **TRADES** (`defense_mode=trades`). TRADES balances standard accuracy and adversarial robustness using a KL-divergence regularizer ($\beta$). The internal optimization automatically pins the network to `.eval()` mode to prevent Batch Normalization statistic drift.

```bash
python main.py \
  action=train_nids \
  model=mlp \
  dataset=ciciot2023 \
  model.training.defense_mode=trades \
  model.training.pgd_epsilon=0.04 \
  model.training.pgd_steps=10 \
  model.training.trades_beta=6.0 \
  seeds=[1,2,3,4,5] \
  gpu=0
```
* **Output Artifacts (in `runs/train_nids/...`):**
  * `checkpoints/<model>_seed<seed>/nids.pt`: The trained model weights, feature schema, and normalization statistics.
  * `checkpoints/<model>_seed<seed>/training_curve.csv`: Live per-epoch metrics tracking `trn_loss`, `val_acc`, `val_f1`, and the active `defense_mode`.
  * `curves/epoch_loss.pdf` and `curves/epoch_metrics.pdf`: Aggregated
    multi-seed learning-curve plots when training-curve files exist.

#### Inference-Time Detector Fitting
Reactive defense mechanisms, such as the **Feature-Space Mahalanobis Detector**, are fitted *after* the NIDS finishes its standard or robust training. The detector does not require adversarial training itself; instead, it makes two streaming passes over the clean training data to estimate class-conditional Gaussian distributions of the NIDS's penultimate features without retaining the full representation matrix.

You can fit the detector automatically at the end of the `train_nids` workflow by enabling it in the configuration:

```bash
python main.py \
  action=train_nids \
  model=mlp \
  dataset=ciciot2023 \
  detector.enabled=true \
  detector.name=mahalanobis \
  seeds=[1,2,3,4,5] \
  gpu=0
```
* **Output Artifacts (in `runs/train_nids/...`):**
  * `checkpoints/<model>_seed<seed>/detector.pt`: The fitted state dictionary of the detector (containing the empirical class means, tied covariance matrices, and precision matrices).

---

### 5.2 Attack Model Training (Generative / Surrogates)
For attacks requiring pre-trained generative networks (e.g., GAN generator/discriminator) or substitute models, use `train_attack`:

```bash
python main.py \
  action=train_attack \
  attack=<registered_attack_with_train_method> \
  attack.params.<parameter>=<value> \
  seeds=[1,2,3,4,5] \
  gpu=0
```
* **Output Artifacts:** Plugin-specific checkpoints are written below the run
  directory. `train_attack` is a generic dispatch path: it succeeds only for
  an attack class that implements a compatible `train(...)` method. The
  currently configured AMD-GAN plugin is not part of the working smoke-tested
  path; PGD, Square, and the dummy problem-space mutator are generation
  attacks rather than attack-training workflows.

---

### 5.3 Adversarial Evaluation & Detection
Evaluates a trained model against an evasion attack. 
* To evaluate a **NIDS**, simply point `model.checkpoint` to the `nids.pt` generated during a TRADES/PGD training run. 
* If multi-class Attack Classification (AC) is detected, the attack automatically executes targeted optimization toward the Benign class.
  
(`evaluation.batch_size` bounds PGD autograd, target inference, and post-generation constraint validation memory. Reduce it for large sequence datasets or constrained CPU/RAM environments; it does not change sample ordering or the reported metrics.)

#### Evaluating Evasion & Enabling the Detector
To test the attack and simultaneously evaluate the inference-time detector's ability to flag the adversarial traffic, enable the detector during the `evaluate_attack` action. The framework will automatically locate and load the `detector.pt` state saved next to the target NIDS checkpoint.

```bash
python main.py \
  action=evaluate_attack \
  gpu=0 \
  attack=pgd \
  attack.epsilon=0.03 \
  attack.steps=20 \
  model=mlp \
  model.checkpoint="runs/train_nids/<dataset>/<model>/baseline/<TIMESTAMP>/checkpoints/mlp_seed{seed}/nids.pt" \
  dataset=ciciot2023 \
  evaluation.batch_size=256 \
  detector.enabled=true \
  detector.name=mahalanobis \
  seeds=[1,2,3,4,5]
```

* **Output Artifacts (in `runs/evaluate_attack/...`):**
  * `summary.json` & `seed_metrics.csv`: Tabular raw metrics and statistical aggregates, including `detection_auroc` (which scores the detector's ability to mathematically separate clean baseline traffic from adversarial traffic).
  * `reports/cm_seed_<seed>_adversarial.pdf`: Global normalized confusion matrix covering all evaluated traffic.
  * `reports/cm_seed_<seed>_adversarial.csv`: Raw confusion-matrix counts corresponding to the PDF.
  * `predictions/seed_<seed>_preds.npz`: Raw baseline and adversarial logits/predictions for offline analysis.
  * `per_seed/detector_scores/detector_scores_seed<seed>.npz`: The raw
    detector scores for clean and adversarial samples, plus the accepted mask
    and labels used to compute AUROC.

---

### 5.4 Fast Smoke Testing
To verify the framework without heavy data loads, generate a tiny dataset and limit optimization:
```bash
# 1. Train 1 epoch on CPU
python main.py action=train_nids model=mlp gpu=cpu seeds=[1] model.training.epochs=1 dataset.params.path="data/tiny.parquet"

# 2. Evaluate with 2 steps on CPU
python main.py action=evaluate_attack attack=pgd attack.steps=2 model=mlp gpu=cpu seeds=[1] dataset.params.path="data/tiny.parquet" model.checkpoint="<PATH_TO_GENERATED_PT_ABOVE>"
```

---

## 6. Evaluation Metrics & Statistical Protocol

### Imbalance-Aware Defense Metrics
* **Matthews Correlation Coefficient (MCC):** Correlation coefficient between prediction and ground truth.
* **False Alarm Rate (FAR) & Detection Rate (DR):** Benign flows misclassified as attacks, and true attack identification rate.

### Attacker Performance Metrics
* **Attack Success Rate (ASR):** Percentage of correctly-identified clean attacks that the attack successfully forced into the Benign class. (Misclassifying Attack A as Attack B yields an ASR of 0).
* **Realizability Rate (RR):** Fraction of generated adversarial examples that strictly satisfy all domain constraints.
* **Detection AUROC:** Area Under the Receiver Operating Characteristic curve for inference-time detectors (e.g., Mahalanobis). It evaluates the detector's ability to separate clean baseline traffic (negative class) from adversarial traffic (positive class) based on the generated anomaly scores. Evaluated globally and exclusively on realizable samples (`detection_auroc_realizable`).

### Statistical Protocol
The framework aggregates $N \ge 5$ independent evaluation seeds and executes a non-parametric **Paired Wilcoxon Signed-Rank Test** between clean baseline performance and attacked performance to compute exact $p$-values.

---

## 7. Extending the Framework (Developer Guide)

Adding new plugins requires no edits to central factories. Use decorators to register components automatically.

### Adding a New Attack
Create a file under `attacks/feature_space/my_attack.py`:
```python
from attacks.base import FeatureSpaceAttack, AttackResult
from core.registry import register_attack
import torch

@register_attack("my_gradient_attack")
class MyGradientAttack(FeatureSpaceAttack):
    def generate(self, x, y, **kwargs) -> AttackResult:
        # PGD implementations automatically receive target_label=0 in AC mode
        # Implementation here...
        return AttackResult(adversarial=x_adv.detach())
```

### Adding a Custom Constraint
Add a custom invariant rule into `constraints/rules.py`:
```python
from constraints.rules import ConstraintRule, RuleViolation
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class TTLBoundRule(ConstraintRule):
    name: str = "ttl_bounds"
    def check(self, x: np.ndarray, feature_index: dict) -> RuleViolation | None:
        # Domain invariant checks...
        return None
```

### Adding a New Inference-Time Detector
Create a file under `defenses/my_detector.py` and implement the `BaseDetector` interface:

```python
from defenses.base import BaseDetector
from core.registry import register_detector
from models.wrapper import PyTorchModelWrapper
from torch.utils.data import DataLoader
import numpy as np

@register_detector("my_custom_detector")
class CustomDetector(BaseDetector):
    def fit(self, dataloader: DataLoader, wrapper: PyTorchModelWrapper) -> "CustomDetector":
        # Extract features from the wrapper and fit empirical statistics
        return self

    def score(self, x: np.ndarray, wrapper: PyTorchModelWrapper) -> np.ndarray:
        # Return an array of anomaly scores (higher = more anomalous)
        return np.zeros(len(x))
        
    def state_dict(self) -> dict:
        return {"fitted_params": ...}
        
    def load_state_dict(self, state: dict) -> None:
        pass
```

## 8. Credits & Acknowledgements

This framework builds upon the foundational research and open-source implementations of several key papers in the field of adversarial machine learning. We extend our gratitude to the authors for making their code publicly available:

* **TRADES Adversarial Training:**
  * **Paper:** Theoretically Principled Trade-off between Robustness and Accuracy - Zhang et al., ICML 2019
  * **Original Repository:** [https://github.com/yaodongyu/TRADES](https://github.com/yaodongyu/TRADES)
  * *Adapted for the robust adversarial training loops within our PyTorch NIDS trainer.*

* **Deep Mahalanobis Detector:**
  * **Paper:** A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks - Lee et al., NeurIPS 2018
  * **Original Repository:** [https://github.com/pokaxpoka/deep_Mahalanobis_detector](https://github.com/pokaxpoka/deep_Mahalanobis_detector)
  * *Adapted for the reactive inference-time `MahalanobisDetector` plugin to evaluate detection AUROC on NIDS penultimate features.*

* **Recommended reading:**
  * **Paper:** Adversarial Robustness of Deep Learning-Based Intrusion Detection Systems: A Systematic Literature Review of Attacks, Defenses, Datasets, and Open Challenges - Winanto et al., IEEE Access 2026
  * **Original Repository:** [https://github.com/pokaxpoka/deep_Mahalanobis_detector](https://github.com/pokaxpoka/deep_Mahalanobis_detector)
  * *Adapted for the reactive inference-time `MahalanobisDetector` plugin to evaluate detection AUROC on NIDS penultimate features.*