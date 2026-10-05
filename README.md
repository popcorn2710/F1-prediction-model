# 🏎️ F1 Singapore Grand Prix Prediction System

A probabilistic machine-learning system designed to predict the outcome of the **Formula 1 Singapore Grand Prix**.

The system combines historical Formula 1 data, qualifying performance, driver and constructor performance, circuit characteristics, recent form, reliability, weather conditions, and other pre-race information to estimate the probability of different race outcomes.

Rather than simply predicting:

> **"Driver X will win."**

the system attempts to answer:

> **"What is the probability of every driver finishing in each position, and how confident are we in those predictions?"**

The final prediction is generated through a combination of **machine learning, statistical modeling, ranking, and Monte Carlo race simulation**.

---

# 🎯 Objective

The objective is to generate a **pre-race probabilistic forecast** for the Singapore Grand Prix.

The system should estimate:

- 🥇 Probability of winning
- 🥈 Probability of finishing P2
- 🥉 Probability of finishing P3
- 🏆 Podium probability
- 🔝 Top-5 probability
- 🔝 Top-10 probability
- 📍 Expected finishing position
- 💥 DNF probability
- 🏁 Expected race performance
- 📊 Probability distribution across all finishing positions

For example:

| Driver | Win | Podium | Top 5 | Top 10 | DNF | Expected Pos |
|---|---:|---:|---:|---:|---:|---:|
| Driver A | 42% | 76% | 91% | 97% | 4% | 2.3 |
| Driver B | 28% | 64% | 87% | 95% | 6% | 3.4 |
| Driver C | 16% | 49% | 73% | 90% | 8% | 5.1 |

These numbers are produced from the model and simulation rather than manually assigned.

---

# 🏁 Why Singapore Is Different

The Singapore Grand Prix is not simply another circuit in the dataset.

The **Marina Bay Street Circuit** has characteristics that can significantly affect race outcomes.

Important characteristics include:

- Street-circuit layout
- High physical demand
- High temperatures and humidity
- Significant tire management
- Long race duration
- Limited opportunities for overtaking
- Qualifying performance importance
- Safety-car potential
- Track evolution
- Strategy sensitivity

Therefore, the model should not treat every circuit equally.

The Singapore prediction should give appropriate weight to:

```text
Qualifying performance
        +
Street-circuit performance
        +
Tire degradation
        +
Driver performance
        +
Car performance
        +
Reliability
        +
Weather
        +
Strategy
        ↓
Singapore GP prediction
```

---

# 🧠 Core Prediction Philosophy

The system does **not** directly ask:

> "Who will finish first?"

Instead, it estimates the underlying factors that influence the final classification.

```text
Pre-Race Information
        ↓
Driver Performance
        ↓
Car Performance
        ↓
Qualifying / Grid
        ↓
Race Pace
        ↓
Tire / Strategy Effects
        ↓
Reliability / DNF
        ↓
Race Outcome Distribution
        ↓
Final Classification
```

This makes the system probabilistic rather than deterministic.

---

# 📊 Data Used

The model should only use information that would have been available **before the Singapore Grand Prix begins**.

Potential data includes:

### Driver

- Recent finishing positions
- Recent qualifying performance
- Career performance
- Season performance
- Singapore GP history
- Street-circuit performance
- DNF history
- Average race pace

### Constructor / Car

- Recent team performance
- Qualifying pace
- Race pace
- Reliability
- Tire degradation
- Performance on similar circuits
- Season performance

### Singapore Circuit

- Historical results
- Qualifying importance
- Overtaking characteristics
- Safety-car history
- Tire degradation
- Track temperature
- Circuit layout characteristics

### Race Weekend

- Practice performance
- Qualifying results
- Starting grid
- Weather forecast
- Track temperature
- Tire information

---

# ⚠️ No Future Information

The model must not use information that becomes available after the prediction timestamp.

For example:

If the prediction is generated **before qualifying**, qualifying results cannot be used.

If the prediction is generated **after qualifying but before the race**, qualifying and grid position can be used.

The prediction timestamp must therefore be explicitly defined.

```text
                Prediction Point
                       │
          ┌────────────┴────────────┐
          │                         │
     Information                  Future
       BEFORE                    Information
          │                         │
          ↓                         ↓
       Allowed                  Forbidden
```

This prevents **data leakage**.

---

# 🧮 Feature Engineering

Raw race statistics are transformed into predictive features.

## Recent Form

The model can calculate:

```text
Last 3 race average
Last 5 race average
Last 10 race average
Season average
```

Recent races can be weighted more heavily than older races.

---

# 🏎️ Driver Performance

Driver performance should be represented independently from the car where possible.

Potential features:

```text
Qualifying skill
Race pace
Overtake performance
Consistency
Recent form
Street-circuit performance
Singapore history
DNF rate
```

---

# 🚗 Constructor Performance

The model estimates current car/team strength using information available before the Singapore GP.

Examples:

```text
Average qualifying gap
Average race pace
Recent points
Recent podium rate
Reliability
Tire degradation
Circuit suitability
```

---

# 🌡️ Weather

Weather is particularly important for a Singapore prediction.

The system can incorporate:

```text
Air temperature
Track temperature
Humidity
Rain probability
Expected rainfall
Wind
Weather uncertainty
```

Importantly, the system should use the **forecast available at prediction time**, rather than actual future race conditions.

---

# 💥 DNF Model

A separate model estimates the probability that each driver will fail to finish.

Example:

```text
Driver A → 4% DNF
Driver B → 7% DNF
Driver C → 11% DNF
```

This is important because a driver can have excellent pace but still fail to finish.

The DNF model can consider:

- Historical reliability
- Driver incidents
- Constructor reliability
- Circuit characteristics
- Weather
- Starting position
- Recent mechanical issues

---

# ⏱️ Race Pace Model

A race-performance model estimates the expected race time or relative race pace for each driver.

Example:

```text
Driver A → 1:42:18
Driver B → 1:42:35
Driver C → 1:42:47
```

These predictions are conditional on the driver successfully completing the race.

---

# 🏆 Ranking Model

Because Formula 1 produces a **complete ranking**, the project can use learning-to-rank techniques rather than treating finishing position as a simple continuous variable.

Possible approaches include:

- LambdaMART
- LightGBM ranking
- XGBoost ranking
- Bradley-Terry
- Plackett-Luce

The model can estimate relationships such as:

```text
P(Driver A finishes ahead of Driver B)
```

rather than directly forcing:

```text
Driver A → 2.43rd position
```

This better represents the structure of a racing competition.

---

# 🎲 Monte Carlo Race Simulation

The final race prediction is generated by simulating the Singapore Grand Prix many times.

For example:

```text
Simulation 1
Verstappen → P1
Norris     → P2
Leclerc    → P3

Simulation 2
Norris     → P1
Verstappen → P2
Leclerc    → P3

Simulation 3
Leclerc    → P1
Norris     → P2
Verstappen → P4

...

Simulation 10,000
```

After thousands of simulations, the system calculates the probability distribution.

Example:

```text
                         WIN
Verstappen              41.7%
Norris                  29.4%
Leclerc                 16.2%
Piastri                  8.1%
Others                   4.6%
```

---

# 📈 Final Singapore GP Output

The dashboard should display something similar to:

```text
╔════════════════════════════════════════════╗
║       SINGAPORE GRAND PRIX PREDICTION     ║
╠══════════════╦══════╦════════╦═════════════╣
║ Driver       ║ Win  ║ Podium ║ Expected P  ║
╠══════════════╬══════╬════════╬═════════════╣
║ Driver A     ║ 42%  ║ 78%    ║ 2.1         ║
║ Driver B     ║ 29%  ║ 67%    ║ 2.8         ║
║ Driver C     ║ 15%  ║ 46%    ║ 4.6         ║
║ Driver D     ║  7%  ║ 31%    ║ 6.2         ║
╚══════════════╩══════╩════════╩═════════════╝
```

The actual driver names will be populated from the race dataset.

---

# 📊 Position Probability Matrix

One of the main outputs should be a full probability matrix.

| Driver | P1 | P2 | P3 | P4 | P5 | P6+ |
|---|---:|---:|---:|---:|---:|---:|
| Driver A | 42% | 19% | 12% | 9% | 7% | 11% |
| Driver B | 29% | 27% | 18% | 12% | 7% | 7% |
| Driver C | 15% | 22% | 24% | 15% | 10% | 14% |

This allows users to understand **uncertainty**, not just the predicted order.

---

# 🧪 Backtesting

The model will be evaluated against historical races.

The system should simulate the situation as if the model were actually making predictions before each historical race.

Example:

```text
Historical Race
       ↓
Pretend we are before the race
       ↓
Use only available information
       ↓
Generate prediction
       ↓
Compare with actual result
```

This is repeated across multiple seasons and races.

---

# ⏳ Walk-Forward Validation

Randomly splitting F1 races into train/test sets can produce misleading results.

Instead:

```text
Train → 2018–2021
Test  → 2022

Train → 2018–2022
Test  → 2023

Train → 2018–2023
Test  → 2024
```

This reproduces the real-world prediction scenario.

---

# 📏 Evaluation Metrics

The system should evaluate multiple dimensions.

### Race Time

- MAE
- RMSE

### Ranking

- Spearman correlation
- Kendall's Tau
- NDCG
- Pairwise ranking accuracy

### DNF

- ROC-AUC
- PR-AUC
- F1
- Log Loss

### Probability

- Brier Score
- Log Loss
- Calibration Error

A prediction should not be considered successful merely because it correctly predicted the winner.

The probabilities must also be **well calibrated**.

---

# 🔬 Explainability

The system should explain why it believes a driver has a particular probability.

Example:

```text
Driver A — Win Probability: 42%

Positive Factors
────────────────────────────
Qualifying pace       +14%
Recent race pace      +11%
Car performance        +9%
Singapore history     +5%
Grid position          +4%

Negative Factors
────────────────────────────
Tire degradation       -3%
DNF risk               -2%
Weather uncertainty    -1%
```

SHAP can be used to provide model-level and individual prediction explanations.

---

# 🖥️ Dashboard

The final application can contain several sections.

## 1. Race Overview

```text
🇸🇬 Singapore Grand Prix

Circuit: Marina Bay
Race Date: ...
Weather: ...
Track Temperature: ...
```

## 2. Predicted Grid / Classification

Display the expected finishing order.

## 3. Win Probability

Interactive probability chart.

## 4. Position Distribution

Show the probability of each driver finishing P1–P20.

## 5. Podium Probability

Compare all drivers.

## 6. DNF Probability

Show reliability risk.

## 7. Feature Importance

Explain the factors influencing the prediction.

## 8. Monte Carlo Simulation

Allow users to run:

```text
1,000
5,000
10,000
50,000
```

simulations.

---

# 🏗️ Project Architecture

```text
racepredict/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── feature_engineering.ipynb
│   ├── baseline_models.ipynb
│   ├── ranking_model.ipynb
│   ├── simulation.ipynb
│   └── evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── cleaning.py
│   │   └── validation.py
│   │
│   ├── features/
│   │   ├── driver.py
│   │   ├── constructor.py
│   │   ├── circuit.py
│   │   ├── weather.py
│   │   └── pipeline.py
│   │
│   ├── models/
│   │   ├── pace_model.py
│   │   ├── dnf_model.py
│   │   ├── ranking_model.py
│   │   └── calibration.py
│   │
│   ├── simulation/
│   │   ├── race_simulator.py
│   │   └── monte_carlo.py
│   │
│   └── evaluation/
│       ├── metrics.py
│       └── backtesting.py
│
├── models/
│   ├── pace/
│   ├── dnf/
│   └── ranking/
│
├── api/
│   └── main.py
│
├── dashboard/
│
├── tests/
│
├── requirements.txt
├── config.yaml
└── README.md
```

---

# 🛠️ Technology Stack

### Data

- Python
- Pandas
- NumPy
- PostgreSQL

### Machine Learning

- Scikit-learn
- XGBoost
- LightGBM
- CatBoost

### Statistical Modeling

- SciPy
- Statsmodels
- PyMC

### Explainability

- SHAP

### Visualization

- Plotly
- Matplotlib

### Backend

- FastAPI

### Frontend

- React
- TypeScript
- Tailwind CSS

### Deployment

- Docker
- GitHub Actions

---

# 🚀 Development Plan

### Phase 1 — Historical F1 Dataset

Collect and normalize:

- Race results
- Qualifying
- Practice
- Drivers
- Constructors
- Circuits
- Lap times
- Pit stops
- Reliability
- Weather

### Phase 2 — Singapore-Specific Features

Build:

- Singapore historical performance
- Street-circuit performance
- Tire behavior
- Qualifying importance
- Safety-car statistics
- Weather features

### Phase 3 — Baseline

Implement:

```text
Historical baseline
        ↓
Linear Regression
        ↓
Random Forest
        ↓
XGBoost / LightGBM
```

### Phase 4 — Ranking

Implement learning-to-rank models and compare them against the baseline.

### Phase 5 — DNF

Develop the independent DNF probability model.

### Phase 6 — Monte Carlo

Build the race simulation engine.

### Phase 7 — Backtesting

Run the complete system against historical F1 races.

### Phase 8 — Singapore GP Prediction

Generate the final Singapore Grand Prix probability distribution using only information available at the defined prediction timestamp.

### Phase 9 — Dashboard

Expose the prediction through an interactive web application.

---

# ⚠️ Limitations

The model cannot perfectly predict an F1 race.

Unexpected events can dramatically change the outcome:

- Crashes
- Safety Cars
- Red flags
- Mechanical failures
- Penalties
- Strategy errors
- Unexpected weather
- Tire failures
- Driver mistakes

Therefore, the system should **never present predictions as certainties**.

The correct interpretation is:

> **Probability, not certainty.**

---

# 🏁 Final Goal

The final system should not simply output:

```text
🏆 Driver A will win.
```

It should output something closer to:

```text
🇸🇬 SINGAPORE GRAND PRIX

Driver A
Win:       42%
Podium:    78%
Top 5:     93%
Expected:  2.1
DNF:        4%

Driver B
Win:       29%
Podium:    67%
Top 5:     88%
Expected:  3.2
DNF:        6%

...

Most likely winner:
Driver A — 42%

Most likely podium:
Driver A
Driver B
Driver C
```

The prediction should be backed by **historical evidence, machine learning, probabilistic modeling, and Monte Carlo simulation**, with strict temporal validation to ensure that no future information leaks into the model.
