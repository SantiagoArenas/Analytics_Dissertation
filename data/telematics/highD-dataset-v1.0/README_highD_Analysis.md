# highD Dataset Analysis & Synthetic Data Generation

## Project Overview

This documentation describes the complete pipeline for analyzing### 3.3 Velocity Analysis

![Velocity Analysi### 3.5 Lane Change Behavior

![Lane Change Analysis](lane_change_analysis.png)
*Figure 4: Lane change behavior analysis showing frequency and patterns by vehicle type.*

| Metric | Cars | Trucks |
|--------|------|--------|
| % with ≥1 lane change | 13.6% | 5.7% |
| Mean lane changes | 0.14 | 0.06 |
| Max lane changes | 3 | 2 |

**Key Insight**: Cars perform 2x more lane changes than trucks, indicating higher maneuver-related risk.

### 3.6 Trajectory Analysis

![Trajectory Analysis](trajectory_analysis.png)
*Figure 5: Vehicle trajectory visualizations showing spatial patterns and driving paths.*

---_analysis.png)
*Figure 2: Velocity distributions and speed profiles comparing cars and trucks.*

| Metric | Cars | Trucks |
|--------|------|--------|
| Mean Velocity | 111.8 km/h | 86.3 km/h |
| Std Dev | 20.6 km/h | 10.3 km/h |
| 95th Percentile Max | 146.3 km/h | 101.0 km/h |
| Speed Variability | 11.0 km/h | 6.9 km/h |

**Key Insight**: Cars drive approximately 30% faster than trucks and show more speed variability, indicating higher risk exposure.

### 3.4 Safety Metrics Analysis

![Safety Metrics](safety_metrics.png)
*Figure 3: Distribution of safety metrics including Time Headway (THW), Distance Headway (DHW), and Time to Collision (TTC).*

#### Time Headway (THW)High-Definition Highway Driving) dataset** and generating synthetic telematics data for training dynamic pricing algorithms in insurance applications.

**Author:** Santiago Arenas  
**Project:** Analytics Dissertation - Dynamic Pricing Algorithm Using Telematics and Sociological Data  
**Date:** December 2025

---

## Table of Contents

1. [Dataset Description](#1-dataset-description)
2. [Data Structure](#2-data-structure)
3. [Exploratory Data Analysis](#3-exploratory-data-analysis)
4. [Key Findings](#4-key-findings)
5. [Synthetic Data Generation](#5-synthetic-data-generation)
6. [Risk Scoring Methodology](#6-risk-scoring-methodology)
7. [Output Files](#7-output-files)
8. [Usage Guide](#8-usage-guide)

---

## 1. Dataset Description

### What is highD?

The **highD dataset** is a collection of naturalistic vehicle trajectory data recorded on German highways using drone-mounted cameras. It provides high-precision tracking data at 25 frames per second, making it ideal for studying driving behavior and developing telematics-based insurance models.

### Dataset Scale

| Metric | Value |
|--------|-------|
| Total Recordings | 60 drone recordings |
| Total Vehicles | 110,516 unique vehicles |
| Cars | 89,139 (80.7%) |
| Trucks | 21,377 (19.3%) |
| Total Distance Driven | 44,476 km |
| Total Recording Time | 16.7 hours |
| Frame Rate | 25 fps |
| Locations | 6 German highway sections |

### Data Collection Method

- **Drone-based recording**: Drones positioned above highways capture bird's-eye view
- **Computer vision processing**: Vehicles are detected and tracked frame-by-frame
- **High precision**: Position accuracy within centimeters
- **No privacy concerns**: No personally identifiable information collected

![Highway Recording Example](data/01_highway.png)
*Example of a highway recording location from the highD dataset.*

---

## 2. Data Structure

The highD dataset consists of three types of CSV files for each recording:

### 2.1 Tracks File (`XX_tracks.csv`)

Frame-by-frame vehicle tracking data with the following columns:

| Column | Description | Unit |
|--------|-------------|------|
| `frame` | Frame number | - |
| `id` | Unique vehicle ID | - |
| `x` | Longitudinal position | meters |
| `y` | Lateral position | meters |
| `width` | Vehicle width | meters |
| `height` | Vehicle length | meters |
| `xVelocity` | Longitudinal velocity | m/s |
| `yVelocity` | Lateral velocity | m/s |
| `xAcceleration` | Longitudinal acceleration | m/s² |
| `yAcceleration` | Lateral acceleration | m/s² |
| `frontSightDistance` | Distance to road end ahead | meters |
| `backSightDistance` | Distance to road end behind | meters |
| `dhw` | Distance headway to preceding vehicle | meters |
| `thw` | Time headway to preceding vehicle | seconds |
| `ttc` | Time to collision | seconds |
| `precedingId` | ID of preceding vehicle | - |
| `followingId` | ID of following vehicle | - |
| `laneId` | Current lane number | - |

### 2.2 Tracks Meta File (`XX_tracksMeta.csv`)

Per-vehicle summary statistics:

| Column | Description | Unit |
|--------|-------------|------|
| `id` | Unique vehicle ID | - |
| `width` | Vehicle width | meters |
| `height` | Vehicle length | meters |
| `initialFrame` | First frame of appearance | - |
| `finalFrame` | Last frame of appearance | - |
| `numFrames` | Total frames tracked | - |
| `class` | Vehicle type (Car/Truck) | - |
| `drivingDirection` | Direction (1 or 2) | - |
| `traveledDistance` | Total distance traveled | meters |
| `minXVelocity` | Minimum velocity | m/s |
| `maxXVelocity` | Maximum velocity | m/s |
| `meanXVelocity` | Average velocity | m/s |
| `minDHW` | Minimum distance headway | meters |
| `minTHW` | Minimum time headway | seconds |
| `minTTC` | Minimum time to collision | seconds |
| `numLaneChanges` | Number of lane changes | - |

### 2.3 Recording Meta File (`XX_recordingMeta.csv`)

Recording-level metadata:

| Column | Description |
|--------|-------------|
| `id` | Recording ID |
| `frameRate` | Frames per second (25) |
| `locationId` | Highway location identifier |
| `speedLimit` | Posted speed limit |
| `month` | Recording month |
| `weekDay` | Day of the week |
| `startTime` | Recording start time |
| `duration` | Recording duration (seconds) |
| `totalDrivenDistance` | Total km driven by all vehicles |
| `numVehicles` | Total vehicles in recording |
| `numCars` | Number of cars |
| `numTrucks` | Number of trucks |

---

## 3. Exploratory Data Analysis

### 3.1 Dataset Overview

![Dataset Overview](highD_overview.png)
*Figure 1: Overview of the highD dataset showing vehicle distribution, recording statistics, and basic data structure.*

### 3.2 Vehicle Class Distribution

The dataset contains primarily passenger cars:
- **Cars**: 80.7% (89,139 vehicles)
- **Trucks**: 19.3% (21,377 vehicles)

### 3.2 Velocity Analysis

| Metric | Cars | Trucks |
|--------|------|--------|
| Mean Velocity | 111.8 km/h | 86.3 km/h |
| Std Dev | 20.6 km/h | 10.3 km/h |
| 95th Percentile Max | 146.3 km/h | 101.0 km/h |
| Speed Variability | 11.0 km/h | 6.9 km/h |

**Key Insight**: Cars drive approximately 30% faster than trucks and show more speed variability, indicating higher risk exposure.

### 3.3 Safety Metrics Analysis

#### Time Headway (THW)
- **Median**: 1.34 seconds
- **Tailgating (<1s)**: 34.9% of vehicles
- Safe threshold: >1.0 second

#### Distance Headway (DHW)
- **Median**: 38.4 meters
- **Dangerous (<20m)**: 19.4% of vehicles

#### Time to Collision (TTC)
- **Critical (<3s)**: 1.3% of vehicles
- This metric indicates imminent collision risk

### 3.4 Lane Change Behavior

| Metric | Cars | Trucks |
|--------|------|--------|
| % with ≥1 lane change | 13.6% | 5.7% |
| Mean lane changes | 0.14 | 0.06 |
| Max lane changes | 3 | 2 |

**Key Insight**: Cars perform 2x more lane changes than trucks, indicating higher maneuver-related risk.

---

## 4. Key Findings

### 4.1 Risk Indicators Identified

1. **Speed Behavior**
   - Higher mean velocity → Higher crash severity potential
   - Higher speed variability → Aggressive/inconsistent driving

2. **Following Behavior**
   - Lower THW → Tailgating tendency (direct crash risk)
   - Lower DHW → Reduced reaction time

3. **Maneuvering Patterns**
   - More lane changes → Higher exposure to side collisions
   - Combined with speed variability → Aggressive driving profile

### 4.2 Correlations Discovered

![Correlation Analysis](correlation_analysis.png)
*Figure 6: Feature correlation heatmap showing relationships between driving behavior metrics.*

- **Speed ↔ THW**: Faster drivers tend to follow closer (inverse correlation)
- **Speed variability ↔ Lane changes**: More variable drivers change lanes more often
- **Vehicle class effects**: Clear behavioral differences between cars and trucks

### 4.3 Implications for Dynamic Pricing

1. **Risk Segmentation**: Data supports creating distinct risk tiers
2. **Feature Selection**: THW and speed variability are strong predictors
3. **Vehicle Type**: Should be a base rate modifier in pricing models

---

## 5. Synthetic Data Generation

### 5.1 Why Synthetic Data?

| Reason | Benefit |
|--------|---------|
| **Privacy preservation** | No actual driver data in training set |
| **Avoid overfitting** | Model learns patterns, not specific recordings |
| **Data augmentation** | Generate unlimited training samples |
| **Controllable scenarios** | Create specific risk profiles for testing |

### 5.2 Generation Methodology

#### Step 1: Distribution Fitting

We analyzed the statistical distributions of key features in the real data:

| Feature | Cars Distribution | Trucks Distribution |
|---------|-------------------|---------------------|
| Mean Velocity | Normal (μ=111.8 km/h) | Normal (μ=86.3 km/h) |
| Min THW | Lognormal (μ=1.81s) | Lognormal (μ=2.91s) |
| Min DHW | Lognormal (μ=57.2m) | Lognormal (μ=69.4m) |
| Lane Changes | Poisson-like | Poisson-like |

#### Step 2: Correlation Preservation

The generator preserves realistic correlations between features:
- Speed and following distance
- Speed variability and lane changes
- Vehicle type and all behavioral metrics

#### Step 3: Synthetic Profile Generation

For each synthetic driver profile:

```python
1. Sample vehicle type (Car: 80.7%, Truck: 19.3%)
2. Generate base velocity from fitted distribution
3. Add correlated noise for min/max velocity
4. Generate THW inversely correlated with speed
5. Generate DHW correlated with THW
6. Generate lane changes based on speed variability
7. Calculate derived risk indicators
```

### 5.3 Validation

![Synthetic Validation](synthetic_validation.png)
*Figure 7: Comparison of real vs synthetic data distributions to validate the generation process.*

The synthetic data was validated against real data:

| Metric | Real | Synthetic | Difference |
|--------|------|-----------|------------|
| Mean Velocity (Cars) | 111.8 km/h | 112.0 km/h | 0.1% |
| Mean Velocity (Trucks) | 86.3 km/h | 86.3 km/h | 0.0% |
| Mean THW | 2.02s | 2.24s | 10.8% |
| Mean DHW | 59.6m | 71.0m | 19.2% |

---

## 6. Risk Scoring Methodology

![Risk Scoring](risk_scoring.png)
*Figure 8: Risk score distributions and premium calculations based on driving behavior.*

### 6.1 Composite Risk Score

The risk score (0-100) is calculated using the following components:

| Component | Weight | Scoring Logic |
|-----------|--------|---------------|
| **Speed Risk** | 30 pts max | >130 km/h: 30pts, >120: 20pts, >110: 10pts |
| **Speed Variability** | 20 pts max | >30 km/h range: 20pts, >20: 15pts, >10: 5pts |
| **Tailgating (THW)** | 25 pts max | <0.5s: 25pts, <1.0s: 20pts, <1.5s: 10pts |
| **Following Distance** | 15 pts max | <15m: 15pts, <25m: 10pts, <40m: 5pts |
| **Lane Changes** | 10 pts max | 3 pts per lane change (max 10) |

### 6.2 Risk Tiers

![Synthetic Risk Distribution](synthetic_risk_distribution.png)
*Figure 9: Distribution of risk tiers in the synthetic dataset.*

| Tier | Score Range | % of Drivers |
|------|-------------|--------------|
| Very Low | 0-20 | 34.8% |
| Low | 20-40 | 35.4% |
| Medium | 40-60 | 21.5% |
| High | 60-80 | 4.8% |
| Very High | 80-100 | 0.3% |

### 6.3 Premium Calculation

```python
premium_multiplier = 0.7 + (risk_score / 100) * 1.3
final_premium = base_premium * premium_multiplier
```

| Risk Level | Multiplier Range | Premium Range (Base: $1,000) |
|------------|------------------|------------------------------|
| Safest | 0.70x | $700 |
| Average | 1.00x | $1,000 |
| Riskiest | 2.00x | $2,000 |

---

## 7. Output Files

### 7.1 File Locations

```
data/telematics/highD-dataset-v1.0/
├── data/
│   ├── synthetic/
│   │   ├── synthetic_telematics_50k.csv    # Full dataset
│   │   ├── synthetic_telematics_50k.pkl    # Pickle format
│   │   ├── synthetic_train.csv             # Training set (80%)
│   │   ├── synthetic_test.csv              # Test set (20%)
│   │   └── synthetic_generator.pkl         # Generator object
│   └── [original highD data files]
├── src/
│   └── highD_main.ipynb                    # Analysis notebook
└── README_highD_Analysis.md                # This documentation
```

### 7.2 Synthetic Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `vehicle_type` | string | "Car" or "Truck" |
| `avg_speed_kmh` | float | Average speed in km/h |
| `min_speed_kmh` | float | Minimum speed observed |
| `max_speed_kmh` | float | Maximum speed observed |
| `speed_variability_kmh` | float | Max - Min speed |
| `min_following_distance_m` | float | Minimum DHW in meters |
| `min_time_headway_s` | float | Minimum THW in seconds |
| `min_time_to_collision_s` | float | Minimum TTC in seconds |
| `lane_change_count` | int | Number of lane changes |
| `trip_distance_m` | float | Total distance traveled |
| `trip_duration_s` | float | Observation time |
| `tailgating_indicator` | int | 1 if THW < 1s, else 0 |
| `speeding_indicator` | int | 1 if speed > 130 km/h, else 0 |
| `aggressive_indicator` | int | 1 if aggressive behavior detected |
| `risk_score` | float | Composite risk score (0-100) |
| `risk_tier` | string | Risk category |
| `premium_multiplier` | float | Price modifier (0.7-2.0) |
| `final_premium` | float | Calculated annual premium |

### 7.3 Visualization Outputs

The following PNG files are generated:

| File | Description |
|------|-------------|
| `highD_overview.png` | Dataset overview and vehicle distribution |
| `velocity_analysis.png` | Speed profiles by vehicle class |
| `safety_metrics.png` | THW, DHW, TTC distributions |
| `lane_change_analysis.png` | Lane change behavior analysis |
| `trajectory_analysis.png` | Vehicle trajectory visualizations |
| `risk_scoring.png` | Risk score and premium distributions |
| `correlation_analysis.png` | Feature correlation heatmap |
| `synthetic_validation.png` | Real vs synthetic comparison |
| `synthetic_risk_distribution.png` | Synthetic data risk distribution |

---

## 8. Usage Guide

### 8.1 Loading the Synthetic Data

```python
import pandas as pd

# Load training and test sets
train_df = pd.read_csv('data/synthetic/synthetic_train.csv')
test_df = pd.read_csv('data/synthetic/synthetic_test.csv')

print(f"Training samples: {len(train_df):,}")
print(f"Test samples: {len(test_df):,}")
```

### 8.2 Preparing Features for ML

```python
# Define feature columns
feature_cols = [
    'vehicle_type',
    'avg_speed_kmh',
    'speed_variability_kmh',
    'min_following_distance_m',
    'min_time_headway_s',
    'lane_change_count',
    'tailgating_indicator',
    'speeding_indicator',
    'aggressive_indicator'
]

# Target variable options
# - 'risk_score': Regression (0-100)
# - 'risk_tier': Classification (5 categories)
# - 'final_premium': Regression ($700-$2000)

X_train = train_df[feature_cols]
y_train = train_df['risk_score']  # or 'final_premium'
```

### 8.3 Training a Model

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
le = LabelEncoder()
X_train['vehicle_type'] = le.fit_transform(X_train['vehicle_type'])

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
X_test = test_df[feature_cols]
X_test['vehicle_type'] = le.transform(X_test['vehicle_type'])
y_test = test_df['risk_score']

predictions = model.predict(X_test)
print(f"RMSE: {mean_squared_error(y_test, predictions, squared=False):.2f}")
print(f"R²: {r2_score(y_test, predictions):.3f}")
```

### 8.4 Generating More Synthetic Data

```python
import pickle

# Load the generator
with open('data/synthetic/synthetic_generator.pkl', 'rb') as f:
    generator = pickle.load(f)

# Generate new data
new_synthetic_df = generator.generate_synthetic_dataset(
    n_samples=100000,  # Generate 100k samples
    car_ratio=0.8      # 80% cars, 20% trucks
)
```

### 8.5 Combining with Sociological Data

To integrate sociological variables for your dissertation:

```python
# Example: Add demographic features
synthetic_df['driver_age'] = np.random.normal(40, 15, len(synthetic_df)).clip(18, 80)
synthetic_df['years_licensed'] = (synthetic_df['driver_age'] - 18).clip(0, None)
synthetic_df['urban_rural'] = np.random.choice(['Urban', 'Suburban', 'Rural'], len(synthetic_df))

# Adjust risk scores based on demographics
# (Implement your own logic based on actuarial research)
```

---

## References

1. **highD Dataset**: Krajewski, R., et al. "The highD Dataset: A Drone Dataset of Naturalistic Vehicle Trajectories on German Highways for Validation of Highly Automated Driving Systems." IEEE ITSC 2018.

2. **Usage-Based Insurance**: Husnjak, S., et al. "Telematics System in Usage Based Motor Insurance." Procedia Engineering, 2015.

3. **Dynamic Pricing in Insurance**: Denuit, M., et al. "Effective Statistical Learning Methods for Actuaries." Springer, 2019.

---

## Contact

For questions about this analysis or the synthetic data generation process, please contact:

**Santiago Arenas**  
Analytics Dissertation Project  
December 2025

---

*This documentation was generated as part of the dissertation project on dynamic pricing algorithms using telematics and sociological data.*
