# EV Range Prediction Using Linear Regression

A machine learning project that predicts an electric vehicle's remaining
driving range from battery, vehicle, driving, and environmental
measurements. The project implements **linear regression with gradient
descent from scratch**, including feature standardization, loss
functions, gradient calculations, and L2 regularization.

## Project Overview

Estimating remaining range can help explain how factors such as battery
state, power consumption, speed, road gradient, and weather relate to an
EV's available driving distance. This notebook builds a baseline
regression model using NumPy and evaluates its predictions on a held-out
dataset.

## Dataset

File: `EV_Range_Prediction_Dataset_2000.csv`

-   **Rows:** 2,000
-   **Columns:** 17
-   **Target:** `remaining_range`
-   **Missing values:** None were found during the notebook's initial
    inspection.

### Columns

  Column                Description
  --------------------- ----------------------------------
  `timestamp`           Timestamp of the observation
  `battery_soc`         Battery state of charge
  `voltage`             Battery voltage
  `current`             Electrical current
  `power_consumption`   Power consumption measurement
  `speed`               Vehicle speed
  `distance_traveled`   Distance traveled measurement
  `road_gradient`       Road gradient
  `traffic_density`     Traffic category
  `temperature`         Ambient temperature
  `humidity`            Humidity
  `wind_speed`          Wind speed
  `driving_style`       Driving-style category
  `vehicle_load`        Vehicle load
  `battery_health`      Battery-health measurement
  `regen_braking`       Regenerative-braking measurement
  `remaining_range`     **Prediction target**

The notebook currently excludes `timestamp`, `traffic_density`, and
`driving_style` from the model input. The two categorical columns are
not encoded in this version; the model uses the 13 remaining numeric
predictor columns.

> **Dataset note:** The repository dataset is used as provided. Confirm
> its source, units, and whether it represents real-world or synthetic
> measurements before using the model for operational decisions.

## Workflow

1.  **Load and inspect data** with pandas; check shape, columns, and
    missing values.
2.  **Separate features and target** using `remaining_range` as the
    target.
3.  **Split the data** into training, validation, and test subsets with
    `random_state=42`:
    -   Training: 1,280 rows
    -   Validation: 320 rows
    -   Test: 400 rows
4.  **Standardize features** using the training-set mean and standard
    deviation, then apply those same values to validation and test data.
5.  **Add a bias term** to model the intercept.
6.  **Implement regression utilities from scratch**, including
    prediction, MSE, MAE, RMSE, NRMSE, and L1/L2 regularized objectives.
7.  **Implement gradients** for MSE, MAE, RMSE, and L1/L2 regularized
    objectives.
8.  **Train with gradient descent** using an L2-regularized objective.
9.  **Visualize loss convergence** and compare predicted and actual
    remaining range.
10. **Inspect prediction errors and learned weights**.

## Model and Implementation

-   **Language:** Python
-   **Libraries:** NumPy, pandas, Matplotlib, scikit-learn
    (`train_test_split`)
-   **Model:** Multiple linear regression
-   **Optimization:** Batch gradient descent
-   **Regularization:** L2 (ridge-style penalty implemented in the
    objective)
-   **Feature scaling:** Custom z-score standardization
-   **Initial hyperparameters in the notebook:**
    -   Learning rate: `0.01`
    -   Maximum iterations: `1000`
    -   L2 regularization coefficient: `0.01`
    -   Gradient-norm tolerance: `1e-6`

The notebook defines L1-related functions for learning and
experimentation, but the training routine shown uses the L2-regularized
gradient.

## Getting Started

### 1. Clone the repository

``` bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Install dependencies

``` bash
python -m pip install numpy pandas matplotlib scikit-learn jupyter
```

### 3. Keep the files together

Place these files in the same project folder:

``` text
.
├── Linear_reg (ev_range ).ipynb
├── EV_Range_Prediction_Dataset_2000.csv
└── README.md
```

### 4. Run the notebook

Start Jupyter:

``` bash
jupyter notebook
```

Open `Linear_reg (ev_range ).ipynb` and run the cells in order.

**Important:** The notebook currently contains a local Windows file path
in its CSV-loading cell. For portability, replace it with a relative
path:

``` python
df = pd.read_csv("EV_Range_Prediction_Dataset_2000.csv")
```

## Evaluation

The notebook calculates the following metrics on the held-out test set:

-   **MSE (Mean Squared Error):** average squared prediction error.
-   **MAE (Mean Absolute Error):** average absolute prediction error.
-   **RMSE (Root Mean Squared Error):** square root of MSE, in the
    target's units.
-   **NRMSE (Normalized RMSE):** RMSE divided by the target range for
    the evaluated subset.
-   **L2 objective:** MSE plus the L2 penalty on non-intercept
    coefficients.

It also plots the training objective over iterations and compares actual
with predicted range.

> **Reproducibility note:** Re-run the notebook from top to bottom
> before publishing numerical results. The saved notebook output
> includes an inconsistency between the displayed test-set size and the
> split defined in the code, so the existing saved metric values should
> not be treated as verified results.

## Limitations and Future Improvements

-   This is a **baseline linear model**; real-world EV range may have
    nonlinear relationships.
-   The categorical columns `traffic_density` and `driving_style` are
    currently excluded rather than encoded.
-   The notebook uses a random split even though the data contains
    timestamps; a time-based split may be more suitable if the goal is
    to predict future observations.
-   Validate the dataset's origin, measurement units, and
    representativeness before interpreting results.
-   Useful next steps include encoding categorical variables, comparing
    against a scikit-learn baseline, checking residuals, testing on
    independent real-world data, and reporting verified metrics.

## Repository Files

-   `Linear_reg (ev_range ).ipynb` --- data preparation, custom
    regression functions, gradient descent training, and evaluation.
-   `EV_Range_Prediction_Dataset_2000.csv` --- dataset used by the
    notebook.
-   `README.md` --- project documentation.

## License

No license is specified here. Add a license file if you want others to
be able to reuse or modify the project.
