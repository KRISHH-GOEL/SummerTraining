# Lesson 04 – Regression Algorithms

## 📌 Objective

The objective of this lesson was to understand Regression as a supervised Machine Learning problem and implement several regression algorithms using Scikit-learn.

Regression models are used when the target variable is a continuous numerical value.

Examples include:

- House price prediction
- Salary prediction
- Sales forecasting
- Demand prediction
- Temperature prediction

This lesson covers linear, regularized, nonlinear, and tree-based regression algorithms.

---

# 📚 Topics Covered

## Regression Fundamentals

- What is Regression?
- Continuous target variables
- Independent and dependent variables
- Regression equation
- Coefficients
- Intercept
- Residuals
- Predictions

## Linear Regression

- Simple Linear Regression
- Multiple Linear Regression
- `LinearRegression()`
- Coefficients
- Intercept
- Predictions

## Polynomial Regression

- Nonlinear relationships
- Polynomial features
- `PolynomialFeatures`
- Polynomial Regression

## Regularization

### Ridge Regression

- L2 regularization
- `Ridge()`
- Controlling model complexity

### Lasso Regression

- L1 regularization
- `Lasso()`
- Feature selection through coefficient shrinkage

### ElasticNet

- Combination of L1 and L2 regularization
- `ElasticNet()`

## Tree-Based Regression

### Decision Tree Regression

- Splitting
- Decision rules
- Tree depth
- Nonlinear relationships

### Random Forest Regression

- Ensemble of decision trees
- Bootstrap sampling
- Random feature selection
- Aggregating predictions

---

# 🧠 What Is Regression?

Regression is a supervised learning technique where the target variable is numerical and continuous.

Example:

```text
Area     Bedrooms     Price
1000       2          200000
1500       3          300000
2000       4          400000
```

The model learns the relationship between:

```text
Features → Target
```

and predicts the target for unseen observations.

---

# 📐 Linear Regression

A simple linear regression model can be represented as:

```text
y = β₀ + β₁x
```

Multiple linear regression:

```text
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

Where:

- `y` = predicted target
- `β₀` = intercept
- `β₁, β₂, ...` = coefficients
- `x₁, x₂, ...` = features

---

# 📊 Residuals

A residual represents the difference between the actual and predicted value.

```text
Residual = Actual Value - Predicted Value
```

A good regression model attempts to minimize prediction errors according to its optimization objective.

---

# 🔄 Polynomial Regression

Linear regression assumes a linear relationship.

Polynomial regression introduces additional terms:

```text
y = β₀ + β₁x + β₂x² + β₃x³
```

It can model nonlinear relationships.

However, excessively high polynomial degrees can cause overfitting.

---

# 🛡️ Regularization

Regularization discourages excessively large model coefficients.

## Ridge

Uses L2 regularization.

Conceptually:

```text
Loss + α × Σ(coefficient²)
```

A larger `alpha` means stronger regularization.

---

## Lasso

Uses L1 regularization.

Conceptually:

```text
Loss + α × Σ|coefficient|
```

Lasso can shrink some coefficients to exactly zero, which can provide a form of feature selection.

---

## ElasticNet

Combines L1 and L2 regularization.

It provides a balance between:

```text
Lasso
+
Ridge
```

---

# 🌳 Decision Tree Regression

A Decision Tree learns a sequence of rules that divide the feature space into regions.

Example:

```text
Income > 50000?
       │
   ┌───┴───┐
  Yes      No
   │        │
Age > 30?  ...
```

Advantages:

- Captures nonlinear relationships
- Does not require feature scaling
- Easy to visualize conceptually

Disadvantage:

- Can overfit without appropriate constraints.

Important hyperparameters include:

```text
max_depth
min_samples_split
min_samples_leaf
```

---

# 🌲 Random Forest Regression

Random Forest combines multiple decision trees.

```text
Dataset
   ↓
Multiple Decision Trees
   ↓
Individual Predictions
   ↓
Average Predictions
   ↓
Final Prediction
```

It generally provides stronger generalization than a single unconstrained decision tree.

Important hyperparameters include:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
max_features
```

---

# 📏 Regression Evaluation Metrics

The main metrics introduced in this lesson are:

## MAE

Mean Absolute Error.

```text
MAE = average(|Actual - Prediction|)
```

It represents the average absolute prediction error.

---

## MSE

Mean Squared Error.

```text
MSE = average((Actual - Prediction)²)
```

Large errors receive greater penalty.

---

## RMSE

Root Mean Squared Error.

```text
RMSE = √MSE
```

It is expressed in the same units as the target.

---

## R² Score

Measures the proportion of target variance explained by the model.

```text
R² = 1 - SS_res / SS_tot
```

Higher values generally indicate better fit, although the metric must be interpreted in context.

---

# ⚠️ Important Concepts

### Overfitting

A model performs extremely well on training data but poorly on unseen data.

### Underfitting

A model is too simple to capture important patterns.

### Regularization

Helps control model complexity.

### Scaling

Linear regression itself does not strictly require feature scaling, but scaling can be important for regularized linear models and many other algorithms.

Tree-based models generally do not require feature scaling.

---

# 📦 Libraries Used

- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Install:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

# 🧪 Dataset Used

The implementation creates a synthetic house-price dataset containing:

### Features

- Area
- Bedrooms
- Bathrooms
- Age
- DistanceFromCity

### Target

- Price

The same dataset is used to compare different regression algorithms.

---

# 🔄 Regression Workflow

```text
Dataset
   ↓
Separate X and y
   ↓
Train-Test Split
   ↓
Preprocessing
   ↓
Train Multiple Regression Models
   ↓
Make Predictions
   ↓
Calculate MAE / MSE / RMSE / R²
   ↓
Compare Models
   ↓
Select Appropriate Model
```

---

# 🎯 Key Takeaway

There is no universally best regression algorithm.

Model selection depends on:

- Dataset size
- Relationship between variables
- Noise
- Outliers
- Feature characteristics
- Interpretability requirements
- Computational requirements

The correct approach is to train multiple reasonable models and evaluate them using an appropriate validation strategy.

---

# 📝 Mini Practice

1. Train Linear Regression using only `Area`.

2. Train Multiple Linear Regression using all features.

3. Try Polynomial Regression with degrees 2 and 3.

4. Compare Ridge and Lasso.

5. Experiment with different regularization strengths.

6. Train a Decision Tree Regressor.

7. Change `max_depth` and observe overfitting.

8. Train a Random Forest Regressor.

9. Compare all models using:
   - MAE
   - RMSE
   - R²

10. Identify the model with the best test performance.

11. Compare training and testing performance to detect overfitting.

12. Plot actual vs predicted values for the best model.
