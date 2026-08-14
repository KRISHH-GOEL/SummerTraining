# Lesson 01 – Machine Learning Fundamentals

## 📌 Objective

The objective of this lesson was to understand the fundamental concepts of Machine Learning and how a Machine Learning problem is formulated and solved.

Machine Learning enables computers to learn patterns from data and use those patterns to make predictions or decisions without explicitly programming every possible rule.

This lesson establishes the foundation required before implementing Machine Learning algorithms.

---

## 📚 Topics Covered

### Introduction to Machine Learning

- What is Machine Learning?
- Why Machine Learning?
- Traditional Programming vs Machine Learning
- AI vs Machine Learning vs Deep Learning

### Types of Machine Learning

- Supervised Learning
- Unsupervised Learning
- Semi-Supervised Learning
- Reinforcement Learning

### Supervised Learning

- Features
- Target
- Training Data
- Testing Data
- Regression
- Classification

### Unsupervised Learning

- Clustering
- Dimensionality Reduction

### Important ML Concepts

- Model
- Training
- Prediction
- Generalization
- Underfitting
- Overfitting
- Bias
- Variance
- Bias-Variance Tradeoff

### Machine Learning Workflow

```text
Problem Definition
        ↓
Data Collection
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Feature / Target Separation
        ↓
Train-Test Split
        ↓
Model Training
        ↓
Prediction
        ↓
Evaluation
        ↓
Model Improvement
        ↓
Deployment
```

---

## 🎯 Learning Outcomes

After completing this lesson, I can:

- Explain what Machine Learning is.
- Differentiate AI, ML, and Deep Learning.
- Identify different types of Machine Learning.
- Distinguish features from the target variable.
- Understand regression and classification problems.
- Explain training and testing data.
- Understand overfitting and underfitting.
- Understand the bias-variance tradeoff.
- Describe the complete Machine Learning workflow.

---

## 🧠 Traditional Programming vs Machine Learning

### Traditional Programming

```text
Rules + Data
     ↓
  Program
     ↓
  Output
```

### Machine Learning

```text
Data + Expected Output
        ↓
     Learning
        ↓
      Model
        ↓
   New Data
        ↓
    Prediction
```

---

## 🔵 Supervised Learning

In supervised learning, the model learns from data where the correct target/output is already known.

### Example

Predicting house prices:

```text
Features:
- Area
- Bedrooms
- Location
- Age

Target:
- House Price
```

Two major supervised learning problems are:

### Regression

Predicts a continuous numerical value.

Examples:

- House price
- Temperature
- Sales
- Salary

### Classification

Predicts a category/class.

Examples:

- Spam / Not Spam
- Disease / No Disease
- Churn / No Churn
- Fraud / Not Fraud

---

## 🟢 Unsupervised Learning

In unsupervised learning, the target variable is not provided.

The algorithm attempts to discover hidden patterns or structures in the data.

Examples:

- Customer segmentation
- Clustering
- Dimensionality reduction

Common algorithms:

- K-Means
- Hierarchical Clustering
- DBSCAN
- PCA

---

## 🟡 Semi-Supervised Learning

Uses a combination of:

```text
Small Amount of Labeled Data
+
Large Amount of Unlabeled Data
```

This is useful when obtaining labeled data is expensive or time-consuming.

---

## 🔴 Reinforcement Learning

An agent learns by interacting with an environment.

The agent receives:

- Rewards
- Penalties

and learns a strategy that maximizes cumulative reward.

Examples:

- Game playing
- Robotics
- Autonomous systems
- Recommendation strategies

---

## ⚠️ Overfitting vs Underfitting

### Underfitting

The model is too simple and fails to learn important patterns.

```text
Training Performance → Poor
Testing Performance  → Poor
```

### Overfitting

The model learns the training data too closely, including noise.

```text
Training Performance → Very Good
Testing Performance  → Poor
```

### Good Generalization

```text
Training Performance → Good
Testing Performance  → Good
```

---

## ⚖️ Bias-Variance Tradeoff

### High Bias

Usually results in:

```text
Underfitting
```

### High Variance

Usually results in:

```text
Overfitting
```

The goal is to find a model that balances bias and variance and generalizes well to unseen data.

---

## 📦 Libraries Used

- Python
- NumPy
- Pandas
- Scikit-learn

Install Scikit-learn:

```bash
pip install scikit-learn
```

---

## 📝 Practical Concepts

The accompanying Python file demonstrates:

- Creating a supervised learning dataset
- Separating features and target
- Performing a train-test split
- Training a simple model
- Making predictions
- Demonstrating classification and regression concepts
- Demonstrating underfitting and overfitting using polynomial models

---

## 🚀 Key Takeaway

Machine Learning is not simply about selecting an algorithm.

A successful Machine Learning solution requires:

```text
Good Problem Definition
        +
Good Data
        +
Good Features
        +
Correct Validation
        +
Appropriate Model
        +
Correct Evaluation
```

Understanding the complete workflow is more important than memorizing individual algorithms.
