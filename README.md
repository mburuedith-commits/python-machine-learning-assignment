# python-machine-learning-assignmentPython Machine Learning Assignment
Student Name: Edith Mburu 
Topic: Supervised and Unsupervised Machine Learning 

# Project Overview
This repository contains hands-on implementations of basic machine learning concepts using Python and `scikit-learn`. The codebase is split into two primary paradigms:
Supervised Learning: A linear regression application predicting continuous student exam scores based on hours studied.
Unsupervised Learning: A K-Means clustering implementation categorizing retail customers into behavioral segments using spending profiles.

# Repository Structure

My project follows this mandatory clean structure to satisfy all repository guidelines:

python-machine-learning-assignment
supervised_ml.py - Supervised ML: Linear Regression Model
unsupervised_ml.py - Unsupervised ML: K-Means Clustering Model
requirements.txt - Production third-party dependencies
README.md - Assignment documentation & Theory solutions

# Installation & Execution Instructions
Followed these sequential steps to set up the runtime environment and execute the programs without errors.
Prerequisites
I have **Python 3.8+** installed on your machine. 

# Install Dependencies
opened terminal or command prompt inside the project directory and installed the necessary data science libraries listed in `requirements.txt`:
bash
pip install -r requirements.txt

# Running the Supervised ML Model (`supervised_ml.py`)
Ran the script to launch the interactive CLI prompt. Input numeric value representing study hours to receive an immediate predicted exam score:
`bash
python supervised_ml.py

# Project Detailed Descriptions
# Part A: Supervised Learning (Student Exam Score Prediction)
Objective: A predictive relationships model determining how studying duration impacts academic scoring thresholds.
Algorithm Used:`LinearRegression` from `sklearn.linear_model`.
Workflow: Data points are prepared via native arrays and scaled into a matching matrix format. The model is fitted using a loss function, opening an active standard input stream to process dynamic calculations.
Bonus Integration: Includes a Matplotlib graph projecting actual coordinates, a continuous regression path, and visual crosshairs mapping user-generated test criteria.

Part B: Unsupervised Learning (Customer Segmentation via K-Means)
Objective: Organize distinct customer categories utilizing metrics tracking total monthly purchasing velocity (KES) and physical visit tallies.
Algorithm Used: `KMeans` from `sklearn.cluster`.
Workflow: Features are parsed through `StandardScaler` to bring disparate units (thousands of KES vs single-digit visits) into a zero-mean, unit-variance distribution. The K-Means model applies distance optimization (`n_clusters=3`, `random_state=42`, `n_init=10`) to tag similar profiles.
Bonus Integration: Generates an multi-colored analytical scatter plot segmenting user behavior distributions along a clear color legend.

PART C: THEORY ANSWERS

# What is supervised machine learning?
Supervised learning uses labeled training data, meaning every input sample comes with its correct output answer.

# What is unsupervised machine learning?
Unsupervised learning works with unlabeled data, automatically finding hidden patterns or structures on its own.

# Explain the difference between supervised and unsupervised learning.
Supervised learning maps data directly to known target outputs. To predict outcomes or classify new, unseen data. 
Core Tasks: Classification and Regression.
Unsupervised learning groups data points based on mathematical similarity without target outputs. To discover inherent patterns, groupings, or associations.
Core taske: Clustering, Association, and Dimensionality Reduction.

# Why is Linear Regression considered supervised learning?
It uses paired target data (historical study hours matched to historical exam scores) to compute a mathematical line of best fit.

# Why is K-Means considered unsupervised learning?
It reads raw, untagged data points and automatically groups them into spatial segments without relying on pre-existing answer keys.

# What is the difference between fit() and predict()?
The fit() method trains the model parameters (calculating weights or cluster centroids), while predict() uses those trained parameters to estimate tags on data.

# Real-world applications:
Supervised: Email spam sorting and real estate property valuation.
Unsupervised: Customer market segmentation and credit card transaction fraud detection.

