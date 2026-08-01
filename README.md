# Food Crisis Early-Warning System for Kenya

## Overview
A machine learning system that predicts food insecurity at the county-month level across Kenya, built to give early warning ahead of a food crisis rather than reacting after the fact.

## Data
- IPC (Integrated Food Security Phase Classification) phase labels
- CHIRPS rainfall data
- WFP (World Food Programme) food price data

All three sources were merged on county, year, and month keys.

## Approach
Followed the CRISP-DM framework:
1. Cleaned and merged data across three independent sources
2. Ran exploratory data analysis
3. Addressed severe class imbalance using SMOTE
4. Compared 8 model variants, including XGBoost and TabTransformer
5. Tuned the classification threshold (t = 0.08) to balance precision and recall for the minority, crisis class

## Result
Recommended a tuned XGBoost model for deployment over a deep learning alternative, TabTransformer, prioritizing explainability and lighter production dependencies for real-world use by NGOs or government agencies.

## Team
Built collaboratively with a 6-person team using a shared GitHub repository, including forks, branches, and pull requests.

## Tech stack
Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, matplotlib, seaborn
