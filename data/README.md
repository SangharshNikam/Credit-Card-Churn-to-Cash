# Data

This project uses the public Kaggle dataset **Credit Card customers (BankChurners)**:
https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers

The raw file is not redistributed here; download it from Kaggle and place `BankChurners.csv` in this folder (or point the scripts at it).

- 10,127 credit card customers, 23 columns, no missing values, no date column
- Target: `Attrition_Flag` (1,627 attrited, 16.07%)
- The last two columns (`Naive_Bayes_Classifier_...`) are pre-computed predictions bundled with the file. They leak the target and are dropped in every script, together with the `CLIENTNUM` id.
