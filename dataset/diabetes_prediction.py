# Libraries for data and ML
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Load dataset
df = pd.read_csv("diabetes.csv")  # Replace with your dataset path
print(df.head())  # First 5 rows
print(df.info())  # Info about columns & data types
print(df.describe())  # Stats of numeric columns
