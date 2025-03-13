# Random Forest

- Random Forest is an ensemble learning algorithm that builds multiple decision trees and combines their outputs to improve accuracy and reduce overfitting. It is commonly used for classification and regression tasks.

## Step 1: Import Required Libraries




```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE  # Install using: pip install imbalanced-learn

```

#### Imports all required libraries for data processing, visualization, and machine learning.
#### imblearn (SMOTE) is used to handle class imbalance.


```python

```

## Step 2: Load Dataset


```python
# Load dataset (for CSV file)
file_path = "BigfraudTest.csv"  # Ensure the file is in the working directory
df = pd.read_csv(file_path)

# Display basic dataset information
print("Dataset Shape:", df.shape)
print("Missing Values:", df.isnull().sum().sum())
print("First 5 rows of dataset:")
df.head()

```

    Dataset Shape: (50676, 23)
    Missing Values: 0
    First 5 rows of dataset:
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>trans_date_trans_time</th>
      <th>cc_num</th>
      <th>merchant</th>
      <th>category</th>
      <th>amt</th>
      <th>first</th>
      <th>last</th>
      <th>gender</th>
      <th>street</th>
      <th>...</th>
      <th>lat</th>
      <th>long</th>
      <th>city_pop</th>
      <th>job</th>
      <th>dob</th>
      <th>trans_num</th>
      <th>unix_time</th>
      <th>merch_lat</th>
      <th>merch_long</th>
      <th>is_fraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>6/21/2020 12:14</td>
      <td>2.291164e+15</td>
      <td>fraud_Kirlin and Sons</td>
      <td>personal_care</td>
      <td>2.86</td>
      <td>Jeff</td>
      <td>Elliott</td>
      <td>M</td>
      <td>351 Darlene Green</td>
      <td>...</td>
      <td>33.9659</td>
      <td>-80.9355</td>
      <td>333497</td>
      <td>Mechanical engineer</td>
      <td>3/19/1968</td>
      <td>2da90c7d74bd46a0caf3777415b3ebd3</td>
      <td>1371816865</td>
      <td>33.986391</td>
      <td>-81.200714</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6/21/2020 12:14</td>
      <td>3.573030e+15</td>
      <td>fraud_Sporer-Keebler</td>
      <td>personal_care</td>
      <td>29.84</td>
      <td>Joanne</td>
      <td>Williams</td>
      <td>F</td>
      <td>3638 Marsh Union</td>
      <td>...</td>
      <td>40.3207</td>
      <td>-110.4360</td>
      <td>302</td>
      <td>Sales professional, IT</td>
      <td>1/17/1990</td>
      <td>324cc204407e99f51b0d6ca0055005e7</td>
      <td>1371816873</td>
      <td>39.450498</td>
      <td>-109.960431</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>6/21/2020 12:14</td>
      <td>3.598215e+15</td>
      <td>fraud_Swaniawski, Nitzsche and Welch</td>
      <td>health_fitness</td>
      <td>41.28</td>
      <td>Ashley</td>
      <td>Lopez</td>
      <td>F</td>
      <td>9333 Valentine Point</td>
      <td>...</td>
      <td>40.6729</td>
      <td>-73.5365</td>
      <td>34496</td>
      <td>Librarian, public</td>
      <td>10/21/1970</td>
      <td>c81755dbbbea9d5c77f094348a7579be</td>
      <td>1371816893</td>
      <td>40.495810</td>
      <td>-74.196111</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>6/21/2020 12:15</td>
      <td>3.591920e+15</td>
      <td>fraud_Haley Group</td>
      <td>misc_pos</td>
      <td>60.05</td>
      <td>Brian</td>
      <td>Williams</td>
      <td>M</td>
      <td>32941 Krystal Mill Apt. 552</td>
      <td>...</td>
      <td>28.5697</td>
      <td>-80.8191</td>
      <td>54767</td>
      <td>Set designer</td>
      <td>7/25/1987</td>
      <td>2159175b9efe66dc301f149d3d5abf8c</td>
      <td>1371816915</td>
      <td>28.812398</td>
      <td>-80.883061</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>6/21/2020 12:15</td>
      <td>3.526826e+15</td>
      <td>fraud_Johnston-Casper</td>
      <td>travel</td>
      <td>3.19</td>
      <td>Nathan</td>
      <td>Massey</td>
      <td>M</td>
      <td>5783 Evan Roads Apt. 465</td>
      <td>...</td>
      <td>44.2529</td>
      <td>-85.0170</td>
      <td>1126</td>
      <td>Furniture designer</td>
      <td>7/6/1955</td>
      <td>57ff021bd3f328f8738bb535c302a31b</td>
      <td>1371816917</td>
      <td>44.959148</td>
      <td>-85.884734</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



#### Loads the dataset from a CSV file.
#### Prints the shape (rows, columns) and checks for missing values.


```python

```

## Step 3: Drop Unnecessary Columns


```python
# Drop columns that are not useful for fraud detection
drop_columns = ["Unnamed: 0", "first", "last", "street", "trans_num"]
df = df.drop(columns=drop_columns, errors="ignore")

# Confirm dataset after dropping columns
print("Dataset Shape after Dropping Columns:", df.shape)
df.head()

```

    Dataset Shape after Dropping Columns: (50676, 18)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trans_date_trans_time</th>
      <th>cc_num</th>
      <th>merchant</th>
      <th>category</th>
      <th>amt</th>
      <th>gender</th>
      <th>city</th>
      <th>state</th>
      <th>zip</th>
      <th>lat</th>
      <th>long</th>
      <th>city_pop</th>
      <th>job</th>
      <th>dob</th>
      <th>unix_time</th>
      <th>merch_lat</th>
      <th>merch_long</th>
      <th>is_fraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6/21/2020 12:14</td>
      <td>2.291164e+15</td>
      <td>fraud_Kirlin and Sons</td>
      <td>personal_care</td>
      <td>2.86</td>
      <td>M</td>
      <td>Columbia</td>
      <td>SC</td>
      <td>29209</td>
      <td>33.9659</td>
      <td>-80.9355</td>
      <td>333497</td>
      <td>Mechanical engineer</td>
      <td>3/19/1968</td>
      <td>1371816865</td>
      <td>33.986391</td>
      <td>-81.200714</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6/21/2020 12:14</td>
      <td>3.573030e+15</td>
      <td>fraud_Sporer-Keebler</td>
      <td>personal_care</td>
      <td>29.84</td>
      <td>F</td>
      <td>Altonah</td>
      <td>UT</td>
      <td>84002</td>
      <td>40.3207</td>
      <td>-110.4360</td>
      <td>302</td>
      <td>Sales professional, IT</td>
      <td>1/17/1990</td>
      <td>1371816873</td>
      <td>39.450498</td>
      <td>-109.960431</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6/21/2020 12:14</td>
      <td>3.598215e+15</td>
      <td>fraud_Swaniawski, Nitzsche and Welch</td>
      <td>health_fitness</td>
      <td>41.28</td>
      <td>F</td>
      <td>Bellmore</td>
      <td>NY</td>
      <td>11710</td>
      <td>40.6729</td>
      <td>-73.5365</td>
      <td>34496</td>
      <td>Librarian, public</td>
      <td>10/21/1970</td>
      <td>1371816893</td>
      <td>40.495810</td>
      <td>-74.196111</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6/21/2020 12:15</td>
      <td>3.591920e+15</td>
      <td>fraud_Haley Group</td>
      <td>misc_pos</td>
      <td>60.05</td>
      <td>M</td>
      <td>Titusville</td>
      <td>FL</td>
      <td>32780</td>
      <td>28.5697</td>
      <td>-80.8191</td>
      <td>54767</td>
      <td>Set designer</td>
      <td>7/25/1987</td>
      <td>1371816915</td>
      <td>28.812398</td>
      <td>-80.883061</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6/21/2020 12:15</td>
      <td>3.526826e+15</td>
      <td>fraud_Johnston-Casper</td>
      <td>travel</td>
      <td>3.19</td>
      <td>M</td>
      <td>Falmouth</td>
      <td>MI</td>
      <td>49632</td>
      <td>44.2529</td>
      <td>-85.0170</td>
      <td>1126</td>
      <td>Furniture designer</td>
      <td>7/6/1955</td>
      <td>1371816917</td>
      <td>44.959148</td>
      <td>-85.884734</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### Removes unnecessary columns (IDs, names, transaction numbers) that do not help in fraud detection.


```python

```

## Step 4: Convert Categorical Columns to Numerical


```python
# Use Label Encoding for categorical variables
label_encoder = LabelEncoder()
categorical_cols = ["category", "state", "job", "gender"]

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col].astype(str))  # Convert categorical data to numeric

# Check updated dataset
print("Dataset after Encoding Categorical Features:")
df.head()

```

    Dataset after Encoding Categorical Features:
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trans_date_trans_time</th>
      <th>cc_num</th>
      <th>merchant</th>
      <th>category</th>
      <th>amt</th>
      <th>gender</th>
      <th>city</th>
      <th>state</th>
      <th>zip</th>
      <th>lat</th>
      <th>long</th>
      <th>city_pop</th>
      <th>job</th>
      <th>dob</th>
      <th>unix_time</th>
      <th>merch_lat</th>
      <th>merch_long</th>
      <th>is_fraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6/21/2020 12:14</td>
      <td>2.291164e+15</td>
      <td>fraud_Kirlin and Sons</td>
      <td>10</td>
      <td>2.86</td>
      <td>1</td>
      <td>Columbia</td>
      <td>39</td>
      <td>29209</td>
      <td>33.9659</td>
      <td>-80.9355</td>
      <td>333497</td>
      <td>275</td>
      <td>3/19/1968</td>
      <td>1371816865</td>
      <td>33.986391</td>
      <td>-81.200714</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6/21/2020 12:14</td>
      <td>3.573030e+15</td>
      <td>fraud_Sporer-Keebler</td>
      <td>10</td>
      <td>29.84</td>
      <td>0</td>
      <td>Altonah</td>
      <td>43</td>
      <td>84002</td>
      <td>40.3207</td>
      <td>-110.4360</td>
      <td>302</td>
      <td>391</td>
      <td>1/17/1990</td>
      <td>1371816873</td>
      <td>39.450498</td>
      <td>-109.960431</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6/21/2020 12:14</td>
      <td>3.598215e+15</td>
      <td>fraud_Swaniawski, Nitzsche and Welch</td>
      <td>5</td>
      <td>41.28</td>
      <td>0</td>
      <td>Bellmore</td>
      <td>33</td>
      <td>11710</td>
      <td>40.6729</td>
      <td>-73.5365</td>
      <td>34496</td>
      <td>259</td>
      <td>10/21/1970</td>
      <td>1371816893</td>
      <td>40.495810</td>
      <td>-74.196111</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6/21/2020 12:15</td>
      <td>3.591920e+15</td>
      <td>fraud_Haley Group</td>
      <td>9</td>
      <td>60.05</td>
      <td>1</td>
      <td>Titusville</td>
      <td>8</td>
      <td>32780</td>
      <td>28.5697</td>
      <td>-80.8191</td>
      <td>54767</td>
      <td>406</td>
      <td>7/25/1987</td>
      <td>1371816915</td>
      <td>28.812398</td>
      <td>-80.883061</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6/21/2020 12:15</td>
      <td>3.526826e+15</td>
      <td>fraud_Johnston-Casper</td>
      <td>13</td>
      <td>3.19</td>
      <td>1</td>
      <td>Falmouth</td>
      <td>21</td>
      <td>49632</td>
      <td>44.2529</td>
      <td>-85.0170</td>
      <td>1126</td>
      <td>196</td>
      <td>7/6/1955</td>
      <td>1371816917</td>
      <td>44.959148</td>
      <td>-85.884734</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### Encodes categorical columns (like category, state, job, gender) into numerical values.
#### Prevents memory errors by using Label Encoding instead of One-Hot Encoding.


```python

```

## Step 5: Process Date Columns


```python
# Convert 'dob' to age and drop the original column
df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
df["age"] = datetime.now().year - df["dob"].dt.year
df = df.drop(columns=["dob"])

# Convert 'trans_date_trans_time' and extract useful time-based features
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
df["trans_hour"] = df["trans_date_trans_time"].dt.hour
df["trans_day"] = df["trans_date_trans_time"].dt.day
df["trans_month"] = df["trans_date_trans_time"].dt.month
df = df.drop(columns=["trans_date_trans_time"])

# Display dataset after processing dates
print("Dataset after Date Processing:")
df.head()

```

    Dataset after Date Processing:
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cc_num</th>
      <th>merchant</th>
      <th>category</th>
      <th>amt</th>
      <th>gender</th>
      <th>city</th>
      <th>state</th>
      <th>zip</th>
      <th>lat</th>
      <th>long</th>
      <th>city_pop</th>
      <th>job</th>
      <th>unix_time</th>
      <th>merch_lat</th>
      <th>merch_long</th>
      <th>is_fraud</th>
      <th>age</th>
      <th>trans_hour</th>
      <th>trans_day</th>
      <th>trans_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.291164e+15</td>
      <td>fraud_Kirlin and Sons</td>
      <td>10</td>
      <td>2.86</td>
      <td>1</td>
      <td>Columbia</td>
      <td>39</td>
      <td>29209</td>
      <td>33.9659</td>
      <td>-80.9355</td>
      <td>333497</td>
      <td>275</td>
      <td>1371816865</td>
      <td>33.986391</td>
      <td>-81.200714</td>
      <td>0</td>
      <td>57</td>
      <td>12</td>
      <td>21</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.573030e+15</td>
      <td>fraud_Sporer-Keebler</td>
      <td>10</td>
      <td>29.84</td>
      <td>0</td>
      <td>Altonah</td>
      <td>43</td>
      <td>84002</td>
      <td>40.3207</td>
      <td>-110.4360</td>
      <td>302</td>
      <td>391</td>
      <td>1371816873</td>
      <td>39.450498</td>
      <td>-109.960431</td>
      <td>0</td>
      <td>35</td>
      <td>12</td>
      <td>21</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.598215e+15</td>
      <td>fraud_Swaniawski, Nitzsche and Welch</td>
      <td>5</td>
      <td>41.28</td>
      <td>0</td>
      <td>Bellmore</td>
      <td>33</td>
      <td>11710</td>
      <td>40.6729</td>
      <td>-73.5365</td>
      <td>34496</td>
      <td>259</td>
      <td>1371816893</td>
      <td>40.495810</td>
      <td>-74.196111</td>
      <td>0</td>
      <td>55</td>
      <td>12</td>
      <td>21</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.591920e+15</td>
      <td>fraud_Haley Group</td>
      <td>9</td>
      <td>60.05</td>
      <td>1</td>
      <td>Titusville</td>
      <td>8</td>
      <td>32780</td>
      <td>28.5697</td>
      <td>-80.8191</td>
      <td>54767</td>
      <td>406</td>
      <td>1371816915</td>
      <td>28.812398</td>
      <td>-80.883061</td>
      <td>0</td>
      <td>38</td>
      <td>12</td>
      <td>21</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.526826e+15</td>
      <td>fraud_Johnston-Casper</td>
      <td>13</td>
      <td>3.19</td>
      <td>1</td>
      <td>Falmouth</td>
      <td>21</td>
      <td>49632</td>
      <td>44.2529</td>
      <td>-85.0170</td>
      <td>1126</td>
      <td>196</td>
      <td>1371816917</td>
      <td>44.959148</td>
      <td>-85.884734</td>
      <td>0</td>
      <td>70</td>
      <td>12</td>
      <td>21</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



#### Converts dob into age and removes the original column.
#### Extracts hour, day, and month from trans_date_trans_time to capture fraud patterns.


```python

```

## Step 6: Define Features & Target Variable


```python
# Separate features (X) and target (y)
X = df.drop(columns=["is_fraud"])  # Features
y = df["is_fraud"]  # Target variable

# Confirm the shapes
print("Feature Set Shape:", X.shape)
print("Target Set Shape:", y.shape)

```

    Feature Set Shape: (50676, 19)
    Target Set Shape: (50676,)
    

#### Defines features (X) and target (y) for model training.
#### is_fraud is the target variable (0 = Not Fraud, 1 = Fraud).


```python

```

## Step 7: Split Data into Train & Test Sets


```python
# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Confirm the split
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

```

    Training Data Shape: (40540, 19)
    Testing Data Shape: (10136, 19)
    

#### Splits the dataset into 80% training and 20% testing.
#### stratify=y ensures the fraud percentage remains the same in both sets.


```python

```

## Step 8: Handle Class Imbalance Using SMOTE


```python
# Convert all columns to numeric (if any non-numeric data exists)
X_train = X_train.apply(pd.to_numeric, errors="coerce")  
X_test = X_test.apply(pd.to_numeric, errors="coerce")

# Drop columns with NaN values (if conversion caused any issues)
X_train = X_train.dropna(axis=1)
X_test = X_test.dropna(axis=1)

# Apply SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Verify class distribution after SMOTE
print("Class Distribution After SMOTE:\n", y_train_smote.value_counts(normalize=True) * 100)

```

    Class Distribution After SMOTE:
     is_fraud
    0    66.667217
    1    33.332783
    Name: proportion, dtype: float64
    

#### Uses SMOTE to generate synthetic fraud cases and balance the dataset.
#### Ensures the model doesn’t ignore fraud transactions.


```python

```

## Step 9: Feature Scaling (Standardization)




```python
# Standardize only numeric columns
scaler = StandardScaler()
numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

X_train_smote[numeric_cols] = scaler.fit_transform(X_train_smote[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Confirm standardization
print("Dataset after Standardization:")
X_train_smote.head()

```

    Dataset after Standardization:
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cc_num</th>
      <th>category</th>
      <th>amt</th>
      <th>gender</th>
      <th>state</th>
      <th>zip</th>
      <th>lat</th>
      <th>long</th>
      <th>city_pop</th>
      <th>job</th>
      <th>unix_time</th>
      <th>merch_lat</th>
      <th>merch_long</th>
      <th>age</th>
      <th>trans_hour</th>
      <th>trans_day</th>
      <th>trans_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.220944</td>
      <td>-1.511076</td>
      <td>-0.709254</td>
      <td>1.233475</td>
      <td>1.170634</td>
      <td>1.098766</td>
      <td>-1.085550</td>
      <td>-0.595453</td>
      <td>0.059766</td>
      <td>0.790015</td>
      <td>0.591885</td>
      <td>-1.007813</td>
      <td>-0.669002</td>
      <td>38</td>
      <td>21</td>
      <td>2</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.385159</td>
      <td>1.483800</td>
      <td>-0.706290</td>
      <td>1.233475</td>
      <td>1.313020</td>
      <td>-0.893708</td>
      <td>-0.053280</td>
      <td>0.808140</td>
      <td>-0.265682</td>
      <td>-0.930167</td>
      <td>-0.459547</td>
      <td>0.035626</td>
      <td>0.757948</td>
      <td>50</td>
      <td>17</td>
      <td>27</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.383944</td>
      <td>0.394754</td>
      <td>-0.709764</td>
      <td>-0.810718</td>
      <td>-1.748277</td>
      <td>-0.378350</td>
      <td>-1.166334</td>
      <td>0.181907</td>
      <td>-0.299005</td>
      <td>0.287661</td>
      <td>0.332899</td>
      <td>-1.025687</td>
      <td>0.180133</td>
      <td>54</td>
      <td>15</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.385488</td>
      <td>1.483800</td>
      <td>-0.221499</td>
      <td>-0.810718</td>
      <td>0.672284</td>
      <td>0.998394</td>
      <td>-0.713886</td>
      <td>-0.724679</td>
      <td>-0.301060</td>
      <td>-1.638029</td>
      <td>0.782351</td>
      <td>-0.799363</td>
      <td>-0.778320</td>
      <td>55</td>
      <td>20</td>
      <td>3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.387764</td>
      <td>-0.422030</td>
      <td>-0.578398</td>
      <td>-0.810718</td>
      <td>-0.751575</td>
      <td>0.773718</td>
      <td>0.011994</td>
      <td>-0.767171</td>
      <td>-0.301942</td>
      <td>-0.382144</td>
      <td>1.355253</td>
      <td>0.179196</td>
      <td>-0.812666</td>
      <td>64</td>
      <td>16</td>
      <td>6</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



#### Scales numerical values to prevent bias due to large numbers (e.g., transaction amount).


```python

```

## Step 10: Train Random Forest Model


```python
# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_smote, y_train_smote)

# Model training completed
print("✅ Random Forest Model Trained Successfully!")

```

    ✅ Random Forest Model Trained Successfully!
    

#### Trains a Random Forest Classifier to detect fraud.


```python

```

## Step 11: Model Predictions


```python
# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Check sample predictions
print("Sample Predictions:", y_pred[:10])

```

    Sample Predictions: [0 0 0 0 0 0 0 0 0 0]
    

#### Uses the trained Random Forest model to predict fraud (1) or non-fraud (0) on the test dataset.
#### Prints the first 10 predictions to verify output.




```python

```

## Step 12: Model Evaluation



```python
# Print evaluation metrics
print("🔹 Accuracy Score:", accuracy_score(y_test, y_pred))
print("🔹 ROC-AUC Score:", roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1]))
print("🔹 Classification Report:\n", classification_report(y_test, y_pred))

```

    🔹 Accuracy Score: 0.9989147592738753
    🔹 ROC-AUC Score: 0.9984015946909668
    🔹 Classification Report:
                   precision    recall  f1-score   support
    
               0       1.00      1.00      1.00     10096
               1       0.91      0.80      0.85        40
    
        accuracy                           1.00     10136
       macro avg       0.96      0.90      0.93     10136
    weighted avg       1.00      1.00      1.00     10136
    
    

#### Evaluates model performance using key metrics:

- Accuracy Score → Measures the percentage of correct predictions.
- ROC-AUC Score → Shows how well the model distinguishes fraud from non-fraud.
- Classification Report → Provides precision, recall, and F1-score for each class.



```python

```

## Step 13: Visualize Confusion Matrix


```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Generate Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Define labels for each cell
group_labels = np.array([["True Negative (TN)", "False Positive (FP)"], 
                         ["False Negative (FN)", "True Positive (TP)"]])

# Convert numeric values to text labels
group_counts = conf_matrix.flatten()
labels = np.array([f"{group_labels[i, j]}\nCount: {group_counts[i * 2 + j]}" 
                   for i in range(2) for j in range(2)]).reshape(2, 2)

# Plot Confusion Matrix with True/False Labels
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=labels, fmt="", cmap="Blues", xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

```


    
![png](output_51_0.png)
    


#### The meaning (TN, FP, FN, TP).
#### The actual count from the confusion matrix.
#### ✅ Makes fraud detection errors easier to analyze.



```python

```

### Option 2: Precision-Recall Curve



```python
from sklearn.metrics import precision_recall_curve

# Get precision, recall, thresholds
precision, recall, _ = precision_recall_curve(y_test, rf_model.predict_proba(X_test)[:, 1])

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 5))
plt.plot(recall, precision, marker=".", color="blue")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.show()

```


    
![png](output_55_0.png)
    


#### Best for imbalanced datasets like fraud detection.
#### Shows trade-off between precision (reducing false positives) and recall (catching more fraud cases).


```python

```

## Step 14: Feature Importance Analysis



```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Extract feature importance
feature_importances = pd.DataFrame({
    "Feature": X_train.columns, 
    "Importance": rf_model.feature_importances_
})
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

# Plot Top 10 Important Features (Final Fixed Version)
plt.figure(figsize=(10, 5))
ax = sns.barplot(
    data=feature_importances[:10],  # Pass data directly
    x="Importance", 
    y="Feature", 
    hue="Feature",  # Assign hue to fix warning
    dodge=False,  # Ensure no grouping
    legend=False,  # Remove legend
    palette="viridis"
)

plt.title("Top 10 Important Features in Fraud Detection")
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature Names")
plt.show()

```


    
![png](output_59_0.png)
    


#### The higher the importance score, the more crucial the feature is in detecting fraud.
- Example important features could be:
- Transaction Amount (amt) → High-value transactions might indicate fraud.
- Transaction Hour (trans_hour) → Fraud might occur at odd hours.
- Merchant Category (category) → Some categories (e.g., luxury items) may have more fraud.


```python

```


```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Select one tree from the Random Forest
single_tree = rf_model.estimators_[0]  # Extract the first tree

# Plot the tree
plt.figure(figsize=(20, 10))
plot_tree(single_tree, filled=True, feature_names=X_train.columns, class_names=["Not Fraud", "Fraud"], rounded=True)
plt.title("Decision Tree from Random Forest")
plt.show()

```


    
![png](output_62_0.png)
    



```python

```

### The fraud detection pipeline successfully processed and structured the dataset by handling missing values, encoding categorical features, and extracting useful time-based information. To address the class imbalance, SMOTE was applied, generating synthetic fraud cases and ensuring the model was not biased toward non-fraudulent transactions. The Random Forest model performed well, accurately identifying fraudulent activities by leveraging key features such as transaction amount, time, and merchant category. The evaluation metrics, including the confusion matrix and feature importance analysis, provided insights into the model’s strengths and areas for improvement. Overall, the model is effective in fraud detection but could benefit from further optimization, such as hyperparameter tuning, to enhance precision and recall. 


```python

```
