# Python for Data Science - Complete Lesson

## Table of Contents
1. [Introduction](#introduction)
2. [Setting Up Your Environment](#setting-up-your-environment)
3. [Python Basics for Data Science](#python-basics-for-data-science)
4. [Essential Libraries](#essential-libraries)
5. [Data Manipulation with Pandas](#data-manipulation-with-pandas)
6. [Data Visualization](#data-visualization)
7. [Statistical Analysis](#statistical-analysis)
8. [Machine Learning Basics](#machine-learning-basics)
9. [Practical Projects](#practical-projects)
10. [Resources and Next Steps](#resources-and-next-steps)

## Introduction

Data Science combines programming, statistics, and domain expertise to extract insights from data. Python is one of the most popular languages for data science due to its simplicity, extensive libraries, and strong community support.

### Why Python for Data Science?
- **Easy to learn**: Simple, readable syntax
- **Extensive libraries**: NumPy, Pandas, Matplotlib, Scikit-learn, etc.
- **Community support**: Large community with extensive documentation
- **Versatility**: Can handle data collection, processing, analysis, and deployment

## Setting Up Your Environment

### 1. Install Python and Package Manager

```bash
# Install Anaconda (recommended for data science)
# Download from: https://www.anaconda.com/products/distribution

# Or install Miniconda for a lighter setup
# Download from: https://docs.conda.io/en/latest/miniconda.html

# Verify installation
python --version
conda --version
```

### 2. Create a Virtual Environment

```bash
# Create a new environment
conda create -n datascience python=3.9

# Activate the environment
conda activate datascience

# Install essential packages
conda install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
# or
jupyter lab
```

## Python Basics for Data Science

### 1. Data Types and Structures

```python
# Basic data types
integer_var = 42
float_var = 3.14
string_var = "Hello, Data Science!"
boolean_var = True

# Lists (ordered, mutable)
numbers = [1, 2, 3, 4, 5]
mixed_list = [1, "hello", 3.14, True]

# Dictionaries (key-value pairs)
student = {
    "name": "Alice",
    "age": 25,
    "grades": [85, 90, 88]
}

# Tuples (ordered, immutable)
coordinates = (10.5, 20.3)
```

### 2. Control Structures

```python
# If statements
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
else:
    grade = "C"

# Loops
# For loop
for i in range(5):
    print(f"Iteration {i}")

# While loop
count = 0
while count < 5:
    print(f"Count: {count}")
    count += 1

# List comprehension (powerful for data processing)
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]
```

### 3. Functions

```python

def my_function(*args, **kwargs):
    print("Args: {}".format(args))
    print("Kwargs: {}".format(kwargs))
# Usage
my_function(2, 'a', hello=True, goodbye=None)
```

## Essential Libraries

### 1. NumPy - Numerical Computing

```python
import numpy as np

# Creating arrays
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Array operations
arr_squared = arr ** 2
matrix_sum = matrix.sum()
matrix_mean = matrix.mean()

# Mathematical functions
angles = np.array([0, np.pi/2, np.pi])
sin_values = np.sin(angles)

# Random number generation
random_data = np.random.normal(0, 1, 1000)  # Normal distribution
```

### 2. Pandas - Data Manipulation

```python
import pandas as pd

df = pd.read_csv('data/bestsellers.csv')
df.plot(kind='scatter', x='Reviews', y='Price');
df.plot(kind='scatter',
        x='Reviews',
        y='Price',
        color='orange',
        title='Reviews vs. Price',
        figsize=(12, 6))
```

### 3. Matplotlib - Basic Plotting

```python
import matplotlib.pyplot as plt

# Line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.show()

# Bar plot
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]

plt.figure(figsize=(8, 6))
plt.bar(categories, values)
plt.title('Bar Chart Example')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.show()
```

## Data Manipulation with Pandas

### 1. Data Loading and Exploration

```python
# Load sample data (create sample dataset)
np.random.seed(42)
data = {
    'Date': pd.date_range('2023-01-01', periods=100),
    'Sales': np.random.normal(1000, 200, 100),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
    'Product': np.random.choice(['A', 'B', 'C'], 100)
}
sales_df = pd.DataFrame(data)

# Basic exploration
print("Shape:", sales_df.shape)
print("\nColumn names:", sales_df.columns.tolist())
print("\nData types:\n", sales_df.dtypes)
print("\nFirst 5 rows:\n", sales_df.head())
print("\nMissing values:\n", sales_df.isnull().sum())
```

### 2. Data Filtering and Selection

```python
# Select columns
sales_only = sales_df['Sales']
sales_and_region = sales_df[['Sales', 'Region']]

# Filter rows
high_sales = sales_df[sales_df['Sales'] > 1200]
north_region = sales_df[sales_df['Region'] == 'North']

# Multiple conditions
high_sales_north = sales_df[(sales_df['Sales'] > 1200) & (sales_df['Region'] == 'North')]
```

### 3. Data Aggregation and Grouping

```python
# Group by operations
region_stats = sales_df.groupby('Region')['Sales'].agg(['mean', 'sum', 'count'])
product_sales = sales_df.groupby('Product')['Sales'].mean()

# Multiple grouping
region_product_stats = sales_df.groupby(['Region', 'Product'])['Sales'].mean()

# Pivot tables
pivot_table = sales_df.pivot_table(
    values='Sales',
    index='Region',
    columns='Product',
    aggfunc='mean'
)
```

### 4. Data Cleaning

```python
# Handle missing values
# sales_df.dropna()  # Remove rows with missing values
# sales_df.fillna(sales_df['Sales'].mean())  # Fill with mean

# Remove duplicates
sales_df_clean = sales_df.drop_duplicates()

# Data type conversion
sales_df['Date'] = pd.to_datetime(sales_df['Date'])

# Create new columns
sales_df['Month'] = sales_df['Date'].dt.month
sales_df['Sales_Category'] = sales_df['Sales'].apply(
    lambda x: 'High' if x > 1200 else 'Medium' if x > 800 else 'Low'
)
```

## Data Visualization

### 1. Seaborn - Statistical Plotting

```python
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Distribution plots
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
sns.histplot(sales_df['Sales'], bins=20)
plt.title('Sales Distribution')

plt.subplot(2, 2, 2)
sns.boxplot(data=sales_df, x='Region', y='Sales')
plt.title('Sales by Region')

plt.subplot(2, 2, 3)
sns.scatterplot(data=sales_df, x='Date', y='Sales', hue='Region')
plt.title('Sales Over Time by Region')

plt.subplot(2, 2, 4)
correlation_matrix = sales_df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

plt.tight_layout()
plt.show()
```

### 2. Advanced Visualizations

```python
# Time series plot
monthly_sales = sales_df.groupby(sales_df['Date'].dt.to_period('M'))['Sales'].sum()

plt.figure(figsize=(12, 6))
monthly_sales.plot(kind='line', marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()

# Multiple subplots with different chart types
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Sales by region
sales_df.groupby('Region')['Sales'].mean().plot(kind='bar', ax=axes[0, 0])
axes[0, 0].set_title('Average Sales by Region')

# Sales distribution
sales_df['Sales'].hist(bins=20, ax=axes[0, 1])
axes[0, 1].set_title('Sales Distribution')

# Product performance
sales_df.groupby('Product')['Sales'].sum().plot(kind='pie', ax=axes[1, 0])
axes[1, 0].set_title('Total Sales by Product')

# Sales over time
sales_df.plot(x='Date', y='Sales', kind='scatter', ax=axes[1, 1])
axes[1, 1].set_title('Sales Over Time')

plt.tight_layout()
plt.show()
```

## Statistical Analysis

### 1. Descriptive Statistics

```python
# Basic statistics
print("Descriptive Statistics:")
print(sales_df['Sales'].describe())

# Custom statistics
def calculate_statistics(data):
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'variance': np.var(data),
        'min': np.min(data),
        'max': np.max(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75)
    }

stats = calculate_statistics(sales_df['Sales'])
for key, value in stats.items():
    print(f"{key}: {value:.2f}")
```

### 2. Hypothesis Testing

```python
from scipy import stats

# T-test example: Compare sales between two regions
north_sales = sales_df[sales_df['Region'] == 'North']['Sales']
south_sales = sales_df[sales_df['Region'] == 'South']['Sales']

# Perform independent t-test
t_stat, p_value = stats.ttest_ind(north_sales, south_sales)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Significant difference between regions")
else:
    print("No significant difference between regions")

# Chi-square test for categorical variables
contingency_table = pd.crosstab(sales_df['Region'], sales_df['Product'])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

print(f"\nChi-square test:")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")
```

### 3. Correlation Analysis

```python
# Calculate correlations
numeric_cols = sales_df.select_dtypes(include=[np.number])
correlation_matrix = numeric_cols.corr()

print("Correlation Matrix:")
print(correlation_matrix)

# Visualize correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()
```

## Machine Learning Basics

### 1. Data Preprocessing

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Prepare data for machine learning
# Create a more complex dataset
np.random.seed(42)
ml_data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(0, 1, 1000),
    'feature3': np.random.normal(0, 1, 1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000)
})

# Create target variable with some relationship to features
ml_data['target'] = (
    2 * ml_data['feature1'] +
    1.5 * ml_data['feature2'] +
    0.5 * ml_data['feature3'] +
    np.random.normal(0, 0.1, 1000)
)

# Encode categorical variables
label_encoder = LabelEncoder()
ml_data['category_encoded'] = label_encoder.fit_transform(ml_data['category'])

# Prepare features and target
features = ['feature1', 'feature2', 'feature3', 'category_encoded']
X = ml_data[features]
y = ml_data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2. Model Training and Evaluation

```python
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_predictions = lr_model.predict(X_test_scaled)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)  # Random Forest doesn't require scaling
rf_predictions = rf_model.predict(X_test)

# Evaluate models
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    print()

evaluate_model(y_test, lr_predictions, "Linear Regression")
evaluate_model(y_test, rf_predictions, "Random Forest")

# Feature importance (Random Forest)
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance:")
print(feature_importance)
```

### 3. Model Visualization

```python
# Plot predictions vs actual values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, lr_predictions, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression: Predictions vs Actual')

plt.subplot(1, 2, 2)
plt.scatter(y_test, rf_predictions, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest: Predictions vs Actual')

plt.tight_layout()
plt.show()

# Feature importance plot
plt.figure(figsize=(8, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance (Random Forest)')
plt.show()
```

## Practical Projects

### Project 1: Exploratory Data Analysis (EDA)

```python
def perform_eda(df, target_column=None):
    """
    Perform comprehensive Exploratory Data Analysis
    """
    print("=== EXPLORATORY DATA ANALYSIS ===\n")
    
    # Basic info
    print("1. DATASET OVERVIEW")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print()
    
    # Data types
    print("2. DATA TYPES")
    print(df.dtypes)
    print()
    
    # Missing values
    print("3. MISSING VALUES")
    missing_data = df.isnull().sum()
    missing_percent = 100 * missing_data / len(df)
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    print()
    
    # Numerical variables summary
    print("4. NUMERICAL VARIABLES SUMMARY")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        print(df[numeric_columns].describe())
    print()
    
    # Categorical variables summary
    print("5. CATEGORICAL VARIABLES SUMMARY")
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        print(f"\n{col}:")
        print(df[col].value_counts().head())
    
    # Create visualizations
    if len(numeric_columns) > 0:
        # Distribution plots
        n_cols = min(3, len(numeric_columns))
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        for i, col in enumerate(numeric_columns):
            plt.subplot(n_rows, n_cols, i + 1)
            df[col].hist(bins=30, alpha=0.7)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        # Correlation heatmap
        if len(numeric_columns) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = df[numeric_columns].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            plt.show()

# Usage example
perform_eda(sales_df)
```

### Project 2: Time Series Analysis

```python
def analyze_time_series(df, date_col, value_col):
    """
    Perform time series analysis
    """
    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    # Basic time series plot
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 2, 1)
    plt.plot(df[date_col], df[value_col])
    plt.title(f'{value_col} Over Time')
    plt.xlabel('Date')
    plt.ylabel(value_col)
    
    # Moving averages
    df['MA_7'] = df[value_col].rolling(window=7).mean()
    df['MA_30'] = df[value_col].rolling(window=30).mean()
    
    plt.subplot(3, 2, 2)
    plt.plot(df[date_col], df[value_col], label='Original', alpha=0.5)
    plt.plot(df[date_col], df['MA_7'], label='7-day MA')
    plt.plot(df[date_col], df['MA_30'], label='30-day MA')
    plt.title('Moving Averages')
    plt.legend()
    
    # Seasonal decomposition (if enough data)
    if len(df) > 60:
        # Monthly aggregation
        monthly_data = df.groupby(df[date_col].dt.to_period('M'))[value_col].mean()
        
        plt.subplot(3, 2, 3)
        monthly_data.plot()
        plt.title('Monthly Average')
        
    # Distribution by day of week
    df['day_of_week'] = df[date_col].dt.day_name()
    day_avg = df.groupby('day_of_week')[value_col].mean()
    
    plt.subplot(3, 2, 4)
    day_avg.plot(kind='bar')
    plt.title('Average by Day of Week')
    plt.xticks(rotation=45)
    
    # Distribution by month
    df['month'] = df[date_col].dt.month_name()
    month_avg = df.groupby('month')[value_col].mean()
    
    plt.subplot(3, 2, 5)
    month_avg.plot(kind='bar')
    plt.title('Average by Month')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return df

# Usage
sales_ts = analyze_time_series(sales_df, 'Date', 'Sales')
```

## Resources and Next Steps

### Essential Resources

1. **Official Documentation**
   - [Python.org](https://docs.python.org/3/)
   - [Pandas Documentation](https://pandas.pydata.org/docs/)
   - [NumPy Documentation](https://numpy.org/doc/)
   - [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
   - [Scikit-learn Documentation](https://scikit-learn.org/stable/)

2. **Online Courses**
   - [Coursera - Python for Data Science](https://www.coursera.org/specializations/python)
   - [edX - MIT Introduction to Data Science](https://www.edx.org/course/introduction-to-data-science)
   - [Kaggle Learn](https://www.kaggle.com/learn)

3. **Books**
   - "Python for Data Analysis" by Wes McKinney
   - "Hands-On Machine Learning" by Aurélien Géron
   - "Python Data Science Handbook" by Jake VanderPlas

4. **Practice Platforms**
   - [Kaggle](https://www.kaggle.com/) - Competitions and datasets
   - [Google Colab](https://colab.research.google.com/) - Free Jupyter notebooks
   - [Jupyter Notebook](https://jupyter.org/) - Interactive development

### Next Steps

1. **Practice with Real Datasets**
   - Download datasets from Kaggle, UCI ML Repository, or government open data
   - Work on end-to-end projects from data collection to model deployment

2. **Advanced Topics to Explore**
   - Deep Learning with TensorFlow/PyTorch
   - Natural Language Processing (NLP)
   - Computer Vision
   - Big Data tools (Spark, Dask)
   - Cloud platforms (AWS, GCP, Azure)

3. **Build a Portfolio**
   - Create GitHub repositories with your projects
   - Write blog posts about your analyses
   - Contribute to open-source projects

4. **Join the Community**
   - Follow data science blogs and newsletters
   - Attend meetups and conferences
   - Participate in online forums (Stack Overflow, Reddit r/datascience)

### Sample Project Ideas

1. **Beginner Projects**
   - Analyze your personal data (spending, fitness, etc.)
   - Explore public datasets (weather, census, sports)
   - Build simple prediction models

2. **Intermediate Projects**
   - Customer segmentation analysis
   - Sales forecasting
   - Sentiment analysis of social media data

3. **Advanced Projects**
   - Recommendation systems
   - Real-time data processing
   - End-to-end ML pipeline with deployment

### Tips for Success

- **Start small**: Begin with simple projects and gradually increase complexity
- **Focus on understanding**: Don't just copy code, understand what each line does
- **Practice regularly**: Consistency is key to mastering data science
- **Learn from others**: Study code from experienced practitioners
- **Document your work**: Good documentation helps you and others understand your projects

Remember: Data science is a journey, not a destination. Keep learning, practicing, and exploring new techniques and tools!