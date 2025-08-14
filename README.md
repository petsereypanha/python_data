# Python Fundamentals - Interactive Learning Project

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Getting Started](#getting-started)
4. [Learning Modules](#learning-modules)
5. [Dependencies](#dependencies)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [Next Steps](#next-steps)

## Overview

This is a comprehensive Python learning project featuring interactive Jupyter notebooks that cover fundamental Python programming concepts. The project is organized into modular sections, each focusing on specific Python topics with hands-on exercises and practical examples.

### What You'll Learn
- **Python Basics**: Data types, control structures, and core programming concepts
- **Functions**: Basic and advanced function concepts, parameters, and scope
- **Object-Oriented Programming**: Classes, inheritance, polymorphism, and magic methods
- **Data Structures**: Lists, dictionaries, collections, and nested data structures
- **File Management**: Reading/writing files, CSV handling, and HTTP operations
- **Database Operations**: PostgreSQL integration and database APIs
- **Data Serialization**: JSON and XML processing
- **Exception Handling**: Custom exceptions and error management
- **NumPy**: Numerical computing and array operations
- **Pandas**: Data analysis and manipulation
- **Matplotlib**: Data visualization and plotting
- **Standard Library**: Built-in Python modules and utilities

## Project Structure

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
ine/
├── README.md                           # Project documentation
├── introduction/                       # Python basics and fundamentals
│   ├── list.ipynb                     # Lists and list operations
│   ├── Dictionaries.ipynb             # Dictionary data structures
│   └── Loops.ipynb                    # For/while loops and iteration
├── function/                          # Function concepts
│   ├── Basic.ipynb                   # Basic function syntax and usage
│   └── Advanced_Function.ipynb       # Advanced function concepts
├── object-oriented-programming/       # OOP concepts
│   ├── Introduction.ipynb            # OOP basics
│   ├── Atrributes.ipynb             # Class attributes
│   ├── Method.ipynb                 # Class methods
│   ├── Inheritance.ipynb            # Inheritance concepts
│   ├── Polymorphism.ipynb           # Polymorphism examples
│   ├── Super_and_Overriding.ipynb   # Method overriding
│   ├── Magic_Methods.ipynb          # Special methods
│   └── Getattr_Setattr_Hasattr.ipynb # Attribute manipulation
├── collection/                        # Advanced data structures
│   └── Nested Collections.ipynb      # Working with nested collections
├── file-management-and-http/          # File and web operations
│   ├── Intro to File Management.ipynb
│   ├── Intro to Files (Cheatsheet).ipynb
│   ├── Writing_Files.ipynb
│   ├── The with context manager.ipynb
│   ├── CSV Old School.ipynb
│   ├── HTTP Practice.ipynb
│   ├── alice.txt                     # Sample text file
│   └── products.csv                  # Sample CSV data
├── database/                          # Database operations
│   └── pg/                           # PostgreSQL examples
│       ├── DB-API.ipynb
│       ├── Adapters.ipynb
│       ├── Admin-Tools.ipynb
│       └── DDL.ipynb
├── serialization/                     # Data serialization
│   ├── Data - Serialization.ipynb    # JSON serialization
│   ├── XML - Serialization.ipynb     # XML processing
│   └── data/                         # Sample data files
│       ├── data.json
│       ├── movie.csv
│       ├── movie.txt
│       └── quran.xml
├── exceptions/                        # Error handling
│   └── Custom Exceptions.ipynb       # Custom exception classes
├── numpy/                            # Numerical computing
│   └── Introduction.ipynb            # NumPy arrays and operations
├── pandas/                           # Data analysis and manipulation
│   ├── introduction.ipynb            # Getting started with pandas
│   ├── Analys.ipynb                 # Data analysis examples
│   ├── Plot_Analsis.ipynb           # Data visualization with pandas
│   └── data/
│       ├── bestsellers.csv          # Amazon bestsellers dataset
│       ├── cars.csv                 # Car data for analysis
│       └── invoices.csv              # Sample invoice data
├── Matplotlib/                       # Data visualization
│   └── introduction.ipynb            # Plotting and visualization basics
├── module/                           # Python modules
│   └── modules.py                    # Module examples
└── standard_library/                 # Python standard library usage
```

## Getting Started

### Prerequisites
- Python 3.7+ installed on your system
- Jupyter Notebook or JupyterLab
- Basic understanding of programming concepts (helpful but not required)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ine
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   # Using conda
   conda create -n python-fundamentals python=3.9
   conda activate python-fundamentals
   
   # Or using venv
   python -m venv python-fundamentals
   source python-fundamentals/bin/activate  # On Windows: python-fundamentals\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install jupyter pandas matplotlib seaborn numpy psycopg2-binary requests
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

## Learning Modules

### 1. Introduction (Start Here)
Begin with the `introduction/` folder to learn Python basics:
- **Lists**: Understanding list data structures and operations
- **Dictionaries**: Key-value pairs and dictionary methods
- **Loops**: Iteration and control flow

### 2. Functions
Explore `function/` to understand:
- Basic function syntax and parameters
- Advanced concepts like *args, **kwargs
- Scope and local/global variables

### 3. Object-Oriented Programming
The `object-oriented-programming/` section covers:
- Class creation and instantiation
- Attributes and methods
- Inheritance and polymorphism
- Special methods (magic methods)

### 4. Data Structures & Collections
Learn advanced data manipulation:
- **Collections**: Working with nested data structures
- **File Management**: Reading and writing files
- **Serialization**: JSON and XML processing

### 5. Scientific Computing & Data Analysis
Dive into numerical computing and data science:
- **NumPy**: Array operations and numerical computing
- **Pandas**: Data manipulation, analysis, and cleaning
- **Matplotlib**: Creating plots and visualizations

### 6. Advanced Topics
- **Database Operations**: PostgreSQL integration with DB-API
- **Exception Handling**: Custom error management
- **HTTP Operations**: Web requests and APIs
- **Standard Library**: Built-in Python modules

## Dependencies

```text
jupyter>=1.0.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
numpy>=1.21.0
psycopg2-binary>=2.9.0
requests>=2.25.0
```

## Usage

### Running Individual Notebooks
1. Navigate to the desired module folder
2. Open the corresponding `.ipynb` file in Jupyter
3. Run cells sequentially using `Shift + Enter`
4. Complete exercises and experiment with the code

### Recommended Learning Path
1. **Start with `introduction/`** - Python basics (lists, dictionaries, loops)
2. **Progress to `function/`** - Function concepts and advanced features
3. **Move to `object-oriented-programming/`** - OOP principles and implementation
4. **Explore `collection/`** - Advanced data structures
5. **Practice with `file-management-and-http/`** - File operations and web requests
6. **Learn `numpy/`** - Numerical computing fundamentals
7. **Advance to `pandas/`** - Data analysis and manipulation
8. **Visualize with `Matplotlib/`** - Creating plots and charts
9. **Experiment with specialized topics** - databases, serialization, exceptions

### Working with Sample Data
The project includes various sample datasets:
- `pandas/data/bestsellers.csv` - Amazon bestsellers for analysis practice
- `pandas/data/cars.csv` - Car data for basic operations
- `pandas/data/invoices.csv` - Invoice data for business analytics
- `file-management-and-http/products.csv` - Product data for file operations
- `file-management-and-http/alice.txt` - Text file for reading exercises
- `serialization/data/` - Various formats (JSON, CSV, XML) for serialization practice

## Contributing

This is a learning project, but contributions are welcome! Here's how you can help:

### Adding New Content
1. Fork the repository
2. Create a new branch for your feature
3. Add new notebooks following the existing structure
4. Include clear explanations and practical examples
5. Test all code cells to ensure they run without errors
6. Submit a pull request with a description of your additions

### Improving Existing Content
- Fix typos or unclear explanations
- Add more examples or exercises
- Improve code comments and documentation
- Suggest better learning progressions
- Add more sample datasets

### Guidelines
- Keep notebooks focused on specific topics
- Include markdown cells with clear explanations
- Provide practical, runnable examples
- Add sample data when helpful
- Follow consistent naming conventions
- Ensure code works with the specified dependencies

## Next Steps

After completing this fundamental course, consider exploring:

### Advanced Python Topics
- Decorators and context managers
- Generators and iterators
- Asyncio and concurrent programming
- Testing with pytest
- Package development and distribution

### Data Science Specialization
- Advanced pandas operations (groupby, pivot tables, time series)
- Statistical analysis with SciPy
- Machine learning with Scikit-learn
- Deep learning with TensorFlow or PyTorch
- Data visualization with Plotly and Bokeh

### Web Development
- Flask or Django frameworks
- REST API development
- Database design and management
- Frontend integration with JavaScript

### Specialized Applications
- Financial analysis and quantitative finance
- Bioinformatics and computational biology
- Image processing with OpenCV
- Natural language processing with NLTK/spaCy
- Automation and scripting

### Resources for Continued Learning
- [Python.org Documentation](https://docs.python.org/3/)
- [Real Python Tutorials](https://realpython.com/)
- [Python Package Index (PyPI)](https://pypi.org/)
- [Kaggle Learn](https://www.kaggle.com/learn) - For data science
- [LeetCode](https://leetcode.com/) - For algorithm practice
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)

---

**Happy Learning!** 🐍

Remember: Programming is best learned by doing. Don't just read the code—run it, modify it, and experiment with it. Each notebook is designed to be interactive, so make the most of the hands-on exercises!
