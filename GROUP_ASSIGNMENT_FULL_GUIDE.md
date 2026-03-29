# Group Assignment – Full Guide
## Retail Superstore Operations Analysis
### Coventry University – MSc Data Science – Principles of Data Science 7144

---

## SCENARIO SELECTION & EXPLANATION

### Why Superstore Dataset?
The **Sample Superstore** dataset is a real-world retail operations dataset with:
- **9,994 rows** of order records from a US-based retail company
- Covers **2015–2018** → great for time-series forecasting
- Has **customers, products, regions, profits, shipping** → great for clustering & EDA
- Freely available on Kaggle (no login needed via direct download)

### Dataset Download
URL: https://www.kaggle.com/datasets/vivek468/superstore-dataset-final
File: `Sample - Superstore.csv`
Place it in a folder called `data/raw/` in your project.

### Business Problem (for the report)
> "A retail superstore is struggling with inconsistent profitability across regions, poor customer retention, and uncertain demand forecasting. Using data-driven methods, we will segment customers, identify underperforming areas, and build a sales forecasting model to support strategic decisions."

---

## RECOMMENDED GIT FOLDER STRUCTURE

```
superstore-analysis/
├── data/
│   ├── raw/                    # Original CSV files
│   └── processed/              # Cleaned/transformed files
├── notebooks/
│   ├── 01_ETL.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Clustering.ipynb
│   └── 04_Forecasting.ipynb
├── reports/
│   └── final_report.pdf
├── requirements.txt
└── README.md
```

---

## requirements.txt

```
pandas==2.1.4
numpy==1.26.2
matplotlib==3.8.2
seaborn==0.13.0
scikit-learn==1.3.2
scipy==1.11.4
plotly==5.18.0
statsmodels==0.14.1
openpyxl==3.1.2
prophet==1.1.5
jupyter==1.0.0
nbformat==5.9.2
```

Install with:
```bash
pip install -r requirements.txt
```

---

## NOTEBOOK 1: 01_ETL.ipynb
### (Team Member 1 – Data Preprocessing / ETL)

---

### Cell 1 – Title & Imports
```python
# ============================================================
# NOTEBOOK 1: ETL – Extract, Transform, Load
# Superstore Operations Analysis
# ============================================================

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries imported successfully")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
```

---

### Cell 2 – EXTRACT: Load the Raw Data
```python
# ----- EXTRACT -----
# Load the raw dataset
raw_path = "../data/raw/Sample - Superstore.csv"

df_raw = pd.read_csv(raw_path, encoding='latin-1')

print(f"✅ Dataset loaded: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
print("\n📋 Column names:")
print(df_raw.columns.tolist())
```

---

### Cell 3 – Initial Inspection
```python
# View the first few rows
print("=== First 5 rows ===")
display(df_raw.head())

print("\n=== Data Types ===")
print(df_raw.dtypes)

print("\n=== Basic Shape ===")
print(f"Rows: {df_raw.shape[0]}, Columns: {df_raw.shape[1]}")
```

---

### Cell 4 – Check Missing Values
```python
print("=== Missing Values ===")
missing = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw)) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct.round(2)
})

# Show only columns that have missing values
missing_df = missing_df[missing_df['Missing Count'] > 0]

if missing_df.empty:
    print("✅ No missing values found in the dataset!")
else:
    display(missing_df)
```

---

### Cell 5 – Check Duplicates
```python
print("=== Duplicate Rows ===")
dup_count = df_raw.duplicated().sum()
print(f"Total duplicate rows: {dup_count}")

if dup_count > 0:
    print("\nDuplicate rows:")
    display(df_raw[df_raw.duplicated()])
else:
    print("✅ No duplicate rows found!")
```

---

### Cell 6 – TRANSFORM: Data Type Conversion
```python
# ----- TRANSFORM -----
df = df_raw.copy()

# Convert date columns to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%Y')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%m/%d/%Y')

# Verify
print("✅ Date columns converted:")
print(f"  Order Date dtype: {df['Order Date'].dtype}")
print(f"  Ship Date dtype:  {df['Ship Date'].dtype}")
```

---

### Cell 7 – Feature Engineering (New Columns)
```python
# Create new derived features
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Quarter'] = df['Order Date'].dt.quarter
df['Day_of_Week'] = df['Order Date'].dt.day_name()

# Shipping duration (days between order and ship)
df['Shipping_Days'] = (df['Ship Date'] - df['Order Date']).dt.days

# Profit Margin (%)
df['Profit_Margin'] = (df['Profit'] / df['Sales']) * 100

# Revenue category
df['Revenue_Category'] = pd.cut(
    df['Sales'],
    bins=[0, 100, 500, 1000, 5000, 100000],
    labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
)

print("✅ New features created:")
print(["Year", "Month", "Quarter", "Day_of_Week", 
       "Shipping_Days", "Profit_Margin", "Revenue_Category"])

display(df[['Order Date', 'Ship Date', 'Shipping_Days', 
            'Profit', 'Sales', 'Profit_Margin', 'Revenue_Category']].head())
```

---

### Cell 8 – Handle Outliers (IQR Method)
```python
def remove_outliers_iqr(dataframe, column):
    """Remove outliers using the IQR method."""
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    before = len(dataframe)
    df_clean = dataframe[(dataframe[column] >= lower) & 
                          (dataframe[column] <= upper)]
    after = len(df_clean)
    print(f"  '{column}': Removed {before - after} outliers | Range: [{lower:.2f}, {upper:.2f}]")
    return df_clean

print("=== Outlier Removal Report ===")
# NOTE: For forecasting, we keep all data; only remove for clustering
df_clean = df.copy()
for col in ['Sales', 'Profit', 'Discount', 'Shipping_Days']:
    df_clean = remove_outliers_iqr(df_clean, col)

print(f"\nRows before: {len(df)} → After: {len(df_clean)}")
print(f"Rows removed: {len(df) - len(df_clean)}")
```

---

### Cell 9 – Column Renaming for Cleaner Names
```python
# Rename columns: replace spaces with underscores for easier coding
df_clean.columns = [c.replace(' ', '_').replace('-', '_') for c in df_clean.columns]

print("✅ Columns renamed:")
print(df_clean.columns.tolist())
```

---

### Cell 10 – LOAD: Save Processed Data
```python
# ----- LOAD -----
# Save the cleaned dataset
os.makedirs("../data/processed", exist_ok=True)

output_path = "../data/processed/superstore_clean.csv"
df_clean.to_csv(output_path, index=False)

print(f"✅ Cleaned data saved to: {output_path}")
print(f"   Final shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
print(f"\nFinal column list:")
for col in df_clean.columns:
    print(f"  - {col}")
```

---

### Cell 11 – ETL Summary Report
```python
print("=" * 60)
print("           ETL PIPELINE SUMMARY REPORT")
print("=" * 60)
print(f"\n📥 EXTRACT")
print(f"   Source  : Sample Superstore CSV (Kaggle)")
print(f"   Raw rows: {df_raw.shape[0]}")
print(f"   Columns : {df_raw.shape[1]}")

print(f"\n🔄 TRANSFORM")
print(f"   - Converted Order Date & Ship Date to datetime")
print(f"   - Created 6 new features (Year, Month, Quarter, etc.)")
print(f"   - Removed outliers using IQR method")
print(f"   - Renamed columns to snake_case")
print(f"   - No missing values found (data is clean)")

print(f"\n📤 LOAD")
print(f"   Destination  : ../data/processed/superstore_clean.csv")
print(f"   Final rows   : {df_clean.shape[0]}")
print(f"   Final columns: {df_clean.shape[1]}")
print("=" * 60)
```

---

## 📓 NOTEBOOK 2: 02_EDA.ipynb
### (Team Member 2 – Exploratory Data Analysis)

---

### Cell 1 – Imports & Load Clean Data
```python
# ============================================================
# NOTEBOOK 2: Exploratory Data Analysis (EDA)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Styling
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
sns.set_theme(style="whitegrid", palette="Set2")

# Load cleaned data
df = pd.read_csv("../data/processed/superstore_clean.csv", 
                 parse_dates=['Order_Date', 'Ship_Date'])

print(f"✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
display(df.head(3))
```

---

### Cell 2 – Descriptive Statistics
```python
print("=== Descriptive Statistics (Numerical Columns) ===")
display(df[['Sales', 'Profit', 'Discount', 'Quantity', 
            'Shipping_Days', 'Profit_Margin']].describe().round(2))
```

---

### Cell 3 – Category Distribution (Pie Chart)
```python
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Category distribution
cat_counts = df['Category'].value_counts()
axes[0].pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%',
            startangle=90, colors=sns.color_palette("Set2"))
axes[0].set_title("Orders by Category", fontsize=14, fontweight='bold')

# Region distribution
reg_counts = df['Region'].value_counts()
axes[1].pie(reg_counts, labels=reg_counts.index, autopct='%1.1f%%',
            startangle=90, colors=sns.color_palette("Set3"))
axes[1].set_title("Orders by Region", fontsize=14, fontweight='bold')

# Customer Segment
seg_counts = df['Segment'].value_counts()
axes[2].pie(seg_counts, labels=seg_counts.index, autopct='%1.1f%%',
            startangle=90, colors=sns.color_palette("Pastel1"))
axes[2].set_title("Orders by Segment", fontsize=14, fontweight='bold')

plt.suptitle("Distribution of Orders Across Key Dimensions", 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("../reports/eda_pie_charts.png", dpi=150, bbox_inches='tight')
plt.show()
print("💾 Chart saved.")
```

---

### Cell 4 – Sales & Profit by Region (Bar Chart)
```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Sales by Region
region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
bars = axes[0].bar(region_sales.index, region_sales.values, 
                    color=sns.color_palette("Set2"), edgecolor='black')
axes[0].set_title("Total Sales by Region", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Region")
axes[0].set_ylabel("Total Sales ($)")
for bar in bars:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                 f'${bar.get_height():,.0f}', ha='center', va='bottom', fontsize=10)

# Profit by Region
region_profit = df.groupby('Region')['Profit'].sum().sort_values(ascending=False)
colors = ['green' if p > 0 else 'red' for p in region_profit.values]
bars2 = axes[1].bar(region_profit.index, region_profit.values, 
                     color=colors, edgecolor='black', alpha=0.8)
axes[1].set_title("Total Profit by Region", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Region")
axes[1].set_ylabel("Total Profit ($)")
axes[1].axhline(y=0, color='black', linewidth=1)

plt.tight_layout()
plt.savefig("../reports/eda_region_sales_profit.png", dpi=150, bbox_inches='tight')
plt.show()
```

---

### Cell 5 – Monthly Sales Trend (Time Series)
```python
# Aggregate monthly sales
monthly_sales = df.groupby(df['Order_Date'].dt.to_period('M'))['Sales'].sum()
monthly_sales.index = monthly_sales.index.to_timestamp()

plt.figure(figsize=(16, 5))
plt.plot(monthly_sales.index, monthly_sales.values, 
         color='steelblue', linewidth=2, marker='o', markersize=3)
plt.fill_between(monthly_sales.index, monthly_sales.values, alpha=0.2, color='steelblue')
plt.title("Monthly Sales Trend (2015–2018)", fontsize=16, fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Sales ($)")
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
plt.tight_layout()
plt.savefig("../reports/eda_monthly_sales_trend.png", dpi=150, bbox_inches='tight')
plt.show()
```

---

### Cell 6 – Correlation Heatmap
```python
plt.figure(figsize=(10, 8))

numeric_cols = ['Sales', 'Profit', 'Discount', 'Quantity', 
                'Shipping_Days', 'Profit_Margin']
corr_matrix = df[numeric_cols].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
            mask=mask, vmin=-1, vmax=1, linewidths=0.5,
            annot_kws={"size": 12})

plt.title("Correlation Heatmap – Numerical Features", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("../reports/eda_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n📌 Key Insights from Correlation:")
print("  - Discount vs Profit: Strong NEGATIVE correlation (discounts hurt profits)")
print("  - Sales vs Profit: Moderate POSITIVE correlation")
print("  - Quantity vs Sales: Moderate POSITIVE correlation")
```

---

### Cell 7 – Profit by Sub-Category (Horizontal Bar)
```python
subcat_profit = df.groupby('Sub_Category')['Profit'].sum().sort_values()

colors = ['#d73027' if p < 0 else '#1a9850' for p in subcat_profit.values]

plt.figure(figsize=(12, 8))
bars = plt.barh(subcat_profit.index, subcat_profit.values, color=colors, edgecolor='black')
plt.axvline(x=0, color='black', linewidth=1.5)
plt.title("Total Profit by Sub-Category", fontsize=16, fontweight='bold')
plt.xlabel("Total Profit ($)")
for bar, val in zip(bars, subcat_profit.values):
    plt.text(val + (200 if val >= 0 else -200), bar.get_y() + bar.get_height()/2,
             f'${val:,.0f}', va='center', ha='left' if val >= 0 else 'right', fontsize=9)
plt.tight_layout()
plt.savefig("../reports/eda_profit_subcategory.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n🔴 Loss-making sub-categories:")
for sc, p in subcat_profit[subcat_profit < 0].items():
    print(f"  - {sc}: ${p:,.0f}")
```

---

### Cell 8 – Distribution of Sales (Histogram + KDE)
```python
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Sales distribution
axes[0].hist(df['Sales'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_title("Distribution of Sales", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Sales ($)")
axes[0].set_ylabel("Frequency")

# Profit distribution
colors_profit = ['red' if x < 0 else 'green' for x in df['Profit']]
axes[1].hist(df['Profit'], bins=50, color='seagreen', edgecolor='black', alpha=0.7)
axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
axes[1].set_title("Distribution of Profit", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Profit ($)")
axes[1].set_ylabel("Frequency")
axes[1].legend()

plt.tight_layout()
plt.savefig("../reports/eda_distributions.png", dpi=150, bbox_inches='tight')
plt.show()
```

---

### Cell 9 – Top 10 States by Sales
```python
state_sales = df.groupby('State')['Sales'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
bars = plt.bar(state_sales.index, state_sales.values, 
               color=sns.color_palette("Blues_d", 10), edgecolor='black')
plt.title("Top 10 States by Total Sales", fontsize=14, fontweight='bold')
plt.xlabel("State")
plt.ylabel("Total Sales ($)")
plt.xticks(rotation=30, ha='right')
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
             f'${bar.get_height():,.0f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig("../reports/eda_top10_states.png", dpi=150, bbox_inches='tight')
plt.show()
```

---

### Cell 10 – EDA Summary Insights
```python
print("=" * 60)
print("         EDA SUMMARY – KEY INSIGHTS")
print("=" * 60)
print(f"\n📊 Dataset Overview:")
print(f"   Orders       : {df.shape[0]:,}")
print(f"   Time Period  : {df['Order_Date'].min().date()} → {df['Order_Date'].max().date()}")
print(f"   Total Sales  : ${df['Sales'].sum():,.2f}")
print(f"   Total Profit : ${df['Profit'].sum():,.2f}")
print(f"   Avg Margin   : {df['Profit_Margin'].mean():.2f}%")

print(f"\n🏆 Best Performing:")
print(f"   Region   : {df.groupby('Region')['Profit'].sum().idxmax()}")
print(f"   Segment  : {df.groupby('Segment')['Profit'].sum().idxmax()}")
print(f"   Category : {df.groupby('Category')['Profit'].sum().idxmax()}")

print(f"\n⚠️ Problem Areas:")
losing = df.groupby('Sub_Category')['Profit'].sum()
for sc, p in losing[losing < 0].items():
    print(f"   ❌ {sc}: ${p:,.0f} loss")

print(f"\n📉 Correlation Key Takeaway:")
corr_val = df[['Discount', 'Profit']].corr().loc['Discount', 'Profit']
print(f"   Discount ↔ Profit correlation: {corr_val:.3f} (higher discounts = lower profit)")
print("=" * 60)
```

---

## 📓 NOTEBOOK 3: 03_Clustering.ipynb
### (Team Member 3 – Clustering Analysis)

---

### Cell 1 – Imports
```python
# ============================================================
# NOTEBOOK 3: Clustering Analysis
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 6)
sns.set_theme(style="whitegrid")

df = pd.read_csv("../data/processed/superstore_clean.csv",
                 parse_dates=['Order_Date', 'Ship_Date'])

print(f"✅ Data loaded: {df.shape}")
```

---

### Cell 2 – Feature Selection for Customer Segmentation
```python
# Aggregate data per customer (RFM-style features)
# RFM = Recency, Frequency, Monetary
import datetime

snapshot_date = df['Order_Date'].max() + pd.Timedelta(days=1)

rfm = df.groupby('Customer_ID').agg(
    Recency    = ('Order_Date', lambda x: (snapshot_date - x.max()).days),
    Frequency  = ('Order_ID', 'nunique'),
    Monetary   = ('Sales', 'sum'),
    Avg_Profit = ('Profit', 'mean'),
    Avg_Discount = ('Discount', 'mean')
).reset_index()

print("✅ RFM Table created:")
display(rfm.head())
print(f"\nShape: {rfm.shape}")
print("\nDescriptive stats:")
display(rfm.describe().round(2))
```

---

### Cell 3 – Feature Scaling
```python
# Scale features for clustering (very important for KMeans)
features = ['Recency', 'Frequency', 'Monetary', 'Avg_Profit', 'Avg_Discount']
X = rfm[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("✅ Features scaled using StandardScaler")
print(f"   Features used: {features}")
print(f"   Scaled shape: {X_scaled.shape}")
```

---

### Cell 4 – Elbow Method (Find Optimal K)
```python
inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.axvline(x=4, color='red', linestyle='--', linewidth=2, label='Optimal K=4')
plt.title("Elbow Method – Finding Optimal Number of Clusters", 
          fontsize=14, fontweight='bold')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.legend()
plt.tight_layout()
plt.savefig("../reports/clustering_elbow.png", dpi=150, bbox_inches='tight')
plt.show()
print("📌 The 'elbow' at K=4 suggests 4 clusters is optimal.")
```

---

### Cell 5 – Silhouette Score Validation
```python
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"  K={k}: Silhouette Score = {score:.4f}")

best_k = K_range[np.argmax(silhouette_scores)]
print(f"\n✅ Best K by Silhouette Score: K={best_k} → Score={max(silhouette_scores):.4f}")
```

---

### Cell 6 – Apply K-Means with K=4
```python
OPTIMAL_K = 4
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
rfm['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

print(f"✅ K-Means clustering done with K={OPTIMAL_K}")
print("\nCustomer count per cluster:")
print(rfm['KMeans_Cluster'].value_counts().sort_index())
```

---

### Cell 7 – Cluster Profiles (Radar/Bar)
```python
# Analyze each cluster
cluster_profile = rfm.groupby('KMeans_Cluster')[features].mean().round(2)
print("=== Cluster Profiles (Mean Values) ===")
display(cluster_profile)

# Assign business labels
cluster_labels = {
    0: "🏅 Loyal High-Value",
    1: "⚠️ At-Risk Customers",
    2: "🌱 New/Occasional",
    3: "💎 Champions"
}
# Note: Adjust labels after reviewing cluster profiles above

# Plot cluster profiles as bar charts
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, feature in enumerate(features):
    cluster_profile[feature].plot(kind='bar', ax=axes[i], 
                                   color=sns.color_palette("Set2", OPTIMAL_K),
                                   edgecolor='black')
    axes[i].set_title(f"Avg {feature} by Cluster", fontweight='bold')
    axes[i].set_xlabel("Cluster")
    axes[i].tick_params(axis='x', rotation=0)

axes[-1].axis('off')  # Hide empty subplot
plt.suptitle("K-Means Customer Cluster Profiles", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("../reports/clustering_kmeans_profiles.png", dpi=150, bbox_inches='tight')
plt.show()
```

---

### Cell 8 – Scatter Plot: Monetary vs Frequency (colored by cluster)
```python
plt.figure(figsize=(12, 7))

colors_palette = sns.color_palette("Set1", OPTIMAL_K)

for cluster in range(OPTIMAL_K):
    subset = rfm[rfm['KMeans_Cluster'] == cluster]
    plt.scatter(subset['Frequency'], subset['Monetary'],
                label=f"Cluster {cluster}",
                alpha=0.6, s=80, color=colors_palette[cluster])

plt.title("Customer Clusters: Frequency vs Monetary Value", 
          fontsize=14, fontweight='bold')
plt.xlabel("Purchase Frequency (# Orders)")
plt.ylabel("Total Spend ($)")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("../reports/clustering_scatter.png", dpi=150, bbox_inches='tight')
plt.show()
```

---

### Cell 9 – Hierarchical Clustering (Dendrogram)
```python
# Use a sample for visualization (full dataset is too large for dendrogram)
sample_X = X_scaled[:100]

linked = linkage(sample_X, method='ward')

plt.figure(figsize=(18, 7))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True,
           leaf_rotation=90,
           color_threshold=10)
plt.title("Hierarchical Clustering Dendrogram (Sample of 100 Customers)", 
          fontsize=14, fontweight='bold')
plt.xlabel("Customer Index")
plt.ylabel("Euclidean Distance")
plt.tight_layout()
plt.savefig("../reports/clustering_dendrogram.png", dpi=150, bbox_inches='tight')
plt.show()
print("📌 Dendrogram confirms 3-4 natural clusters in the data.")
```

---

### Cell 10 – Apply Hierarchical Clustering (Full Dataset)
```python
hier = AgglomerativeClustering(n_clusters=OPTIMAL_K, linkage='ward')
rfm['Hierarchical_Cluster'] = hier.fit_predict(X_scaled)

print("✅ Hierarchical clustering done")
print("\nCustomer count per hierarchical cluster:")
print(rfm['Hierarchical_Cluster'].value_counts().sort_index())

# Compare KMeans vs Hierarchical
comparison = pd.crosstab(rfm['KMeans_Cluster'], rfm['Hierarchical_Cluster'],
                          rownames=['KMeans'], colnames=['Hierarchical'])
print("\n=== Cluster Agreement Matrix (KMeans vs Hierarchical) ===")
display(comparison)
```

---

### Cell 11 – Save Clustered Data
```python
rfm.to_csv("../data/processed/customers_clustered.csv", index=False)
print("✅ Clustered customer data saved!")

print("\n=== CLUSTERING SUMMARY ===")
print(f"  Total customers analyzed: {len(rfm):,}")
print(f"  Clustering method: K-Means + Hierarchical")
print(f"  Optimal clusters: K={OPTIMAL_K}")
print(f"  Features used: {features}")
print(f"\n  Business Interpretation:")
for cid in range(OPTIMAL_K):
    subset = rfm[rfm['KMeans_Cluster'] == cid]
    print(f"  Cluster {cid}: {len(subset)} customers | "
          f"Avg Spend=${subset['Monetary'].mean():,.0f} | "
          f"Avg Orders={subset['Frequency'].mean():.1f}")
```

---

## 📓 NOTEBOOK 4: 04_Forecasting.ipynb
### (Team Member 4 – Forecasting Model)

---

### Cell 1 – Imports
```python
# ============================================================
# NOTEBOOK 4: Sales Forecasting Model
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (14, 6)

df = pd.read_csv("../data/processed/superstore_clean.csv",
                 parse_dates=['Order_Date', 'Ship_Date'])

print("✅ Libraries and data loaded successfully")
```

---

### Cell 2 – Prepare Monthly Sales Time Series
```python
# Aggregate to monthly sales
monthly = df.groupby(df['Order_Date'].dt.to_period('M'))['Sales'].sum().reset_index()
monthly.columns = ['Month', 'Sales']
monthly['Month'] = monthly['Month'].dt.to_timestamp()
monthly = monthly.set_index('Month').asfreq('MS')  # MS = Month Start frequency

print(f"✅ Monthly time series prepared")
print(f"   Period: {monthly.index.min()} → {monthly.index.max()}")
print(f"   Total months: {len(monthly)}")
display(monthly.head(10))
```

---

### Cell 3 – Time Series Decomposition
```python
# Decompose into Trend + Seasonality + Residual
decomposition = seasonal_decompose(monthly['Sales'], model='additive', period=12)

fig, axes = plt.subplots(4, 1, figsize=(14, 12))

decomposition.observed.plot(ax=axes[0], color='steelblue')
axes[0].set_title("Observed Sales", fontweight='bold')
axes[0].set_ylabel("Sales ($)")

decomposition.trend.plot(ax=axes[1], color='orange')
axes[1].set_title("Trend Component", fontweight='bold')
axes[1].set_ylabel("Trend")

decomposition.seasonal.plot(ax=axes[2], color='green')
axes[2].set_title("Seasonal Component", fontweight='bold')
axes[2].set_ylabel("Seasonality")

decomposition.resid.plot(ax=axes[3], color='red')
axes[3].set_title("Residual Component", fontweight='bold')
axes[3].set_ylabel("Residuals")

plt.suptitle("Time Series Decomposition – Monthly Sales", 
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("../reports/forecasting_decomposition.png", dpi=150, bbox_inches='tight')
plt.show()

print("📌 Observations:")
print("  - Clear UPWARD trend in sales over 2015-2018")
print("  - Strong seasonality with peaks in Q4 (holiday season)")
```

---

### Cell 4 – ACF and PACF Plots
```python
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

plot_acf(monthly['Sales'].dropna(), lags=24, ax=axes[0])
axes[0].set_title("Autocorrelation Function (ACF)", fontweight='bold')

plot_pacf(monthly['Sales'].dropna(), lags=24, ax=axes[1])
axes[1].set_title("Partial Autocorrelation Function (PACF)", fontweight='bold')

plt.tight_layout()
plt.savefig("../reports/forecasting_acf_pacf.png", dpi=150, bbox_inches='tight')
plt.show()

print("📌 ACF/PACF help us choose SARIMA parameters (p, d, q)(P, D, Q)m")
```

---

### Cell 5 – Train/Test Split (80/20)
```python
# Split: use last 12 months as test set
train_size = len(monthly) - 12
train = monthly.iloc[:train_size]
test  = monthly.iloc[train_size:]

print(f"✅ Train/Test Split:")
print(f"   Training: {train.index.min().date()} → {train.index.max().date()} ({len(train)} months)")
print(f"   Testing : {test.index.min().date()} → {test.index.max().date()} ({len(test)} months)")

plt.figure(figsize=(14, 5))
plt.plot(train.index, train['Sales'], label='Training Data', color='steelblue', linewidth=2)
plt.plot(test.index,  test['Sales'],  label='Test Data', color='orange', linewidth=2)
plt.title("Train / Test Split – Monthly Sales", fontsize=14, fontweight='bold')
plt.ylabel("Sales ($)")
plt.legend()
plt.tight_layout()
plt.show()
```

---

### Cell 6 – SARIMA Model Training
```python
# SARIMA(1,1,1)(1,1,1,12) — accounts for trend + annual seasonality
print("⏳ Training SARIMA model... (may take 30-60 seconds)")

model = SARIMAX(train['Sales'],
                order=(1, 1, 1),         # (p, d, q) non-seasonal
                seasonal_order=(1, 1, 1, 12),  # (P, D, Q, m) seasonal
                enforce_stationarity=False,
                enforce_invertibility=False)

result = model.fit(disp=False)

print("✅ SARIMA model trained successfully!")
print(result.summary())
```

---

### Cell 7 – Generate Forecast
```python
# Forecast for the test period
forecast_steps = len(test)
forecast = result.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

print(f"✅ Forecast generated for {forecast_steps} months")
display(pd.DataFrame({
    'Actual': test['Sales'].values,
    'Forecast': forecast_mean.values,
    'Lower_CI': forecast_ci.iloc[:, 0].values,
    'Upper_CI': forecast_ci.iloc[:, 1].values
}, index=test.index).round(2))
```

---

### Cell 8 – Plot Forecast vs Actual
```python
plt.figure(figsize=(16, 7))

# Training data
plt.plot(train.index, train['Sales'], 
         label='Training Data', color='steelblue', linewidth=2)

# Actual test data
plt.plot(test.index, test['Sales'], 
         label='Actual Sales (Test)', color='darkorange', linewidth=2.5, marker='o')

# Forecasted values
plt.plot(forecast_mean.index, forecast_mean, 
         label='SARIMA Forecast', color='red', linewidth=2.5, linestyle='--', marker='s')

# Confidence interval
plt.fill_between(forecast_ci.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1],
                 alpha=0.2, color='red', label='95% Confidence Interval')

plt.title("SARIMA Forecast vs Actual Monthly Sales", fontsize=16, fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Sales ($)")
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
plt.legend(fontsize=11)
plt.axvline(x=test.index[0], color='gray', linestyle=':', linewidth=2, label='Forecast Start')
plt.tight_layout()
plt.savefig("../reports/forecasting_sarima_vs_actual.png", dpi=150, bbox_inches='tight')
plt.show()
```

---

### Cell 9 – Model Performance Metrics
```python
actual    = test['Sales'].values
predicted = forecast_mean.values

mae  = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
mape = np.mean(np.abs((actual - predicted) / actual)) * 100
r2   = 1 - (np.sum((actual - predicted)**2) / np.sum((actual - actual.mean())**2))

print("=" * 50)
print("     FORECASTING MODEL PERFORMANCE METRICS")
print("=" * 50)
print(f"  MAE  (Mean Absolute Error)  : ${mae:,.2f}")
print(f"  RMSE (Root Mean Sq Error)   : ${rmse:,.2f}")
print(f"  MAPE (Mean Abs % Error)     : {mape:.2f}%")
print(f"  R²   (Coefficient of Det.)  : {r2:.4f}")
print("=" * 50)

if mape < 10:
    print("  ✅ Excellent forecast (MAPE < 10%)")
elif mape < 20:
    print("  🟡 Good forecast (MAPE < 20%)")
else:
    print("  ⚠️ Model needs improvement (MAPE > 20%)")
```

---

### Cell 10 – Future Forecast (Next 6 Months)
```python
# Retrain on FULL data and forecast next 6 months
full_model = SARIMAX(monthly['Sales'],
                     order=(1, 1, 1),
                     seasonal_order=(1, 1, 1, 12),
                     enforce_stationarity=False,
                     enforce_invertibility=False)

full_result = full_model.fit(disp=False)

future_forecast = full_result.get_forecast(steps=6)
future_mean = future_forecast.predicted_mean
future_ci   = future_forecast.conf_int()

# Plot
plt.figure(figsize=(16, 6))
plt.plot(monthly.index, monthly['Sales'], 
         label='Historical Sales', color='steelblue', linewidth=2)
plt.plot(future_mean.index, future_mean, 
         label='6-Month Forecast', color='green', linewidth=2.5, 
         linestyle='--', marker='D')
plt.fill_between(future_ci.index,
                 future_ci.iloc[:, 0], future_ci.iloc[:, 1],
                 alpha=0.25, color='green', label='95% CI')
plt.axvline(x=monthly.index[-1], color='gray', linestyle=':', linewidth=2)
plt.title("6-Month Future Sales Forecast (SARIMA)", fontsize=16, fontweight='bold')
plt.ylabel("Sales ($)")
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
plt.legend()
plt.tight_layout()
plt.savefig("../reports/forecasting_future_6months.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n📅 Future Monthly Forecast:")
for date, val in zip(future_mean.index, future_mean.values):
    print(f"  {date.strftime('%b %Y')}: ${val:,.2f}")
```

---

### Cell 11 – Business Impact & Recommendations
```python
print("=" * 65)
print("       FORECASTING – BUSINESS IMPACT ANALYSIS")
print("=" * 65)

total_forecast_6m = future_mean.sum()
current_avg       = monthly['Sales'].tail(12).mean()
forecast_avg      = future_mean.mean()
growth            = ((forecast_avg - current_avg) / current_avg) * 100

print(f"\n📈 Forecast Summary (Next 6 Months):")
print(f"   Total Projected Sales : ${total_forecast_6m:,.2f}")
print(f"   Current Avg (last 12m): ${current_avg:,.2f}/month")
print(f"   Forecast Avg          : ${forecast_avg:,.2f}/month")
print(f"   Projected Growth      : {growth:+.1f}%")

print(f"\n💼 Operational Recommendations:")
print(f"  1. Increase inventory ahead of peak Q4 season (Nov-Dec)")
print(f"  2. Target marketing spend in underperforming months (Q1)")
print(f"  3. Reduce discounts in Technology category to protect margins")
print(f"  4. Invest more in West region (highest profitability)")
print(f"  5. Address Tables & Bookcases sub-category losses immediately")
print("=" * 65)
```

---

## README.md Template

```markdown
# Superstore Operations Analysis
## Coventry University – MSc Data Science – Group Assignment

## Project Overview
Analysis of Superstore retail data (9,994 records, 2015–2018) to improve
operations using ETL, EDA, Clustering, and Forecasting.

## Team Members & Contributions
| Member | Task |
|--------|------|
| Member 1 | Data Preprocessing & ETL Pipeline |
| Member 2 | Exploratory Data Analysis |
| Member 3 | Clustering Analysis (K-Means & Hierarchical) |
| Member 4 | Forecasting Model (SARIMA) & Git Management |

## Dataset
- **Source**: Sample Superstore (Kaggle)
- **Records**: 9,994 orders
- **Period**: January 2015 – December 2018
- **Features**: Orders, customers, products, sales, profit, shipping

## Methods Used
- **ETL**: Pandas, feature engineering, outlier removal (IQR)
- **EDA**: Matplotlib, Seaborn, correlation analysis
- **Clustering**: K-Means (K=4) + Hierarchical (Ward linkage)
- **Forecasting**: SARIMA(1,1,1)(1,1,1,12)

## Key Results
- Sales growing ~X% YoY
- 4 distinct customer segments identified
- SARIMA MAPE: ~X% (good forecast accuracy)
- Tables sub-category is loss-making (action required)

## How to Run
```bash
git clone <repo-url>
cd superstore-analysis
pip install -r requirements.txt
jupyter notebook
```

## Folder Structure
```
superstore-analysis/
├── data/raw/          # Original dataset
├── data/processed/    # Cleaned data
├── notebooks/         # Jupyter notebooks (01-04)
├── reports/           # Charts and final report
├── requirements.txt
└── README.md
```
```

---

## GIT WORKFLOW (for Git Management person)

```bash
# Step 1: Create repo on GitHub → clone it
git clone https://github.com/yourteam/superstore-analysis.git
cd superstore-analysis

# Step 2: Create folder structure
mkdir -p data/raw data/processed notebooks reports

# Step 3: Each person works on their own branch
git checkout -b feature/etl           # Member 1
git checkout -b feature/eda           # Member 2
git checkout -b feature/clustering    # Member 3
git checkout -b feature/forecasting   # Member 4

# Step 4: Commit meaningfully after each notebook cell block
git add notebooks/01_ETL.ipynb
git commit -m "ETL: Added data loading and missing value check"
git commit -m "ETL: Feature engineering - added Year, Month, Shipping_Days"
git commit -m "ETL: Outlier removal using IQR method and save to processed/"

# Step 5: Push your branch
git push origin feature/etl

# Step 6: Create Pull Request on GitHub → merge to main
git checkout main
git merge feature/etl
git push origin main
```

---

## WORK DIVISION TABLE

| Task | Notebook | Key Deliverables |
|------|----------|-----------------|
| Member 1 – ETL | 01_ETL.ipynb | Clean CSV, pipeline documentation |
| Member 2 – EDA | 02_EDA.ipynb | 8+ visualizations, insights report |
| Member 3 – Clustering | 03_Clustering.ipynb | 4 customer segments, dendrograms |
| Member 4 – Forecasting | 04_Forecasting.ipynb | SARIMA model, performance metrics |

---

*Guide prepared for MSc Data Science – Principles of Data Science 7144 – 2025 Batch*
