import pandas as pd
import matplotlib.pyplot as plt
import ast
import os

# Load the datasets
breast_annotations_path = "vindr/1.0.0/breast-level_annotations.csv"
finding_annotations_path = "vindr/1.0.0/finding_annotations.csv"

os.makedirs("data_analysis", exist_ok=True)

breast_annotations_df = pd.read_csv(breast_annotations_path)
finding_annotations_df = pd.read_csv(finding_annotations_path)

def plot_distribution(df, column, title):
    """Plots a bar chart showing the distribution of values in a column with actual counts on bars."""
    value_counts = df[column].value_counts()
    
    plt.figure(figsize=(8, 9))
    ax = value_counts.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add text annotations
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.savefig(f"data_analysis/{column}_distribution.png")

def analyze_data_balances(breast_df, finding_df):
    """Analyzes and plots the distributions of breast_birads, breast_density, and finding_categories."""
    
    # Finding BIRADS distribution
    plot_distribution(finding_df, "breast_birads", "Finding BIRADS Distribution") #
    
    # Breast Density in Finding dataset
    plot_distribution(finding_df, "breast_density", "Breast Density Distribution in Findings") #

    # Finding Categories distribution
    finding_categories_flattened = []
    for row in finding_df["finding_categories"].dropna():
        categories = ast.literal_eval(row)  # Convert string list to actual list
        finding_categories_flattened.extend(categories)

    finding_categories_series = pd.Series(finding_categories_flattened)
    
    plt.figure(figsize=(12, 6))
    ax = finding_categories_series.value_counts().plot(kind="bar", color="coral", edgecolor="black")
    plt.xlabel("Finding Category")
    plt.ylabel("Count")
    plt.title("Finding Categories Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Add text annotations
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.savefig("data_analysis/finding_categories_distribution.png")
# Run the analysis
analyze_data_balances(breast_annotations_df, finding_annotations_df)
