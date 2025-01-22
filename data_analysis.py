from collections import Counter

def analyze_birads_and_findings(df: pd.DataFrame):
    """
    Analyze the dataset for BI-RADS score distribution and finding categories.

    Args:
        df (pd.DataFrame): DataFrame containing 'breast_birads' and 'finding_categories' columns.

    Returns:
        Tuple of dictionaries: (birads_counts, finding_category_counts)
    """
    # Count occurrences of each BI-RADS score
    birads_counts = df['breast_birads'].value_counts().to_dict()

    # Process finding categories: Remove brackets, split on commas, and count occurrences
    all_findings = []
    for entry in df['finding_categories']:
        # Convert string list format to an actual list of categories
        cleaned_entry = entry.replace("[", "").replace("]", "").replace("'", "").strip()
        categories = [cat.strip() for cat in cleaned_entry.split(",") if cat.strip()]
        all_findings.extend(categories)

    # Count occurrences of each finding category
    finding_category_counts = dict(Counter(all_findings))

    return birads_counts, finding_category_counts

# Run the function on the dataset
birads_counts, finding_category_counts = analyze_birads_and_findings(df)

# Display the results
import ace_tools as tools

# Convert to DataFrame for better visualization
birads_df = pd.DataFrame(list(birads_counts.items()), columns=["BI-RADS Score", "Count"])
findings_df = pd.DataFrame(list(finding_category_counts.items()), columns=["Finding Category", "Count"])

# Show results
tools.display_dataframe_to_user(name="BI-RADS Distribution", dataframe=birads_df)
tools.display_dataframe_to_user(name="Finding Categories Distribution", dataframe=findings_df)
