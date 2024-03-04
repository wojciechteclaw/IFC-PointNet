import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO


# Assuming 'your_file_path.csv' is the path to your CSV file
# file_path = 'your_file_path.csv'
file_path = r"C:\Code\IFC-extracted_elements_dataset\data_stats.csv"

# Read the CSV data into a pandas DataFrame
# df = pd.read_csv(file_path, encoding='utf-8-sig')

try:
    # Try reading with UTF-8 encoding first
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    try:
        # If UTF-8 fails, try with Latin1
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        # As a last resort, try reading with UTF-8-sig in case of BOM
        df = pd.read_csv(file_path, encoding='utf-8-sig', errors='replace')

# Count the occurrences of each label
label_counts = df['Label'].value_counts()

# Calculate the percentage of each label
label_percentages = (label_counts / label_counts.sum()) * 100

# Filter labels to keep those with more than 1% and aggregate the rest into 'Other'
main_labels = label_percentages[label_percentages > 1]
other_percentage = label_percentages[label_percentages <= 1].sum()
if other_percentage > 0:
    main_labels['Other'] = other_percentage

# Determine labels for pie chart slices
labels_for_pie = main_labels.index
# For slices smaller than 1%, we'll not show the label next to the chart, but in a legend
slice_labels = [label if percentage > 1 else '' for label, percentage in zip(main_labels.index, main_labels)]


# Aggregate data for the bar chart (average vertices count per label)
average_vertices_per_label = df.groupby('Label')['Vertices Count'].mean()

# Sort the labels from the one with the most to the least vertices on average
average_vertices_per_label_sorted = average_vertices_per_label.sort_values(ascending=False)

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(20, 8))

# Pie chart
# axs[0].pie(main_labels, labels=main_labels.index, autopct='%1.1f%%', startangle=140)
axs[0].pie(main_labels, labels=slice_labels, autopct='%1.1f%%', startangle=35,
                                   pctdistance=0.85, rotatelabels=True)

# axs[0].set_title('Distribution of IFC entities')

# axs[0].set_title('Distribution of Labels (Pie Chart)')
axs[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# # Improve legibility
# plt.setp(autotexts, size=8, color="white")


# Bar chart for average vertices count, sorted and logarithmic scale
average_vertices_per_label_sorted.plot(kind='bar', ax=axs[1], color='skyblue')
axs[1].set_title('Average Vertices Count per IFC Entity (Log Scale)')
# axs[1].set_xlabel('Label')
# axs[1].set_ylabel('Average Vertices Count')
axs[1].set_yscale('log')  # Set the y-axis to a logarithmic scale
axs[1].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()