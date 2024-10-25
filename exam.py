import pandas as pd
import matplotlib.pyplot as plt
from pyrolite.util.classification import TAS
from scipy.stats import linregress
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as ss
import pycodamath as coda
# Load the data into df_clean DataFrame

df_clean = pd.read_csv("C:/Users/konuc/Downloads/basalts.csv", delimiter=';')
df = df_clean.copy()

# Calculate the sum of values for each row and add it as a new column 'RowSum'
df['RowSum'] = df_clean.iloc[:, 1:10].sum(axis=1)

# Divide columns 1 to 9 by the scaling factors and then multiply by the scaled value
scaling_factors = 100 / df['RowSum']
df.iloc[:, 1:10] = df.iloc[:, 1:10].div(df['RowSum'], axis=0) * 100

# Plotting TAS vs SiO2 with color-coded TAS fields
df["Na2O + K2O"] = df["Na2O"] + df["K2O"]
cm = TAS()

fig, ax = plt.subplots(1)
cm.add_to_axes(ax, alpha=0.5, linewidth=0.5, zorder=-1, add_labels=True)
df[["SiO2", "Na2O + K2O"]].pyroplot.scatter(ax=ax, c="k", alpha=0.2)
plt.show()

df["TAS"] = cm.predict(df)
df["Rocknames"] = df.TAS.apply(lambda x: cm.fields.get(x, {"name": None})["name"])

fig, ax = plt.subplots(1)
cm.add_to_axes(ax, alpha=0.5, linewidth=0.5, zorder=-1, add_labels=True)
df[["SiO2", "Na2O + K2O"]].pyroplot.scatter(ax=ax, c=df["TAS"], alpha=0.7)

# Linear Regression
df_bs = df[(df["TAS"] == "Bs") & (df["Unnamed: 0"] != "Apollo14")]
df_bs['RowSum'] = df_bs.iloc[:, 1:10].sum(axis=1)
regress_result = linregress(df_bs["SiO2"], df_bs["Na2O + K2O"])

plt.figure(figsize=(10, 8))
plt.scatter(df_bs["SiO2"], df_bs["Na2O + K2O"], c="green", alpha=0.7, label="Bs Samples")
x_vals = df_bs["SiO2"]
y_vals = regress_result.slope * x_vals + regress_result.intercept
plt.plot(x_vals, y_vals, color="blue", label="Linear Regression")
plt.xlabel("SiO2")
plt.ylabel("Na2O + K2O")
plt.title("Linear Regression of 'Bs' Samples (Excluding Apollo14)")
plt.xlim(40, 55)  # Adjust these values based on your data
plt.ylim(0, 5)   # Adjust these values based on your data
plt.legend()
plt.show()

slope = regress_result.slope
intercept = regress_result.intercept
linear_equation = f"Linear Regression Equation: Na2O + K2O = {slope:.2f} * SiO2 + {intercept:.2f}"
print(linear_equation)
print("Linear Regression Result:")
print("Slope:", regress_result.slope)
print("Intercept:", regress_result.intercept)
print("R-squared:", regress_result.rvalue**2)
print("P-value:", regress_result.pvalue)
print("Standard Error:", regress_result.stderr)

apollo14_row_index = df[df["Unnamed: 0"] == 'Apollo14'].index[0]
df.loc[apollo14_row_index, 'Na2O'] = 0.42625

# Final Data Rescaling
df['RowSum'] = df.iloc[:, 1:10].sum(axis=1)
scaling_factors = 100 / df['RowSum']
df.iloc[:, 1:10] = df.iloc[:, 1:10].multiply(scaling_factors, axis=0)
df['RowSum'] = df.iloc[:, 1:10].sum(axis=1)
df["Na2O + K2O"] = df["Na2O"] + df["K2O"]

# Save to Excel
output_path = "C:/Users/konuc/Downloads/final.xlsx"
df.to_excel(output_path, index=False)

# PCA
# Copy "Unnamed: 0" to a new column "Source"
df["Source"] = df["Unnamed: 0"]

# Apply a function to modify the "Source" column
def modify_source(value):
    if value in ["Apollo11", "Apollo12", "Apollo17", "Apollo14", "Apollo15"]:
        return value[:-2]
    else:
        return value[:-1]

df["Source"] = df["Source"].apply(modify_source)

# Print the modified DataFrame
print(df)

data = df.iloc[:, 1:10]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(data=pca_result, columns=["PC1", "PC2"])
result_df = pd.concat([pca_df, df[['Source']]], axis=1)
print(result_df)


fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)

targets = result_df['Source'].unique()  # Assuming 'TAS' is the target column
colors = plt.cm.viridis(np.linspace(0, 1, len(targets)))

for target, color in zip(targets, colors):
    indicesToKeep = result_df['Source'] == target
    ax.scatter(result_df.loc[indicesToKeep, 'PC1'],
               result_df.loc[indicesToKeep, 'PC2'],
               c=[color],
               s=50, label=target)

ax.legend()
ax.grid()
plt.show()

variance_explained = [0.65069444, 0.22709401]
total_variance = np.sum(variance_explained)

print("Total Variance Explained:", total_variance)



#####
import matplotlib.pyplot as plt

# Create a new DataFrame containing the first 10 columns of the original df
c_df = df.iloc[:, :10]
a_df = df.iloc[:,1:10]
# Plot a stacked bar chart
ax = c_df.plot(kind='bar', stacked=True, figsize=(10, 6))
ax.set_xticklabels(c_df['Unnamed: 0'])
plt.xlabel('Samples')
plt.ylabel('wt%')
plt.title('Stacked Bar Plot')
plt.legend(title='Columns')
plt.show()


##########CLR#######
import matplotlib.pyplot as plt
import numpy as np

# Geometric center
gm = ss.mstats.gmean(a_df)
gm = 100/np.sum(gm) * gm
print(gm)
# [49.69085369  2.93730007 13.09668943 11.88004508  0.20830453  9.29919508 10.94728398  1.51903879  0.42128936]
# Variation matrix
npdata = np.array(a_df)  # Convert CLR DataFrame to numpy array
var_matrix = np.var(np.log(npdata[:, :, None] * 1. / npdata[:, None]), axis=0)
print(var_matrix)

totvar = 1./(2 * 9) * np.sum(var_matrix)
print(totvar)

pert_data = a_df/gm
print(pert_data)
scaled_data = pow(a_df, 1./totvar)
print(scaled_data)

clr = scaled_data.coda.clr()
s, e, l = np.linalg.svd(clr)
# scale loadings with eigenvalues
l = np.inner(np.diag(e), l.T)

import matplotlib.pyplot as plt

# Assuming you have the 'l' list and 'clr' dataset defined

# Define a list of colors for arrows
arrow_colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black']

# Create a figure and axis
fig, ax = plt.subplots()

# Iterate through the indices and draw arrows with labels and colors
for i, color in zip([0, 1, 2, 3, 4, 5, 6, 7, 8], arrow_colors):
    ax.arrow(0, 0, l[0][i], l[1][i], width=0.01, label=clr.columns[i], color=color)

# Set labels and legend
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.legend()

# Calculate the percentage of total variance explained by each eigenvalue
total_variance = sum(e)
eigenvalue_percentages = [(eig / total_variance) * 100 for eig in e]

# Calculate the cumulative percentages
cumulative_percentages = np.cumsum(eigenvalue_percentages)

# Create a cumulative percentage plot as a bar chart
plt.figure(figsize=(8, 5))
bars = plt.bar(range(1, len(cumulative_percentages) + 1), cumulative_percentages, color='b')
plt.title('Cumulative Percentage Plot (Bar Chart)')
plt.xlabel('Component')
plt.ylabel('Cumulative Percentage')
plt.xticks(range(1, len(cumulative_percentages) + 1))
plt.grid(True)

# Add cumulative percentages as text labels above each bar
for bar, percentage in zip(bars, cumulative_percentages):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{percentage:.2f}%', ha='center', va='bottom')

plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Define a list of colors for arrows
arrow_colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black']

# Create a figure and axis
fig, ax = plt.subplots()

# Iterate through the indices and draw arrows with labels and colors
for i, color in zip([0, 1, 2, 3, 4, 5, 6, 7, 8], arrow_colors):
    ax.arrow(0, 0, l[0][i], l[1][i], width=0.01, label=clr.columns[i], color=color)

# Set labels and legend for the arrow plot
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.legend()

# Assuming you have 'scores' data for the first 2 principal components in a NumPy array named 's'
# Assuming you have the original DataFrame df containing the labels
labels = df["Unnamed: 0"]  # Assuming "Unnamed: 0" is the label column

# Transpose the 's' array for plotting
s_transposed = np.transpose(s)

# Plot the first 2 principal components as points with labels
scatter = ax.scatter(s_transposed[0], s_transposed[1], color='gray', marker='o')
for i, label in enumerate(labels):
    ax.annotate(label, (s_transposed[0][i], s_transposed[1][i]), fontsize=8, ha='right', va='bottom')
    plt.plot(s.T[0][i], s.T[1][i], marker='o', markersize=5, color='gray')  # Corrected line

# Display the plot
plt.show()
mypca = coda.pca.Biplot(a_df)

import matplotlib.pyplot as plt

# Assuming you have the 'clr' dataset defined
# If not, replace this with your actual dataset

# Get the row labels and column names
row_labels = df["Unnamed: 0"]
column_names = clr.columns

# Create a heatmap of the 'clr' dataset
plt.figure(figsize=(10, 6))
heatmap = plt.imshow(clr, cmap='viridis', aspect='auto')  # Use the colormap of your choice

# Set row and column labels
plt.xticks(range(len(column_names)), column_names, rotation=45)
plt.yticks(range(len(row_labels)), row_labels)

# Set labels and title
plt.title('Heatmap of clr Dataset')
plt.xlabel('Columns')
plt.ylabel('Rows')

# Add a colorbar
plt.colorbar(heatmap)

# Display the plot
plt.show()


import pandas as pd

# Assuming you have the 'a_df' DataFrame defined
# If not, replace this with your actual DataFrame

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import ternary

# List of columns to select
selected_columns = ["CaO", "MgO", "K2O"]

# Create a new DataFrame with selected columns
ten_df = a_df[selected_columns].copy()

# Close to 100: Normalize the data to sum to approximately 100
normalized_df = (ten_df / ten_df.sum(axis=1).values[:, np.newaxis]) * 100
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycodamath as coda
from pycodamath import plot
from pycodamath import extra
import ternary
plot.ternary(normalized_df)
# Center by Perturbing: Subtract the inverse mean
centered_df = normalized_df - 1 / normalized_df.mean()

# CLR Transform: Perform centered log-ratio (CLR) transform
clr_transformed = np.log(centered_df / centered_df.mean())

# Calculate Loadings and Eigenvalues using SVD
U, S, Vt = np.linalg.svd(clr_transformed, full_matrices=False)
loadings = Vt.T
eigenvalues = S ** 2

# Display loadings and eigenvalues
print("Loadings (Principal Components):")
print(loadings)
print("\nEigenvalues:")
print(eigenvalues)

# Calculate the total sum of eigenvalues
total_variance = np.sum(eigenvalues)

# Calculate the variance explained by each component
variance_explained = eigenvalues / total_variance

# Print the variance explained by each component
for i, explained_variance in enumerate(variance_explained):
    print(f"Variance Explained by PC{i+1}: {explained_variance:.2%}")


import matplotlib.pyplot as plt
import pandas as pd

# Assuming you have the 'a_df' and 'df' DataFrames defined
# If not, replace these with your actual DataFrames

# Create a figure and axis
fig, ax = plt.subplots()

# Loop through each sample and plot its data for each oxide
for index, row in a_df.iterrows():
    ax.plot(row, label=df.loc[index, "Unnamed: 0"])

# Add legend
ax.legend()

# Set labels and title
ax.set_xlabel('Oxides')
ax.set_ylabel('Value')  # Update this label as needed
ax.set_title('Data Visualization for Each Sample and Oxide')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# Assuming you have the 'clr' and 'df' DataFrames defined
# If not, replace these with your actual DataFrames

# Create a figure and axes for subplots
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))

# Define the ranges for each subplot
ranges = [(0, 5), (5, 10), (10, 15)]

# Loop through the ranges and create a subplot for each range
for ax, (start, end) in zip(axes, ranges):
    # Extract the subset of rows from clr and df
    subset_clr = clr.iloc[start:end, :]
    subset_df = df.iloc[start:end, :]

    # Loop through each sample and plot its data for each oxide in the subplot
    for index, row in subset_clr.iterrows():
        ax.plot(row, label=subset_df.loc[index, "Unnamed: 0"])

    # Set title and labels for each subplot
    ax.set_title(f'Samples {start+1} to {end}')
    ax.set_xlabel('Oxides')
    ax.set_ylabel('Value')  # Update this label as needed

    # Rotate x-axis labels for better readability
    ax.set_xticklabels(clr.columns, rotation=45)

    # Add legend to each subplot
    ax.legend()

# Adjust layout and display the subplots
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# Assuming you have the 'clr' and 'df' DataFrames defined
# If not, replace these with your actual DataFrames

# Define the ranges for each subset of rows
ranges = [(0, 5), (5, 10), (10, 15)]

# Loop through the ranges and create a separate graph for each range
for start, end in ranges:
    # Create a new figure and axis for each graph
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract the subset of rows from clr and df
    subset_clr = clr.iloc[start:end, :]
    subset_df = df.iloc[start:end, :]
    
    # Loop through each sample and plot its data for each oxide in the graph
    for index, row in subset_clr.iterrows():
        ax.plot(row, label=subset_df.loc[index, "Unnamed: 0"])
    
    # Set title and labels for the graph
    ax.set_title(f'Samples {start+1} to {end}')
    ax.set_xlabel('Oxides')
    ax.set_ylabel('Value')  # Update this label as needed
    
    # Rotate x-axis labels for better readability
    ax.set_xticklabels(clr.columns, rotation=45)
    
    # Add legend to the graph
    ax.legend()
    
    # Adjust layout and display the graph
    plt.tight_layout()
    plt.show()
