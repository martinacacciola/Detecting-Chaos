import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression

inputs = np.genfromtxt('./data/all_X.csv', delimiter=',')
outputs = np.genfromtxt('./data/all_y.csv', delimiter=',')

# Combine inputs and outputs
#combined_data = np.hstack((inputs, outputs))

# Create column names for the combined data
input_columns = [f'Particle {i//6 + 1} {["X Pos", "Y Pos", "Z Pos", "X Vel", "Y Vel", "Z Vel"][i % 6]}' 
                 for i in range(inputs.shape[1])]
output_columns = [f'Gaussian {i//4 + 1} {["Mean", "Std", "Weight", "Height"][i % 4]}' 
                  for i in range(outputs.shape[1])]
columns = input_columns + output_columns


# Handle invalid values by removing rows with NaNs or infinities#df.replace([np.inf, -np.inf], np.nan, inplace=True)
#df.dropna(inplace=True)

# Separate inputs and outputs again (clean)
#inputs = df[input_columns].values
#outputs = df[output_columns].values

# Compute correlations between inputs and outputs
input_output_corr = pd.DataFrame(
    np.corrcoef(inputs, outputs, rowvar=False)[:inputs.shape[1], inputs.shape[1]:],
    index=input_columns,
    columns=output_columns
)

""" # Filter velocity columns
velocity_columns = [col for col in input_columns if 'Vel' in col]
velocity_corr = input_output_corr.loc[velocity_columns]

# Print the correlation values between the velocities columns and the outputs
print("Correlation values between velocity columns and outputs:")
print(velocity_corr) """

# Plot the reduced correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(input_output_corr, annot=True, fmt=".4f", cmap='coolwarm', cbar=True, 
            xticklabels=output_columns, yticklabels=input_columns, annot_kws={"size": 8})
plt.title('Correlation Between Inputs and Outputs')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()  
plt.savefig('correlation_matrix_inputs_vs_outputs.png')
plt.show()


# Compute F-scores and p-values for each output column separately
f_scores_dict = {}
p_values_dict = {}

for i, output_column in enumerate(output_columns):
    f_scores, p_values = f_regression(inputs, outputs[:, i])
    f_scores_dict[output_column] = f_scores
    p_values_dict[output_column] = p_values

f_scores_df = pd.DataFrame(f_scores_dict, index=input_columns)
p_values_df = pd.DataFrame(p_values_dict, index=input_columns)
print("F-scores:\n", f_scores_df)
print("p-values:\n", p_values_df)

# Plot F-scores heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(f_scores_df, annot=True, cmap='coolwarm', cbar=True)
plt.title('F-scores for Input Features')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
#plt.savefig('f_scores_heatmap.png')
plt.show()

# Plot p-values heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(p_values_df, annot=True, cmap='coolwarm', cbar=True)
plt.title('p-values for Input Features')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
#plt.savefig('p_values_heatmap.png')
plt.show()