import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

inputs = np.genfromtxt('./data/all_X_norm.csv', delimiter=',')
outputs = np.genfromtxt('./data/all_y_norm.csv', delimiter=',')

# Combine inputs and outputs
combined_data = np.hstack((inputs, outputs))

# Create column names for the combined data
input_columns = [f'Particle {i//6 + 1} {["X Pos", "Y Pos", "Z Pos", "X Vel", "Y Vel", "Z Vel"][i % 6]}' 
                 for i in range(inputs.shape[1])]
output_columns = [f'Gaussian {i//4 + 1} {["Mean", "Std", "Weight", "Height"][i % 4]}' 
                  for i in range(outputs.shape[1])]
columns = input_columns + output_columns

# Create a DataFrame for better handling
df = pd.DataFrame(combined_data, columns=columns)

# Handle invalid values by removing rows with NaNs or infinities
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Separate inputs and outputs again (clean)
inputs = df[input_columns].values
outputs = df[output_columns].values

# Compute correlations between inputs and outputs
input_output_corr = pd.DataFrame(
    np.corrcoef(inputs, outputs, rowvar=False)[:inputs.shape[1], inputs.shape[1]:],
    index=input_columns,
    columns=output_columns
)

# Plot the reduced correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(input_output_corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, 
            xticklabels=output_columns, yticklabels=input_columns, annot_kws={"size": 8})
plt.title('Correlation Between Inputs and Outputs')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()  
plt.show()

