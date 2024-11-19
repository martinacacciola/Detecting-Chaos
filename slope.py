import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpmath import mp
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


pd.options.display.float_format = "{:,.120f}".format 

df = pd.read_csv('./Brutus data/plummer_triples_L0_00_i1775_e90_Lw392.csv', dtype=str) 

def count_decimals(value):
    if '.' in value:
        return len(value.split('.')[1])
    return 0

# Find the maximum number of decimal places
max_decimal_places = 0
for col in ['X Position', 'Y Position', 'Z Position', 'X Velocity', 'Y Velocity', 'Z Velocity']:
    max_decimal_places = max(max_decimal_places, df[col].apply(count_decimals).max())

# Set the global decimal places
mp.dps = max_decimal_places
mp.prec = 3.33 * max_decimal_places  # Set precision to 3.33 times the decimal places

# takes input value (expected to be a string) and converts it to mpf object
def string_to_mpf(value):
    return mp.mpf(value)

for col in ['X Position', 'Y Position', 'Z Position', 'X Velocity', 'Y Velocity', 'Z Velocity']:
    df[col] = df[col].apply(string_to_mpf)

# Separate forward and backward trajectories
forward_trajectory = df[df['Phase'].astype(int) == 1]
backward_trajectory = df[df['Phase'].astype(int) == -1]

# Get unique timesteps and calculate midpoint
total_timesteps = len(df['Timestep'].unique())
midpoint = total_timesteps // 2  # Midpoint corresponds to t=T=lifetime
timesteps = df['Timestep'].unique()

# take delta per step from txt file
delta_per_step = np.loadtxt('./data/delta_per_step_L0_00_i1775_e90_Lw392.txt')
 
# Crossing time 
T_c = mp.mpf(2) * mp.sqrt(2)
# Normalize lifetime 
T_norm = [mp.mpf(timestep) / T_c for timestep in timesteps[:len(delta_per_step)]]

# Compute instantaneous slope of phase-space distance
delta_flip = np.flip(delta_per_step)
delta_log = np.log10(np.array(delta_flip, dtype=float))
#delta_arr = np.array(delta_per_step, dtype=float)
slopes = np.diff(delta_log) / np.diff(T_norm)  # Slope calculation using consecutive differences
slopes = np.array(slopes, dtype=float)

# midpoints of T_norm intervals for plotting
T_norm_midpoints = [(T_norm[i] + T_norm[i + 1]) / 2 for i in range(len(T_norm) - 1)]

# distribution of slopes over time  
plt.figure(figsize=(10, 6))
plt.plot(T_norm_midpoints, slopes, color='g', alpha=0.7)
plt.xlabel(r'$T/T_c$')
plt.ylabel(r'$\log10(slope))$')
plt.title('Distribution of slopes over time')
plt.grid(True)
plt.legend()
plt.savefig('./figures/slopes_over_time.png') 
#plt.show()

# Plot histogram of slopes
plt.figure(figsize=(8, 6))
plt.hist(slopes, bins=100, alpha=0.7)
plt.xlabel('Instantaneous Slope of log(delta)')
plt.ylabel('Frequency')
plt.title('Distribution of Instantaneous Slope of Phase-Space Distance')
plt.grid(True)
plt.savefig('./figures/slope_manually.png')
plt.show()


##
# Window slope calculation
# Define window size in terms of timesteps (number of points to include in each fit)
window_size = 50

window_slopes = []
window_midpoints = []

# iterate over the indices of T_norm in steps of 1
# in this way we have a comoving window - for each timestep compute slope over the window
for start_idx in range(0, len(T_norm) - window_size + 1):
    end_idx = start_idx + window_size

    delta_flip = np.flip(delta_per_step)
    # select the window over which computing the slope
    delta_window = delta_flip[start_idx:end_idx]
    T_norm_window = T_norm[start_idx:end_idx]

    delta_log_window = np.log10(np.array(delta_window, dtype=float))
    
    # compute slope using initial and final values of each window
    slope = (delta_log_window[-1] - delta_log_window[0]) / (T_norm_window[-1] - T_norm_window[0])
    window_slopes.append(float(slope))  

    # midpoint of the current time window
    window_midpoints.append((T_norm_window[0] + T_norm_window[-1]) / 2)

plt.figure(figsize=(10, 6))
plt.plot(window_midpoints, window_slopes, color='b', alpha=0.7)
plt.xlabel(r'$T/T_c$')
plt.ylabel('Slope')
plt.title(r'Slope of log($\delta$) Over Time')
plt.grid(True)
plt.savefig('./figures/slope_window.png')
plt.show()

##
# 1) fitting a Gaussian and residuals to the histogram of window slopes
window_midpoints = np.array(window_midpoints, dtype=float)
mu, std = norm.fit(window_slopes)
print('mu:', mu)
print('std:', std)
hist_counts, bin_edges = np.histogram(window_slopes, bins=50, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
gaussian = norm.pdf(bin_centers, loc=mu, scale=std)
# compute residuals
residuals = hist_counts - gaussian
mu_res, std_res = norm.fit(residuals)
print('mu_res:', mu_res)
print('std_res:', std_res)
gaussian_res = norm.pdf(bin_centers, loc=mu_res, scale=std_res)

# Plot histogram of slopes with Gaussian fit
plt.figure(figsize=(8, 6))
plt.hist(window_slopes, bins=50, alpha=0.7, density=True, label='Histogram')
plt.plot(bin_centers, gaussian, color='r', alpha=0.7, label='Gaussian fit')
# add residual gaussian (if i use gaussian_res flat line)
plt.plot(bin_centers, residuals, color='b', alpha=0.7, label='Residuals')
plt.xlabel(r'Slope of log($\delta$)')
plt.ylabel('Density')
plt.title('Distribution of Slope of Phase-Space Distance')
plt.legend()
plt.grid(True)
plt.savefig('./figures/slope_window_hist.png')
plt.show()


##
# 2) fitting a Gaussian Mixture Model to the window slopes
window_slopes = np.array(window_slopes)
gmm = GaussianMixture(n_components=2)
gmm.fit(window_slopes.reshape(-1, 1))

# this generates a smooth pdf from the gmm 
# convert log probabilities to probabilities
x = np.linspace(min(window_slopes), max(window_slopes), 1000)
logprob = gmm.score_samples(x.reshape(-1, 1))
pdf = np.exp(logprob)


# Extract the means and covariances of each component
means = gmm.means_.flatten()
covariances = gmm.covariances_.flatten()
weights = gmm.weights_.flatten()
print('mean and variance of first component:', means[0], covariances[0])
print('mean and variance of second component:', means[1], covariances[1])

# Compute the PDF for each component separately
pdf_individual = [weights[i] * norm.pdf(x, means[i], np.sqrt(covariances[i])) for i in range(gmm.n_components)]

plt.figure(figsize=(8, 6))
plt.hist(window_slopes, bins=50, alpha=0.5, density=True, label='Histogram')
#plt.plot(x, pdf, color='r', alpha=0.7, label='Gaussian Mixture fit')
for i, pdf_i in enumerate(pdf_individual):
    plt.plot(x, pdf_i, alpha=0.7, label=f'Gaussian {i+1}: mean={means[i]:.3f}, var={covariances[i]:.2f}')
    # add dotted vertical line at the mean of each component
    plt.axvline(means[i], color='k', linestyle='dotted')
    # add label with mean value
    #plt.text(means[i], 0.1, f'mean={means[i]:.2f}')
plt.xlabel(r'Slope of log($\delta$)')
plt.ylabel('Density')
plt.title('Distribution of Slope of Phase-Space Distance')
plt.legend()
plt.grid(True)
plt.show()


#sns.histplot(window_slopes, bins=40, kde= True, alpha=0.7, edgecolor='black', stat='density')
#sns.kdeplot(window_slopes, color='r', linewidth=2)




""" 
# Generate histogram data
hist_counts, bin_edges = np.histogram(window_slopes, bins=50, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Fit a single Gaussian
mean1, std1 = norm.fit(window_slopes)
gaussian1 = norm.pdf(bin_centers, loc=mean1, scale=std1)

residuals = hist_counts - gaussian1 """




