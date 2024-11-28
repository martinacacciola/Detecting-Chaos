import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpmath import mp
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


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
plt.plot(T_norm_midpoints, slopes, color='b', alpha=0.7)
plt.xlabel(r'$T/T_c$')
plt.ylabel(r'$\log10(slope))$')
plt.title('Distribution of slopes over time')
plt.grid(True)
#plt.legend()
plt.savefig('./figures/slopes_over_time.png') 
#plt.show()

# Plot histogram of slopes
plt.figure(figsize=(8, 6))
plt.hist(slopes, bins=100, alpha=0.7)
plt.xlabel(r'Instantaneous Slope of log($\delta$)')
plt.ylabel('Frequency')
plt.title('Distribution of Instantaneous Slope of Phase-Space Distance')
plt.grid(True)
plt.savefig('./figures/slope_manually.png')
plt.show()


##
# Window slope calculation
# Define window size in terms of timesteps (number of points to include in each fit)
window_size = 40

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
""" # 1) fitting a Gaussian and residuals to the histogram of window slopes
window_midpoints = np.array(window_midpoints, dtype=float)
mu, std = norm.fit(window_slopes)
#print('mu:', mu)
#print('std:', std)
hist_counts, bin_edges = np.histogram(window_slopes, bins=50, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
gaussian = norm.pdf(bin_centers, loc=mu, scale=std)
# compute residuals
residuals = hist_counts - gaussian
mu_res, std_res = norm.fit(residuals)
#print('mu_res:', mu_res)
#print('std_res:', std_res)
gaussian_res = norm.pdf(bin_centers, loc=mu_res, scale=std_res)

# Plot histogram of slopes with Gaussian fit
plt.figure(figsize=(8, 6))
plt.hist(window_slopes, bins=50, alpha=0.7, density=True, label='Histogram')
plt.plot(bin_centers, gaussian, color='r', alpha=0.7, label='Gaussian fit')
# add residuals (if i use gaussian_res flat line)
plt.plot(bin_centers, residuals, color='b', alpha=0.7, label='Residuals')
plt.xlabel(r'Slope of log($\delta$)')
plt.ylabel('Density')
plt.title('Distribution of Slope of Phase-Space Distance')
plt.legend()
plt.grid(True)
plt.savefig('./figures/slope_window_hist.png')
plt.show() """


##
# 2) fitting a Gaussian Mixture Model to the window slopes
window_slopes = np.array(window_slopes)
gmm = GaussianMixture(n_components=2)
gmm.fit(window_slopes.reshape(-1, 1))
# reshaped into a 2d array as the input required by gmm (even if we have only one feature)

# this generates a smooth pdf from the gmm 
x = np.linspace(min(window_slopes), max(window_slopes), 1000)
# compute the log likelihood of each sampe
logprob = gmm.score_samples(x.reshape(-1, 1))
pdf = np.exp(logprob) # convert log probabilities to probabilities

# extract the means and covariances of each component
means = gmm.means_.flatten()
covariances = gmm.covariances_.flatten()
weights = gmm.weights_.flatten()

# pdf computed for each component separately
pdf_individual = [weights[i] * norm.pdf(x, means[i], np.sqrt(covariances[i])) for i in range(gmm.n_components)]

plt.figure(figsize=(8, 6))
plt.hist(window_slopes, bins=50, alpha=0.5, density=True, label='Histogram')
for i, pdf_i in enumerate(pdf_individual):
    plt.plot(x, pdf_i, alpha=0.7, label=f'Gaussian {i+1}: mean={means[i]:.3f}, var={covariances[i]:.3f}')
    # add dotted vertical line at the mean of each component
    plt.axvline(means[i], color='k', linestyle='dotted')
    # add label with mean value
    #plt.text(means[i], 0.1, f'mean={means[i]:.2f}')
plt.xlabel(r'Slope of log($\delta$)')
plt.ylabel('Density')
plt.title('Distribution of Slope of Phase-Space Distance')
plt.legend()
plt.grid(True)
plt.savefig('./figures/slope_window_gmm.png')
plt.show()

## 
# 3) evaluate the quality of the fit using different window sizes
""" def evaluate_gmm_fit(window_slopes, window_sizes):
    bic_scores = {1: [], 2: [], 3: []}
    aic_scores = {1: [], 2: [], 3: []}
    
    for window_size in window_sizes:
        
        num_windows = len(window_slopes) - int(window_size) + 1

        if num_windows <= 0:  # Skip window sizes larger than data length
            break
        
        window_slopes_array = np.array(window_slopes).reshape(-1, 1)
        
        if len(window_slopes_array) < 2:  # Ensure there are at least 2 samples
            continue
        
        # Fit GMMs with 1, 2, and 3 components on windowed data
        for n_components in [1, 2, 3]:
            gmm = GaussianMixture(n_components=n_components)
            gmm.fit(window_slopes_array)
            bic_scores[n_components].append(gmm.bic(window_slopes_array))
            aic_scores[n_components].append(gmm.aic(window_slopes_array))
        #print(f'Window size: {window_size:.2f} | BIC: {bic_scores} | AIC: {aic_scores}')

    # Plot BIC and AIC scores
    plt.figure(figsize=(12, 6))
    for n_components in [1, 2, 3]:
        plt.plot(window_sizes[:len(bic_scores[n_components])], bic_scores[n_components], label=f'BIC: {n_components} components')
        plt.plot(window_sizes[:len(aic_scores[n_components])], aic_scores[n_components], label=f'AIC: {n_components} components', linestyle='dashed')
    plt.xlabel('Window Size')
    plt.ylabel('BIC / AIC')
    plt.title('BIC and AIC as a Function of Window Size')
    plt.legend()
    plt.grid(True)
    plt.show()

window_sizes = np.arange(0.5, 100, 0.5)
evaluate_gmm_fit(window_slopes, window_sizes) """

## 3) evaluate quality of the fit using the mean of the window slopes
def evaluate_gmm_fit_mean(window_slopes, window_sizes):
    bic_scores = {1: [], 2: [], 3: []}
    aic_scores = {1: [], 2: [], 3: []}
    residuals_scores = {1: [], 2: [], 3: []}
    rmse_scores = {1: [], 2: [], 3: []}
    
    for window_size in window_sizes:
        # Apply rolling window: compute mean of each window
        num_windows = len(window_slopes) - int(window_size) + 1
        if num_windows <= 0:  # Skip window sizes larger than data length
            break
        window_means = []
        for i in range(num_windows):
            window_slice = window_slopes[i:i + int(window_size)]
            if len(window_slice) > 0:  # Ensure the slice is not empty
                window_means.append(window_slice.mean())
        window_means = np.array(window_means)
        
        if len(window_means) < 2:  # Ensure there are at least 2 samples
            continue
        
        # Fit GMMs with 1, 2, and 3 components on windowed data
        for n_components in [1, 2, 3]:
            gmm = GaussianMixture(n_components=n_components)
            gmm.fit(window_means.reshape(-1, 1))
            
            # Compute BIC and AIC
            bic_scores[n_components].append(gmm.bic(window_means.reshape(-1, 1)))
            aic_scores[n_components].append(gmm.aic(window_means.reshape(-1, 1)))

            # Calculate residuals
            x = np.linspace(min(window_slopes), max(window_slopes), 1000)
            logprob = gmm.score_samples(x.reshape(-1, 1))
            pdf = np.exp(logprob)
            hist_counts, bin_edges = np.histogram(window_slopes, bins=50, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            #gaussian = np.exp(gmm.score_samples(bin_centers.reshape(-1, 1))) 
            residuals = hist_counts - pdf[:len(hist_counts)] # values for the overall fit, not for each component, check
            residuals_scores[n_components].append(np.sum(residuals**2))
            rmse_scores[n_components].append(np.sqrt(np.mean(residuals**2)))

        # Evaluate optimal window size based on BIC for n_components=2
        # TODO: sistemare questa parte
        bic_2 = bic_scores[2]
        optimal_window_size = {}

        threshold = 10
        consecutive_count = 5  # Number of consecutive values under the threshold, con 10 e 5 trova 32
        
        bic_diffs = np.diff(bic_2)
        #print('BIC differences:', bic_diffs)
        
        stable_index = None
        for i in range(len(bic_diffs) - consecutive_count + 1):
            if np.all(np.abs(bic_diffs[i:i + consecutive_count]) < threshold):
                stable_index = i + consecutive_count - 1  # Last index in the stable sequence
                break
        
        if stable_index is not None:
            optimal_window_size = window_sizes[stable_index + 1]  # +1 to account for the diff shift
            #print('the chosen window size is:', optimal_window_size)
        else:
            optimal_window_size = window_sizes[np.argmin(bic_2)]
            #print('the chosen window size is:', optimal_window_size)



    # Plot BIC and AIC scores
    plt.figure(figsize=(12, 6))
    for n_components in [1, 2, 3]:
        plt.plot(window_sizes[:len(bic_scores[n_components])], bic_scores[n_components], label=f'BIC: {n_components} components', alpha=0.7)
    plt.xlabel('Window Size')
    plt.ylabel('BIC')
    plt.title('BIC as Function of Window Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('./figures/bic_window.png')
    plt.show()

    """ plt.figure(figsize=(12, 6))
    for n_components in [1, 2, 3]:
        plt.plot(window_sizes[:len(aic_scores[n_components])], aic_scores[n_components], label=f'AIC: {n_components} components', alpha=0.7)
    plt.xlabel('Window Size')
    plt.ylabel('AIC')
    plt.title('AIC as a Function of Window Size')
    plt.legend()
    plt.grid(True)
    plt.show() """

    # Plot BIC scores for n_components=2 and mark the optimal window size
    plt.figure(figsize=(12, 6))
    plt.plot(window_sizes[:len(bic_scores[2])], bic_scores[2], label='BIC: 2 components', alpha=0.7)
    if optimal_window_size:
        plt.axvline(optimal_window_size, color='r', linestyle='--', label=f'Optimal Window Size: {optimal_window_size}')
    plt.xlabel('Window Size')
    plt.ylabel('BIC')
    plt.title('BIC as Function of Window Size (n_components=2)')
    plt.legend()
    plt.grid(True)
    plt.savefig('./figures/bic_window_optimal.png')
    plt.show()

    # residuals scores
    # quantify the overall discrepancy between the observed data and the model's predictions
    plt.figure(figsize=(12, 6))
    for n_components in [1, 2, 3]:
        plt.plot(window_sizes[:len(residuals_scores[n_components])], residuals_scores[n_components], label=f'Residuals: {n_components} components', alpha=0.7)
    plt.xlabel('Window Size')
    plt.ylabel('Residuals')
    plt.title('Residuals as a Function of Window Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('./figures/residuals_window.png')
    plt.show()

    # RMSE (root mean square error) scores
    plt.figure(figsize=(12, 6))
    for n_components in [1, 2, 3]:
        plt.plot(window_sizes[:len(rmse_scores[n_components])], rmse_scores[n_components], label=f'RMSE: {n_components} components', alpha=0.7)
    plt.xlabel('Window Size')
    plt.ylabel('RMSE')
    plt.title('RMSE as a Function of Window Size')
    plt.legend()
    plt.grid(True)
    plt.show()

    # residuals vs. fitted values (remove)
    """ plt.figure(figsize=(12, 6))
    for n_components in [1, 2, 3]:
        plt.scatter(bin_centers, residuals, label=f'Residuals: {n_components} components', alpha=0.5)
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Fitted Values')
    plt.legend()
    plt.grid(True)
    plt.show() """

window_sizes = np.arange(0.5, 100, 0.5)
evaluate_gmm_fit_mean(window_slopes, window_sizes)
    
##
# 4) compute mean and std of the two components for different window sizes
# the windows goes from 0.5 to 100 with a step of 0.5

# initialize lists to store the means and stds of the two components
means1, means2, stds1, stds2, window_sizes, peak1_heights, peak2_heights = [], [], [], [], [], [], []

# Iterate over different window sizes, starting from 0.5 and incrementing by 0.5
for window_size in np.arange(0.5, 100, 0.5):
    window_slopes = []
    window_midpoints = []

    # Skip if the window size exceeds the data length
    if window_size > len(T_norm):
        continue

    for start_idx in range(0, len(T_norm) - int(window_size) + 1):
        end_idx = start_idx + int(window_size)
        delta_flip = np.flip(delta_per_step)
        delta_window = delta_flip[start_idx:end_idx]
        T_norm_window = T_norm[start_idx:end_idx]

        delta_log_window = np.log10(np.array(delta_window, dtype=float))

        # ensure there are at least two points to compute slope
        if len(delta_log_window) < 2:
            continue

        slope = (delta_log_window[-1] - delta_log_window[0]) / (T_norm_window[-1] - T_norm_window[0])
        window_slopes.append(float(slope))
        window_midpoints.append((T_norm_window[0] + T_norm_window[-1]) / 2)

    # fit gaussian mixture model to the window slopes
    if window_slopes:
        window_slopes = np.array(window_slopes)
        gmm = GaussianMixture(n_components=2)
        gmm.fit(window_slopes.reshape(-1, 1))
        means = gmm.means_.flatten()
        covariances = gmm.covariances_.flatten()

        weights = gmm.weights_.flatten()
        peak1_height = weights[0] / np.sqrt(2 * np.pi * covariances[0])
        peak2_height = weights[1] / np.sqrt(2 * np.pi * covariances[1])

        means1.append(means[0])
        means2.append(means[1])
        stds1.append(np.sqrt(covariances[0]))
        stds2.append(np.sqrt(covariances[1]))
        peak1_heights.append(peak1_height)
        peak2_heights.append(peak2_height)
        window_sizes.append(window_size)


# Plot the evolution of the means of the 2nd component wrt window size
# not smoothed version
""" plt.figure(figsize=(8, 6))
plt.plot(window_sizes, means2, color='b', alpha=0.7)
plt.xlabel('Window Size')
plt.ylabel('Mean of Component 2')
plt.title('Evolution of the Mean of Component 2 with Window Size')
plt.savefig('./figures/mean_component2.png')
plt.grid(True)
#plt.show() """

# convolution to take the avg of fixed number of consecutive points
# to smooth out short-term fluctuations
def smoothing(data, points):
    return np.convolve(data, np.ones(points) / points, mode='valid')

# choose the smoothing window size
window_size_ma = 10

means1_smoothed = smoothing(means1, window_size_ma)
std1_smoothed = smoothing(stds1, window_size_ma)

means2_smoothed = smoothing(means2, window_size_ma)
std2_smoothed = smoothing(stds2, window_size_ma)
window_sizes_smoothed = smoothing(window_sizes, window_size_ma)

# smoothed evolution of the means wrt window size
plt.figure(figsize=(8, 6))
plt.plot(window_sizes_smoothed, means1_smoothed, color='r', alpha=0.7, label=r'$\mu_1$')
plt.plot(window_sizes_smoothed, means2_smoothed, color='b', alpha=0.3, label=r'$\mu_2$')
plt.xlabel('Window Size')
plt.ylabel('Value')
plt.title(r'Evolution of $\mu$ of the Gaussians with Window Size')
plt.legend()
plt.grid(True)
plt.savefig('./figures/mean_guassian_smoothed.png')
plt.show()

# smoothed evolution of the std wrt window size
plt.figure(figsize=(8, 6))
plt.plot(window_sizes_smoothed, std1_smoothed, color='r', alpha=0.7, label=r'$\sigma_1$')
plt.plot(window_sizes_smoothed, std2_smoothed, color='b', alpha=0.3, label=r'$\sigma_2$')
plt.xlabel('Window Size')
plt.ylabel('Value')
plt.title(r'Evolution of $\sigma$ of the Gaussians with Window Size')
plt.legend()
plt.grid(True)
plt.savefig('./figures/std_guassian_smoothed.png')
plt.show()



""" 
# Generate histogram data
hist_counts, bin_edges = np.histogram(window_slopes, bins=50, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Fit a single Gaussian
mean1, std1 = norm.fit(window_slopes)
gaussian1 = norm.pdf(bin_centers, loc=mean1, scale=std1)

residuals = hist_counts - gaussian1 """




