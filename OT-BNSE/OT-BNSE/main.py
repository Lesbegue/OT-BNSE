import numpy as np
import matplotlib.pyplot as plt
import scipy
from otbnse import *
import pandas as pd

# Set plot parameters for Matplotlib
plot_params = {'legend.fontsize': 26,
               'figure.figsize': (16, 9),
               'xtick.labelsize': '18',
               'ytick.labelsize': '18',
               'axes.titlesize': '24',
               'axes.labelsize': '22'}
plt.rcParams.update(plot_params)

# Define the methods, method names, and data names to be used
method_ = ['OT-BNSE']
method_name = ['SE_method']
Data_name = ['exam_stress']

# Initialize global variables
global wq, vq, theta, sigma_n
u = -1

# Loop through each method
while u < len(method_) - 1:
    u += 1
    v = -1
    # Loop through each data example
    while v < len(Data_name) - 1:
        v += 1
        try:
            example = Data_name[v]

            # Load the appropriate dataset based on the example
            if example == 'exam_stress':
                file_path = r'data\a-wearable-exam\Data\S3\Midterm 2\HR.csv'
                signal_raw = pd.read_csv(file_path, header=0)
                signal_raw = np.array(signal_raw.iloc[3:, 0])
                time = np.linspace(0, len(signal_raw), len(signal_raw))
                time_label = 'time'
                signal_label = 'A Wearable Exam Stress Dataset'
                size = round(len(signal_raw) / 10)

            elif example == 'Body sway':
                file_path = r'data\body-sway\S4\ECL2.csv'
                signal_raw = pd.read_csv(file_path, header=0)
                signal_raw = np.array(signal_raw.iloc[1:10000, 2])
                time = np.linspace(0, len(signal_raw), len(signal_raw))
                time_label = 'time'
                signal_label = 'Body sway when standing and listening to music'
                size = round(len(signal_raw) / 10)

            elif example == 'posture and gait':
                signal_raw = pd.read_csv(
                    r'data\participant01\participant01_left_0.3\participant01_left_0.3_corner1/aligned_skeleton_2d_posture.csv',
                    header=0)
                signal_raw = np.array(signal_raw.iloc[:, 18])
                time = np.linspace(0, len(signal_raw), len(signal_raw))
                time_label = 'time'
                signal_label = 'posture and gait'
                size = round(len(signal_raw) / 3)

            # Preprocess the signal
            signal_raw = signal_raw - np.mean(signal_raw)
            indices = np.random.randint(0, len(signal_raw), size)
            signal_ind = signal_raw[indices]
            time_ind = time[indices]
            semi_window_len = round(len(signal_raw) / 12)
            tao = [(2 * int(i) + 1) * semi_window_len for i in np.linspace(-3, 2, 6)]
            n = len(tao)
            b = 2 * semi_window_len

            # Perform the OT-BNSE method
            if method_[u] == 'OT-BNSE':
                for i in range(n):
                    if i == 0:
                        my_bse = bse(time_ind, signal_ind, tao[i], b)
                        my_bse.set_labels(time_label, signal_label)
                        if example == 'exam_stress':
                            my_bse.set_freqspace(0.003)
                        elif example == 'Body sway':
                            my_bse.set_freqspace(0.0013)
                        elif example == 'posture and gait':
                            my_bse.set_freqspace(0.03)

                        nll = my_bse.neg_log_likelihood()
                        print(f'Negative log likelihood (before training): {nll}')
                        my_bse.compute_moments()
                        print(my_bse.theta)

                        # Train the model and retrieve parameters
                        [wq, vq, theta, sigma_n] = my_bse.train()
                        nll = my_bse.neg_log_likelihood()
                        print(f'Negative log likelihood (after training): {nll}')

                        # Compute moments and plot results
                        w, post_mean, post_cov, post_mean_r, post_cov_r, post_mean_i, post_cov_i = my_bse.compute_moments()
                        w, mean_psd = my_bse.plot_power_spectral_density(15)
                        plt.close()

                        mean_psd_total = (mean_psd - np.mean(mean_psd)) / np.std(mean_psd)
                        my_bse.Assign(wq, vq, theta, sigma_n, tao[i], b)
                        my_bse.plot_3_plots(r'with_window', r"Figure\window_{}".format(i))
                        print("==================", "The", i, "window", "=============================")
                    else:
                        my_bse = bse(time_ind, signal_ind, tao[i], b)
                        my_bse.set_labels(time_label, signal_label)
                        if example == 'exam_stress':
                            my_bse.set_freqspace(0.003)
                        elif example == 'Body sway':
                            my_bse.set_freqspace(0.0013)
                        elif example == 'posture and gait':
                            my_bse.set_freqspace(0.03)

                        my_bse.Assign(wq, vq, theta, sigma_n, tao[i], b)
                        [wq, vq, theta, sigma_n] = my_bse.train()
                        nll = my_bse.neg_log_likelihood()
                        print(f'Negative log likelihood (after training): {nll}')

                        w, post_mean, post_cov, post_mean_r, post_cov_r, post_mean_i, post_cov_i = my_bse.compute_moments()
                        my_bse.Assign(wq, vq, theta, sigma_n, tao[i], b)
                        my_bse.plot_3_plots(r'with_window', r"Figure\window_{}".format(i))
                        print("==================", "The", i, "window", "=============================")
        # Handle exceptions
        except Exception as e:
            print(e)
            if e == "Singular matrix":
                v = v - 1
                continue
            else:
                break
