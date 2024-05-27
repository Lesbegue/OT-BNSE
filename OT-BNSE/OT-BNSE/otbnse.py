import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import find_peaks, peak_widths



plot_params = {'legend.fontsize': 26,
               'figure.figsize': (16, 9),
               'xtick.labelsize': '18',
               'ytick.labelsize': '18',
               'axes.titlesize': '24',
               'axes.labelsize': '22'}
plt.rcParams.update(plot_params)


class bse:
    # Class Attribute none yet

    # Initializer / Instance Attributes
    def __init__(self, space_input=None, space_output=None, tao=0, b=100, aim=None):

        if aim is None:
            self.tao = tao
            self.b = b / np.sqrt(2)
            self.offset = np.median(space_input)
            self.x = space_input - self.offset
            self.y = space_output
            self.wq = 1 / 2
            self.post_mean = None
            self.post_cov = None
            self.post_mean_r = None
            self.post_cov_r = None
            self.post_mean_i = None
            self.post_cov_i = None
            self.time_label = None
            self.signal_label = None
            self.initialise_params()

        elif aim == 'sampling':
            self.wq = 1
            self.vq = 1 / 2
            self.theta = 0
            self.sigma_n = 0
            
        elif aim == 'regression':
            self.x = space_input
            self.y = space_output
            self.Nx = len(self.x)

    def initialise_params(self):
        self.Nx = len(self.x)
        self.w = np.std(self.y)
        self.vq = 1 / 2 / ((np.max(self.x) - np.min(self.x)) / self.Nx) ** 2 / (2 * np.pi)
        self.theta = 0.01
        self.sigma_n = np.std(self.y) / 10
        self.time = np.linspace(np.min(self.x), np.max(self.x), 500)
        self.w = np.linspace(0, self.Nx / (np.max(self.x) - np.min(self.x)) / 16, 500)

    def initialise_params_prime(self):
        self.Nx = len(self.x)
        self.w = np.std(self.y)
        self.vq = 1 / 2 / ((np.max(self.x) - np.min(self.x)) / self.Nx) ** 2 / (2 * np.pi)
        self.theta = 0.01
        self.sigma_n = np.std(self.y) / 10
        # self.time = np.linspace(np.min(self.x),  np.max(self.x), 500)
        self.time = np.linspace(0,  1800, 500)
        self.w = np.linspace(0, self.Nx / (np.max(self.x) - np.min(self.x)) / 16, 500)

    def neg_log_likelihood(self):
        Y = self.y
        Gram = Spec_Mix(self.x, self.x, self.vq, self.theta, self.wq) + 1e-8 * np.eye(self.Nx)
        K = Gram + self.sigma_n ** 2 * np.eye(self.Nx)
        (sign, logdet) = np.linalg.slogdet(K)
        return 0.5 * (Y.T @ np.linalg.solve(K, Y) + logdet + self.Nx * np.log(2 * np.pi))

    def nlogp(self, hypers):
        wq = np.exp(hypers[0])
        vq = np.exp(hypers[1])
        theta = np.exp(hypers[2])
        sigma_n = np.exp(hypers[3])

        Y = self.y
        Gram = Spec_Mix(self.x, self.x, vq, theta, wq)
        K = Gram + sigma_n ** 2 * np.eye(self.Nx) + 1e-5 * np.eye(self.Nx)
        (sign, logdet) = np.linalg.slogdet(K)
        return 0.5 * (Y.T @ np.linalg.solve(K, Y) + logdet + self.Nx * np.log(2 * np.pi))

    def dnlogp(self, hypers):
        wq = np.exp(hypers[0])
        vq = np.exp(hypers[1])
        theta = np.exp(hypers[2])
        sigma_n = np.exp(hypers[3])

        Y = self.y
        Gram = Spec_Mix(self.x, self.x, vq, theta, wq)
        K = Gram + sigma_n ** 2 * np.eye(self.Nx) + 1e-5 * np.eye(self.Nx)
        h = np.linalg.solve(K, Y).T

        dKdsigma = Gram / wq
        dKdgamma = -2 * np.pi ** 2 * outersum(self.x, -self.x) ** 2 * Gram
        dKdtheta = -2 * np.pi * outersum(self.x, -self.x) * wq * np.exp(
            -2 * np.pi ** 2 * outersum(self.x, -self.x) ** 2 * vq) * np.sin(
            2 * np.pi * outersum(self.x, -self.x) * theta)
        dKdsigma_n = 2 * sigma_n * np.eye(self.Nx)

        H = (np.outer(h, h) - np.linalg.inv(K))
        dlogp_dsigma = wq * 0.5 * np.trace(H @ dKdsigma)
        dlogp_dgamma = vq * 0.5 * np.trace(H @ dKdgamma)
        dlogp_dtheta = theta * 0.5 * np.trace(H @ dKdtheta)
        dlogp_dsigma_n = sigma_n * 0.5 * np.trace(H @ dKdsigma_n)
        return np.array([-dlogp_dsigma, -dlogp_dgamma, -dlogp_dtheta, -dlogp_dsigma_n])

    def train(self):
        hypers0 = np.array([np.log(self.wq), np.log(self.vq), np.log(self.theta), np.log(self.sigma_n)])
        res = minimize(self.nlogp, hypers0, args=(), method='L-BFGS-B', jac=self.dnlogp,
                       options={'maxiter': 500, 'disp': True})
        self.wq = np.exp(res.x[0])
        self.vq = np.exp(res.x[1])
        self.theta = np.exp(res.x[2])
        self.sigma_n = np.exp(res.x[3])
        print('Hyperparameters are:')
        print(f'wq ={self.wq}')
        print(f'vq ={self.vq}')
        print(f'theta ={self.theta}')
        print(f'sigma_n ={self.sigma_n}')
        wq = self.wq
        vq = self.vq
        theta = self.theta
        sigma_n = self.sigma_n
        return [wq, vq, theta, sigma_n]

    def Assign(self, wq, vq, theta, sigma_n,tao,b):
        self.wq = wq
        self.vq = vq
        self.theta = theta
        self.sigma_n = sigma_n
        self.tao = tao
        self.b = b

    def sample(self, space_input=None):

        if space_input is None:
            self.Nx = 100
            self.x = np.random.random(self.Nx)
        elif np.size(space_input) == 1:
            self.Nx = space_input
            self.x = np.random.random(self.Nx)
        else:
            self.x = space_input
            self.Nx = len(space_input)
        self.x = np.sort(self.x)
        cov_space = Spec_Mix(self.x, self.x, self.vq, self.theta, self.wq) + self.sigma_n ** 2 * np.eye(
            self.Nx)
        self.y = np.random.multivariate_normal(np.zeros_like(self.x), cov_space)

        return self.y

    def acf(self, instruction):
        times = outersum(self.x, -self.x)
        corrs = np.outer(self.y, self.y)
        times = np.reshape(times, self.Nx ** 2)
        corrs = np.reshape(corrs, self.Nx ** 2)

        # aggregate for common lags
        t_unique = np.unique(times)
        # common_times = t_unique[:, np.newaxis] == times[:, np.newaxis].T
        common_times = np.isclose(t_unique[:, np.newaxis], times[:, np.newaxis].T)
        corrs_unique = np.dot(common_times, corrs)

        if instruction == 'plot.':
            plt.plot(t_unique, corrs_unique, '.')
        if instruction == 'plot-':
            plt.plot(t_unique, corrs_unique)

        return t_unique, corrs_unique

    def compute_moments_time(self):
        # posterior moments for time
        cov_space = Spec_Mix(self.x, self.x, self.vq, self.theta, self.wq) + 1e-5 * np.eye(
            self.Nx) + self.sigma_n ** 2 * np.eye(self.Nx)
        cov_time = Spec_Mix(self.time, self.time, self.vq, self.theta, self.wq)
        cov_star = Spec_Mix(self.time, self.x, self.vq, self.theta, self.wq)
        self.post_mean = np.squeeze(cov_star @ np.linalg.solve(cov_space, self.y))
        self.post_cov = cov_time - (cov_star @ np.linalg.solve(cov_space, cov_star.T))

    def compute_moments(self):
        # posterior moments for time
        cov_space = Spec_Mix(self.x, self.x, self.vq, self.theta, self.wq) + 1e-5 * np.eye(
            self.Nx) + self.sigma_n ** 2 * np.eye(self.Nx)
        cov_time = Spec_Mix(self.time, self.time, self.vq, self.theta, self.wq)
        cov_star = Spec_Mix(self.time, self.x, self.vq, self.theta, self.wq)
        self.post_mean = np.squeeze(cov_star @ np.linalg.solve(cov_space, self.y))
        self.post_cov = cov_time - (cov_star @ np.linalg.solve(cov_space, cov_star.T))

        # posterior moment for frequency
        cov_real, cov_imag = freq_covariances(self.w, self.w, self.b, self.vq, self.theta, self.tao, self.wq,
                                              kernel='sm')
        xcov_real, xcov_imag = time_freq_covariances(self.w, self.x, self.b, self.vq, self.theta, self.tao, self.wq,
                                                     kernel='sm')
        self.post_mean_r = np.squeeze(xcov_real @ np.linalg.solve(cov_space, self.y))
        self.post_cov_r = cov_real - (xcov_real @ np.linalg.solve(cov_space, xcov_real.T))
        self.post_mean_i = np.squeeze(xcov_imag @ np.linalg.solve(cov_space, self.y))
        self.post_cov_i = cov_imag - (xcov_imag @ np.linalg.solve(cov_space, xcov_imag.T))
        self.post_cov_ri = - ((xcov_real @ np.linalg.solve(cov_space, xcov_imag.T)))

        self.post_mean_F = np.concatenate((self.post_mean_r, self.post_mean_i))
        self.post_cov_F = np.vstack(
            (np.hstack((self.post_cov_r, self.post_cov_ri)), np.hstack((self.post_cov_ri.T, self.post_cov_i))))
        post_mean = self.post_mean
        post_mean_r = self.post_mean_r
        post_cov_r = self.post_cov_r
        post_cov_i = self.post_cov_i
        post_mean_i = self.post_mean_i
        post_cov = self.post_cov
        w = self.w
        return w, post_mean, post_cov, post_mean_r, post_cov_r, post_mean_i, post_cov_i

    def plot_time_posterior(self, flag=None):
        # posterior moments for time
        plt.figure(figsize=(18, 6))
        plt.plot(self.x, self.y, '.r', markersize=10, label='observations')
        plt.plot(self.time, self.post_mean, color='blue', label='posterior mean')
        error_bars = 2 * np.sqrt(np.diag(self.post_cov))
        plt.fill_between(self.time, self.post_mean - error_bars, self.post_mean + error_bars, color='blue',
                         alpha=0.1, label='95% error bars')
        if flag == 'with_window':
            plt.plot(self.time, 2 * np.sqrt(self.wq) * np.exp(-(((self.time - self.tao) / self.b) ** 2)))
        plt.title('Observations and posterior interpolation')
        plt.xlabel(self.time_label)
        plt.legend()
        plt.xlim([min(self.x), max(self.x)])
        plt.tight_layout()

    def plot_freq_posterior_real(self):
        plt.figure(figsize=(18, 6))
        plt.plot(self.w, self.post_mean_r, color='blue', label='posterior mean')
        error_bars = 2 * np.sqrt((np.diag(self.post_cov_r)))
        plt.fill_between(self.w, self.post_mean_r - error_bars, self.post_mean_r + error_bars, color='blue',
                         alpha=0.1, label='95% error bars')
        plt.title('Posterior spectrum (real part)')
        plt.xlabel('frequency')
        plt.legend()
        plt.xlim([min(self.w), max(self.w)])
        plt.tight_layout()

    def plot_freq_posterior_imag(self):
        plt.figure(figsize=(18, 6))
        plt.plot(self.w, self.post_mean_i, color='blue', label='posterior mean')
        error_bars = 2 * np.sqrt((np.diag(self.post_cov_i)))
        plt.fill_between(self.w, self.post_mean_i - error_bars, self.post_mean_i + error_bars, color='blue',
                         alpha=0.1, label='95% error bars')
        plt.title('Posterior spectrum (imaginary part)')
        plt.xlabel('frequency')
        plt.legend()
        plt.xlim([min(self.w), max(self.w)])
        plt.tight_layout()

    def plot_freq_posterior(self):
        self.plot_freq_posterior_real()
        self.plot_freq_posterior_imag()

    def plot_power_spectral_density_old(self, how_many, flag=None):
        # posterior moments for frequency
        plt.figure(figsize=(18, 6))
        freqs = len(self.w)
        samples = np.zeros((freqs, how_many))
        for i in range(how_many):
            sample_r = np.random.multivariate_normal(self.post_mean_r,
                                                     (self.post_cov_r + self.post_cov_r.T) / 2 + 1e-5 * np.eye(
                                                         freqs))
            sample_i = np.random.multivariate_normal(self.post_mean_i,
                                                     (self.post_cov_i + self.post_cov_i.T) / 2 + 1e-5 * np.eye(
                                                         freqs))
            samples[:, i] = sample_r ** 2 + sample_i ** 2
        plt.plot(self.w, samples, color='red', alpha=0.35)
        plt.plot(self.w, samples[:, 0], color='red', alpha=0.35, label='posterior samples')
        posterior_mean_psd = self.post_mean_r ** 2 + self.post_mean_i ** 2 + np.diag(
            self.post_cov_r + self.post_cov_i)
        plt.plot(self.w, posterior_mean_psd, color='black', label='(analytical) posterior mean')
        if flag == 'show peaks':
            peaks, _ = find_peaks(posterior_mean_psd, prominence=500000)
            widths = peak_widths(posterior_mean_psd, peaks, rel_height=0.5)
            plt.stem(self.w[peaks], posterior_mean_psd[peaks], markerfmt='ko', label='peaks')
        plt.title('Sample posterior power spectral density')
        plt.xlabel('frequency')
        plt.legend()
        plt.xlim([min(self.w), max(self.w)])
        plt.tight_layout()
        if flag == 'show peaks':
            return peaks, widths

    def plot_3_plots(self, flag = None,loc = None):
        # posterior moments for time
        plt.plot(self.x, self.y, '.r', markersize=10, label='observations')
        error_bars = 2 * np.sqrt(np.diag(self.post_cov))
        time_window = np.linspace(self.tao-self.b/2,self.tao+self.b/2,1000)
        plt.fill_between(self.time, self.post_mean - error_bars, self.post_mean + error_bars, color= 'blue',
                          alpha=0.1, label='95% error bars')
        plt.fill_between(time_window, np.min(self.y)*np.ones(len(time_window)), 2 *np.sqrt(self.wq)*np.exp(-(((time_window - self.tao) /self.b) ** 2)), color='grey',alpha=0.3, label='window region')
        if flag == 'with_window':
           plt.plot(self.time, 2 * np.sqrt(self.wq) * np.exp(-(((self.time - self.tao) / self.b) ** 2)),linewidth=2,linestyle=':',color = "slateblue")
        plt.title('Observations and posterior interpolation')
        plt.xlabel(self.time_label)
        plt.legend()
        plt.xlim([min(self.x), max(self.x)])
        plt.tight_layout()
        plt.savefig(loc + r"\Observations and posterior interpolation.pdf", pad_inches=0)
        plt.close()
        plt.plot(self.w, self.post_mean_r, color='blue', label='posterior mean')
        error_bars = 2 * np.sqrt((np.diag(self.post_cov_r)))
        plt.fill_between(self.w, self.post_mean_r - error_bars, self.post_mean_r + error_bars, color='blue',
                          alpha=0.1, label='95% error bars')
        plt.title('Posterior spectrum (real part)')
        plt.legend()
        plt.xlabel(r'frequency $ \xi $')
        plt.xlim([min(self.w), max(self.w)])
        plt.tight_layout()
        plt.savefig(loc + r"\Posterior spectrum real part.pdf", pad_inches=0)
        plt.close()
        plt.plot(self.w, self.post_mean_i, color='blue', label='posterior mean')
        error_bars = 2 * np.sqrt((np.diag(self.post_cov_i)))
        plt.fill_between(self.w, self.post_mean_i - error_bars, self.post_mean_i+ error_bars, color='blue',
                          alpha=0.1, label='95% error bars')
        plt.title('Posterior spectrum (imaginary part)')
        plt.xlabel(r'frequency $ \xi $')
        plt.legend()
        plt.xlim([min(self.w), max(self.w)])
        plt.tight_layout()
        plt.savefig(loc + r"\Posterior spectrum imaginary part.pdf", pad_inches=0)
        plt.close()
        
    def plot_power_spectral_density(self, how_many, flag=None):
        #posterior moments for frequency
        plt.figure(figsize=(16,9))
        freqs = len(self.w)
        samples = np.zeros((freqs,how_many))
        for i in range(how_many):
            sample = np.random.multivariate_normal(self.post_mean_F,(self.post_cov_F+self.post_cov_F.T)/2 + 1e-5*np.eye(2*freqs))
            samples[:,i] = sample[0:freqs]**2 + sample[freqs:]**2
        plt.plot(self.w,samples, color='red', alpha=0.35)
        plt.plot(self.w,samples[:,0], color='red', alpha=0.35, label='posterior samples')
        posterior_mean_psd = self.post_mean_r**2 + self.post_mean_i**2 + np.diag(self.post_cov_r + self.post_cov_i)
        plt.plot(self.w,posterior_mean_psd, color='black', label = '(analytical) posterior mean')
        if flag == 'show peaks':
            peaks, _  = find_peaks(posterior_mean_psd, prominence=500000)
            widths = peak_widths(posterior_mean_psd, peaks, rel_height=0.5)
            plt.stem(self.w[peaks],posterior_mean_psd[peaks], markerfmt='ko', label='peaks')
        plt.title('Sample posterior power spectral density')
        plt.xlabel('frequency')
        plt.legend()
        plt.xlim([min(self.w),max(self.w)])
        plt.tight_layout()
        if flag == 'show peaks':
            return peaks, widths
        w = self.w
        posterior_mean_psd =posterior_mean_psd
        return w, posterior_mean_psd

    def set_labels(self, time_label, signal_label):
        self.time_label = time_label
        self.signal_label = signal_label

    def set_freqspace(self, max_freq, dimension=500):
        self.w = np.linspace(0, max_freq, dimension)


def outersum(a, b):
    return np.outer(a, np.ones_like(b)) + np.outer(np.ones_like(a), b)


def Spec_Mix(x, y, vq, theta, wq=1):
    return wq * np.exp(-2 * np.pi ** 2 * outersum(x, -y) ** 2 * vq) * np.cos(2 * np.pi * outersum(x, -y) * theta)


def Spec_Mix_sine(x, y, vq, theta, wq=1):
    return wq * np.exp(-2 * np.pi ** 2 * outersum(x, -y) ** 2 * vq) * np.sin(2 * np.pi * outersum(x, -y) * theta)


def Spec_Mix_spectral_real(x, y, b, vq, theta, tao, wq=1):
    magnitude = (np.pi * b ** 2 * wq) * np.sqrt(1 / (1 + 4 * np.pi ** 2 * b ** 2 * vq)) / 2
    return magnitude * np.exp(
        -np.pi ** 2 * b ** 2 * outersum(x, -y) ** 2 / 2 - 2 * np.pi ** 2 * b ** 2 * (
                outersum(x, y) / 2 - theta) ** 2 / (
                1 + 4 * np.pi ** 2 * b ** 2 * vq)) * np.cos(2 * np.pi * outersum(x, -y) * tao)


def Spec_Mix_spectral_imag(x, y, b, vq, theta, tao, wq=1):
    magnitude = (np.pi * b ** 2 * wq) * np.sqrt(1 / (1 + 4 * np.pi ** 2 * b ** 2 * vq)) / 2
    return magnitude * np.exp(
        -np.pi ** 2 * b ** 2 * outersum(x, -y) ** 2 / 2 - 2 * np.pi ** 2 * b ** 2 * (
                outersum(x, y) / 2 - theta) ** 2 / (
                1 + 4 * np.pi ** 2 * b ** 2 * vq)) * np.sin(2 * np.pi * outersum(x, -y) * tao)


def freq_covariances(x, y, b, vq, theta, tao, wq=1, kernel='sm'):
    if kernel == 'sm':
        N = len(x)
        # compute kernels
        K = Spec_Mix_spectral_real(x, y, b, vq, theta, tao, wq) + Spec_Mix_spectral_real(x, y, b, vq,
                                                                                         -theta, tao, wq)
        P = Spec_Mix_spectral_real(x, -y, b, vq, theta, tao, wq) + Spec_Mix_spectral_real(x, -y, b,
                                                                                          vq, -theta,
                                                                                          tao, wq)
        real_cov = 1 / 2 * (K + P) + 1e-8 * np.eye(N)
        imag_cov = 1 / 2 * (K - P) + 1e-8 * np.eye(N)

    return real_cov, imag_cov


def time_freq_SM_re(x, y, b, vq, theta, tao, wq=1):
    Lq = 1 + 2*np.pi**2*b**2*vq
    magnitude = b * wq * np.sqrt(np.pi / Lq) / 2
    return magnitude * np.exp(outersum(-np.pi**2*b**2*(x-theta)**2/Lq,-2*np.pi**2*vq*(y-tao)**2/Lq))  * np.cos(-2*np.pi*np.outer(x,y)+2*np.pi*np.outer(x-theta, y-tao)/Lq)


def time_freq_SM_im(x, y, b, vq, theta, tao, wq=1):
    Lq = 1 + 2*np.pi**2*b**2*vq
    magnitude = b * wq * np.sqrt(np.pi / Lq) / 2
    return magnitude * np.exp(outersum(-np.pi**2*b**2*(x-theta)**2/Lq,-2*np.pi**2*vq*(y-tao)**2/Lq))  * np.sin(-2*np.pi*np.outer(x,y)+2*np.pi*np.outer(x-theta, y-tao)/Lq)


def time_freq_covariances(x, t, b, vq, theta, tao, wq, kernel='sm'):
    if kernel == 'sm':
        tf_real_cov = time_freq_SM_re(x, t, b, vq, theta, tao, wq) + time_freq_SM_re(x, t, b, vq,
                                                                                     -theta, tao, wq)
        tf_imag_cov = time_freq_SM_im(x, t, b, vq, theta, tao, wq) + time_freq_SM_im(x, t, b, vq,
                                                                                     -theta, tao, wq)
    return tf_real_cov, tf_imag_cov
