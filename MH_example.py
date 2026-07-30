import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular

np.random.seed(42)

N = 80
x = np.linspace(0, 1, N)
kernel_size = N
kernel_gamma = 0.03

# noise precision
lam = 5.35

# hyperprior parameters
alpha_del = 1
beta_del = 1e-4
alpha_lam = 1
beta_lam = 1e-4

def true_signal(x): # smooth gaussian bump and two sharp step-function blocks
    return 50*np.exp(-((x-0.75)/0.1)**2) + 37*np.logical_and(x>0.1,x<0.25) + 13*np.logical_and(x>0.3,x<0.32) 

def generate_blurred_data(true_signal, x, A, lam):
    noise_std = 1/np.sqrt(lam) # lam: precision
    blurred_signal = np.dot(A,true_signal(x))
    noise = np.random.normal(0, noise_std, len(x)) # eps ~ N(0,noise_std I)
    return blurred_signal + noise

def gaussian_kernel(x, gamma):
    return np.exp(-x**2 / (2 * gamma**2))/np.sqrt(np.pi*gamma**2)

np.random.seed(42)
N = 80
x = np.linspace(0, 1, N)
kernel_gamma = 0.03
lam = 5.35
alpha_del = 1
beta_del = 1e-4
alpha_lam = 1
beta_lam = 1e-4

A = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        d = min(np.abs(i-j), N - np.abs(i-j))
        A[i, j] = gaussian_kernel(d/N, kernel_gamma) / N
A = A / np.sum(A[0, :]) 

observed_data = generate_blurred_data(true_signal, x, A, lam)

def neglogpi_theta(delta, lam, A, data):
    # compute prior precision L
    L = 2*np.diag(np.ones(N)) - np.diag(np.ones(N-1), k=1) - np.diag(np.ones(N-1), k=-1)
    L = delta*L
    
    # compute posterior precision and mean
    Lpost = lam*np.dot(np.transpose(A),A) + L
    xpost = lam*np.dot(np.transpose(A),data)
    xpost = np.linalg.solve(Lpost, xpost)
    
    # compute negative log of determinant ratio
    (sdetL, logdetL) = np.linalg.slogdet(L)
    (sdetLpost, logdetLpost) = np.linalg.slogdet(Lpost)
    det_ratio = 0.5*logdetLpost - 0.5*logdetL - N/2*np.log(lam)
    
    # compute data, posterior mean, and hyperprior terms
    uQu_term = - 0.5*np.dot(xpost, np.dot(Lpost, xpost))
    yQy_term = 0.5*lam*np.dot(data,data) 
    prior_term = beta_del*delta + beta_lam*lam - (alpha_del-1)*np.log(delta) - (alpha_lam-1)*np.log(lam)
    
    return det_ratio+uQu_term+prior_term+yQy_term


def metropolis_hastings(n, delta_initial, lambda_initial, delta_std_proposal, lambda_std_proposal, A, data):
    samples = np.zeros((n,2))

    delta_current = delta_initial
    lambda_current = lambda_initial

    neglogpi_current = neglogpi_theta(delta_current, lambda_current, A, observed_data)

    accepted_count = 0

    for i in range(n):
        delta_candidate = delta_current + np.random.normal(0, delta_std_proposal)
        lambda_candidate = lambda_current + np.random.normal(0, lambda_std_proposal)

        if (delta_candidate > 0 and lambda_candidate > 0):
            neglogpi_candidate = neglogpi_theta(delta_candidate, lambda_candidate, A, observed_data)

            log_alpha = neglogpi_current - neglogpi_candidate

            if (np.log(np.random.uniform(0,1)) < log_alpha):
                delta_current = delta_candidate
                lambda_current = lambda_candidate
                neglogpi_current = neglogpi_candidate
                accepted_count += 1

        samples[i]= [delta_current, lambda_current]
    print(f"Metropolis-Hastings acceptance rate: {accepted_count / n : .2%}")
    return samples

def autocorrelation_1d(samples, max_lag = 50): 
    samples = np.asarray(samples)
    samples = samples - np.mean(samples)

    autocorrelation = []

    denonimator = np.sum(samples*samples)
    for s in range(max_lag+1):
        if s == 0: 
            ac = 1.0
        else: 
            ac = np.sum(samples[:-s] * samples[s:]) / denonimator
        autocorrelation.append(ac)

    return np.array(autocorrelation)

def plot_ac(samples, max_lag=50, labels=None):
    samples = np.asarray(samples)
    dim = samples.shape[1]

    plt.figure(figsize=(8,5))

    for d in range(dim):
        if(labels and len(labels) == dim):
            label = labels[d]
        else: 
            label = f"dim {d+1}"

        plt.plot(range(max_lag+1), autocorrelation_1d(samples[:,d], max_lag), label = label)

    plt.xlabel("lag")
    plt.ylabel("autocorrelation")
    plt.title("autocorrelation per parameter")
    plt.grid(True, linestyle = ":", alpha = 0.6)
    plt.legend()
    plt.show()

       


n = 10000 # num of samples
delta_initial = 0.02
delta_std_proposal = 0.002
lambda_initial = 5
lambda_std_proposal = 0.4


samples = metropolis_hastings(n, delta_initial, lambda_initial, delta_std_proposal, lambda_std_proposal, A, observed_data)
plot_ac(samples, max_lag=500,labels = [r'$\delta$ (Prior Precision)', r'$\lambda$ (Noise Precision)'])
 
