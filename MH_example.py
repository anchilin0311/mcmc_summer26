import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular
import time

np.random.seed(42)

N = 80
x = np.linspace(0, 1, N, endpoint=False)
kernel_size = N
kernel_gamma = 0.03

# noise precision
lambda_true = 5.35

# hyperprior parameters
alpha_del = 1
beta_del = 1e-4
alpha_lam = 1
beta_lam = 1e-4

def true_signal(x): # smooth gaussian bump and two sharp step-function blocks
    return 50*np.exp(-((x-0.75)/0.1)**2) + 37*np.logical_and(x>0.1,x<0.25) + 13*np.logical_and(x>0.3,x<0.32) 

# x is grid
def generate_blurred_data(true_signal, x, A, lam):
    noise_std = 1/np.sqrt(lam) # lam: precision
    blurred_signal = np.dot(A,true_signal(x))
    noise = np.random.normal(0, noise_std, len(x)) # eps ~ N(0,noise_std I)

    return blurred_signal + noise

def gaussian_kernel(x, gamma):
    return np.exp(-x**2 / (2 * gamma**2))/np.sqrt(np.pi*gamma**2)

def blur_matrix(dim, gamma):
    A = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(dim):
            d = min(np.abs(i-j), dim - np.abs(i-j))
            A[i, j] = gaussian_kernel(d/dim, gamma) / dim

    A = A / np.sum(A[0, :]) 

    return A


def prior_matrix(dim):
    L = (2*np.eye(dim) - np.eye(dim, k=1) - np.eye(dim, k=-1))

    return L
 

def neglogpi_theta(delta, lam, A, L, data):
    dim = len(data)

    # scaled prior precision
    L = delta*L
    
    # compute posterior precision and mean
    Lpost = lam*np.dot(np.transpose(A),A) + L
    xpost = lam*np.dot(np.transpose(A),data) # rhs for posterior mean
    xpost = np.linalg.solve(Lpost, xpost) # conditional posterior mean of the image
    
    # compute negative log of determinant ratio
    (sdetL, logdetL) = np.linalg.slogdet(L)
    (sdetLpost, logdetLpost) = np.linalg.slogdet(Lpost)

    if (sdetL <= 0 or sdetLpost <= 0):
        return np.inf

    det_ratio = 0.5*logdetLpost - 0.5*logdetL - dim/2*np.log(lam)
    
    # compute data, posterior mean, and hyperprior terms
    uQu_term = - 0.5*np.dot(xpost, np.dot(Lpost, xpost)) # posterior mean term
    yQy_term = 0.5*lam*np.dot(data,data) # data term
    prior_term = beta_del*delta + beta_lam*lam - (alpha_del-1)*np.log(delta) - (alpha_lam-1)*np.log(lam) # hyperprior term
    
    return det_ratio+uQu_term+prior_term+yQy_term


def shrink_fine_to_loose(data_fine, dim_loose):
    dim_fine = len(data_fine)

    grid_fine = np.linspace(0, 1, dim_fine, endpoint=False)
    grid_loose = np.linspace(0, 1, dim_loose, endpoint=False)

    data_loose = np.interp(grid_loose, grid_fine, data_fine, period=1.0)

    return data_loose



def metropolis_hastings(n, delta_initial, lambda_initial, delta_std_proposal, lambda_std_proposal, A, L, data):
    samples = np.zeros((n,2))

    delta_current = delta_initial
    lambda_current = lambda_initial

    neglogpi_current = neglogpi_theta(delta_current, lambda_current, A, L, data)

    accepted_count = 0

    for i in range(n):
        delta_candidate = (delta_current + np.random.normal(0, delta_std_proposal))
        lambda_candidate = (lambda_current + np.random.normal(0, lambda_std_proposal))

        if (delta_candidate > 0 and lambda_candidate > 0):
            neglogpi_candidate = neglogpi_theta(delta_candidate, lambda_candidate, A, L, data)

            log_alpha = min(0.0, neglogpi_current - neglogpi_candidate)

            if (np.log(np.random.uniform(0,1)) < log_alpha):
                delta_current = delta_candidate
                lambda_current = lambda_candidate
                neglogpi_current = neglogpi_candidate
                accepted_count += 1

        samples[i]= [delta_current, lambda_current]
    print(f"Metropolis-Hastings acceptance rate: {accepted_count / n : .2%}")
    
    return samples

def mh_delayed_acceptance(n, delta_initial, lambda_initial, delta_std_proposal, lambda_std_proposal, A_fine, A_loose, L_fine, L_loose, data_fine, data_loose):
    samples = np.zeros((n, 2))

    delta_current = delta_initial
    lambda_current = lambda_initial

    fine_current = neglogpi_theta(delta_current, lambda_current, A_fine, L_fine, data_fine)
    loose_current = neglogpi_theta(delta_current, lambda_current, A_loose, L_loose, data_loose)

    proposals_valid = 0
    accepted_1 = 0
    accepted_2 = 0

    fine_eval_count = 1

    for i in range(n):
        delta_candidate = (delta_current + np.random.normal(0, delta_std_proposal))
        lambda_candidate = (lambda_current + np.random.normal(0, lambda_std_proposal))

        if (delta_candidate <= 0 or lambda_candidate <= 0):
            samples[i] = [delta_current, lambda_current]
            continue

        proposals_valid += 1

        # stage 1:
        loose_candidate = neglogpi_theta(delta_candidate, lambda_candidate, A_loose, L_loose, data_loose)
        log_alpha_1 = min(0.0, loose_current - loose_candidate)

        if (np.log(np.random.uniform()) < log_alpha_1):
            accepted_1 += 1

            # stage 2
            fine_candidate = neglogpi_theta(delta_candidate, lambda_candidate, A_fine, L_fine, data_fine)
            fine_eval_count += 1

            log_alpha_2 = min(0.0, fine_current - fine_candidate + loose_candidate - loose_current)

            if (np.log(np.random.uniform()) < log_alpha_2):
                 delta_current = delta_candidate
                 lambda_current = lambda_candidate
                 loose_current = loose_candidate
                 fine_current = fine_candidate

                 accepted_2 += 1

        samples[i] = [delta_current, lambda_current]

    rate_1 = accepted_1 / n
    rate = accepted_2 / n

    if (rate_1 > 0):
        stage_2_given_stage_1_rate = accepted_2 / accepted_1
    else: 
        stage_2_given_stage_1_rate = 0.0

    print(f"Stage 1 acceptance rate: {rate_1:.2f}")
    print(f"Stage 2 acceptance among accepted in stage 1: {stage_2_given_stage_1_rate:.2f}")
    print(f"Overall acceptance rate: {rate:.2f}")
    print(f"Amount of fine posterior evaluations: {fine_eval_count}")

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

def ac_window(taus, c = 5):
    #use the sokal rule (stop at first M where M>=c*tau(M))
    taus = np.asarray(taus).reshape(-1)

    for m in range(1, len(taus)):
        if(taus[m] > 0 and m >= c*taus[m]):
            return m
        
    return None #avliable autocorrelaiton sequence is too short to find a valid good window

# estimate integrated ac time of 1d mcmc chain
def tau_1d(samples, c, max_lag = 1000):

    samples = np.asarray(samples, dtype=float).reshape(-1)

    max_lag = min(max_lag, len(samples)-1)

    ac = autocorrelation_1d(samples, max_lag=max_lag)
    ac = np.asarray(ac, dtype=float).reshape(-1)

    taus = np.zeros(len(ac))

    for m in range(len(ac)):
        taus[m]=1.0+2.0*np.sum(ac[1:m+1])

    window = ac_window(taus,c)

    if window is None:
        print(f'no valid window found through lag {max_lag}\n'
              f'last run estimate = {taus[-1]:.4f}')
        return np.nan
        
    return float(taus[window])

def plot_ac(samples, max_lag=250, labels=None):
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

def time_vs_N_MH(Ns=[20, 40, 60, 80, 100, 120, 140, 160, 250], num_samples=300):
    runtimes = []

    for N_dim in Ns:
        x = np.linspace(0, 1, N_dim, endpoint=False)

        A = blur_matrix(N_dim, kernel_gamma)
        L = prior_matrix(N_dim)

        data = generate_blurred_data(true_signal, x, A, lambda_true)

        t0 = time.time()
        _ = metropolis_hastings(num_samples,delta_initial,lambda_initial,delta_std_proposal,lambda_std_proposal,A,L,data)
        t1 = time.time()

        elapsed = t1 - t0
        runtimes.append(elapsed)
        print(f"N = {N_dim:3d};  execution Time: {elapsed:.3f} s")

    # Plot timing curve
    plt.figure(figsize=(8, 5))
    plt.plot(Ns, runtimes, 'o-', color='crimson', linewidth=2, markersize=7, label='Measured sampling time')
    plt.title(f'execution time vs. dimension ($N$)\n[with {num_samples} MCMC iterations]')
    plt.xlabel('dimension ($N$)')
    plt.ylabel('seconds')
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return Ns, runtimes

def trace_plot(samples, title):
    fig, axes = plt.subplots(2, 1, figsize=(12,8), sharex=True)
    axes[0].plot(samples[:, 0])
    axes[0].set_ylabel(r"$\delta$")

    axes[1].plot(samples[:, 1])
    axes[1].set_ylabel(r"$\lambda$")
    axes[1].set_xlabel("iterations")

    fig.suptitle(title)
    plt.tight_layout
    plt.show()

def summary(samples, title, runtime, c=5, max_lag=1000):
    deltas = samples[:, 0]
    lambdas = samples[:,1]

    tau_delta = tau_1d(deltas, c=c, max_lag=max_lag)
    tau_lambda = tau_1d(lambdas, c=c, max_lag=max_lag)

    print(f"\n{title}")
    print(f"delta mean: {np.mean(deltas):.3f}")
    print(f"delta std: {np.std(deltas, ddof=1):.3f}")

    if (np.isnan(tau_delta)):
        print("delta tau, ess, ess/sec unavaliable")
    else: 
        ess_delta = len(deltas)/tau_delta
        ess_per_sec_delta = ess_delta/runtime

        print(f"delta tau: {tau_delta:.3f}")
        print(f"delta ess: {ess_delta:.3f}")
        print(f"delta ess/sec: {ess_per_sec_delta:.3f}")


    print(f"\n\nlambda mean: {np.mean(lambdas):.3f}")
    print(f"lambda std: {np.std(lambdas, ddof=1):.3f}")

    if (np.isnan(tau_lambda)):
        print("delta tau, ess, ess/sec unavaliable")
    else: 
        ess_lambda = len(lambdas)/tau_lambda
        ess_per_sec_lambda = ess_lambda/runtime

        print(f"lambda tau: {tau_lambda:.3f}")
        print(f"lambda ess: {ess_lambda:.3f}")
        print(f"lambda ess/sec: {ess_per_sec_lambda:.3f}")

n = 10000 # num of samples

delta_initial = 0.05
lambda_initial = 5.0

delta_std_proposal = 0.01
lambda_std_proposal = 0.5

n_loose = 159
N_fine = 160

# build fine problem
x_fine = np.linspace(0, 1, N_fine, endpoint=False)
A_fine = blur_matrix(N_fine, kernel_gamma)
L_fine = prior_matrix(N_fine)

data_fine = generate_blurred_data(true_signal, x_fine, A_fine, lambda_true)
data_loose = shrink_fine_to_loose(data_fine, n_loose)

x_loose = np.linspace(0, 1, n_loose, endpoint=False)
A_loose = blur_matrix(n_loose, kernel_gamma)
L_loose = prior_matrix(n_loose)

# samples = metropolis_hastings(n, delta_initial, lambda_initial, delta_std_proposal, lambda_std_proposal, A_fine, L_fine , data_fine)
# plot_ac(samples, max_lag=500,labels = [r'$\delta$ (Prior Precision)', r'$\lambda$ (Noise Precision)'])

# standard metropolis hastings
start_standard = time.perf_counter()
samples_standard = metropolis_hastings(n, delta_initial, lambda_initial, delta_std_proposal, lambda_std_proposal, A_fine, L_fine , data_fine)
runtime_standard = time.perf_counter() - start_standard


print(f"Standard MH runtime: {runtime_standard:.3f}")


# delayed acceptance mh
start_delayed = time.perf_counter()

samples_delayed = mh_delayed_acceptance(n, delta_initial, lambda_initial, delta_std_proposal, lambda_std_proposal, A_fine, A_loose, L_fine, L_loose, data_fine, data_loose)
runtime_delayed = time.perf_counter() - start_delayed

print(f"Delayed-acceptance MH runtime: {runtime_delayed:.3f} seconds. ")

print(f"Runtime ratio: {runtime_standard / runtime_delayed :.3f} times faster")


Ns, runtimes = time_vs_N_MH()
# why is accpetance rate high? 
# convergence plot (error in pi(theta; N) vs N)

# ??? test whether the coarse objective is reasonable
# test = [(0.01, 3.0), (0.02, 5.0), (0.03, 7.0), (0.04, 9.0)]

# for delta_test, lambda_test in test:
#     fine = neglogpi_theta(delta_test, lambda_test, A_fine, L_fine, data_fine)
#     loose = neglogpi_theta(delta_test, lambda_test, A_loose, L_loose, data_loose)

#     print(f"delta = {delta_test:.4f}")
#     print(f"lambda = {lambda_test:.4f}")
    # print(f"fine objective: {fine:.4f}")
    # print(f"loose objective: {loose:.4f}")

summary(samples_standard[200:], "Standard MH", runtime_standard, c=5, max_lag=1000)
summary(samples_delayed[200:], "Delayed-acceptance MH", runtime_delayed, c=5, max_lag=1000)

trace_plot(samples_standard[200:], "Standard MH trace plot")
trace_plot(samples_delayed[200:], "Delayed-acceptance MH trace plot")


labels = ["prior precision (delta)", "noise precision (lambda)"]

plot_ac(samples_standard[200:], max_lag = 500, labels = labels)
plot_ac(samples_delayed[200:], max_lag = 500, labels=labels)

# evaluate 2d at a grid (heat map) 159 vs. 160 save samples that we rejects. 
# 2 stage bug? approximation shrinking? 
