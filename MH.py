import numpy as np
import matplotlib.pyplot as plt

def metropolis_hastings(target_distribution, proposal_distribution, density_q, x_0, iter):

    samples = np.zeros(iter)
    samples[0] = x_0

    accepted = 0
    for i in range(1, iter):
        current_sample = samples[i - 1]

        proposal = proposal_distribution(current_sample)
        acceptance_ratio = (target_distribution(proposal) * density_q(current_sample, proposal)) / (target_distribution(current_sample) * density_q(proposal, current_sample))
        acceptance_prob = min(1, acceptance_ratio)

        if np.random.rand() < acceptance_prob:
            samples[i] = proposal
            accepted += 1
        else:
            samples[i] = current_sample

    return samples, accepted/(iter - 1)  # return samples and acceptance rate


# switch up target distribution, proposal distribution, and proposal density function

# normal
def target_distribution_normal(x):
    return np.exp(-0.5 * x**2)  # standard normal distribution


# bimodal
def target_distribution_bimodal(x):
    return (
        0.8 * np.exp(-0.5 * ((x + 2) / 0.8)**2)
        + 0.2 * np.exp(-0.5 * ((x - 3) / 0.5)**2)
    )


def proposal_distribution_normal(x):
    return np.random.normal(x, 1)  # normal proposal distribution with mean x and std 1
#how acceptance rate changes with different mean and std. (plot figure out)

def q_normal(proposed, current):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (proposed - current)**2)  # normal density function




# def proposal_distribution(x):
#     return np.random.normal(x, 1.5)

# def density_q(y, x):
#     return (1 / (np.sqrt(2 * np.pi) * 1.5)) * np.exp(
#         -0.5 * ((y - x) / 1.5)**2
    # )

# plot stuff
def plot_samples(samples, acceptance_rate):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(samples)), samples, s=3, alpha=0.3)
    plt.title('m-h samples')
    plt.xlabel('iter')
    plt.ylabel('sample value')

    plt.subplot(1, 2, 2)
    plt.hist(samples, bins=30, density=True, alpha=0.6, label='samples')
    
    x = np.linspace(-4, 4, 100)
    plt.plot(x, target_distribution_bimodal(x), 'r', label='target distribution')
    
    plt.xlabel('value')
    plt.ylabel('density')
    plt.legend()
    
    print(f'acceptance Rate: {acceptance_rate:.2f}')
    
    plt.tight_layout()
    plt.show()

def mh_progress_1d(samples, step=50):
    y_min, y_max = samples.min() - 0.5, samples.max() + 0.5

    for end in range(step, len(samples) + 1, step):
        current_samples = samples[:end]

        plt.figure(figsize=(8, 4))
        plt.scatter(range(end), current_samples, s=12)
        plt.title(f'first #{end} iterations')
        plt.xlabel('iter')
        plt.ylabel('sample')
        plt.ylim(y_min, y_max)

        plt.show()

def mh_1d_plot(samples, target_distribution, step=50, pause_time=1):
    x_min, x_max = samples.min() - 1, samples.max() + 1
    x_grid = np.linspace(x_min, x_max, 500)
    target_vals = target_distribution(x_grid)

    # normalize target distribution (match histogram)
    area = np.trapezoid(target_vals, x_grid)
    target_vals = target_vals / area


    plt.figure(figsize=(10, 5))

    for end in range(step, len(samples) + 1, step):
        current_samples = samples[:end]

        plt.clf()

        # samples collected till now
        plt.hist(
            current_samples,
            bins=30,
            density=True,
            alpha=0.5,
            label=f'Samples so far: {end}'
        )

        # target distribution curve
        plt.plot(
            x_grid,
            target_vals,
            linewidth=2,
            label='target distribution'
        )

        # rug plot of samples till now
        y_scatter = np.zeros_like(current_samples) - 0.02
        plt.scatter(
            current_samples,
            y_scatter,
            s=12,
            alpha=0.5,
            label='Sample locations'
        )

        # current sample
        plt.scatter(
            current_samples[-1],
            -0.04,
            s=80,
            label='Current sample'
        )

        plt.title(f'First {end} Iterations')
        plt.xlabel('sample value')
        plt.ylabel('density')
        plt.xlim(x_min, x_max)
        plt.ylim(-0.08, max(target_vals) * 1.2)
        plt.legend()
        plt.pause(pause_time)

    plt.show()

samples, acceptance_rate = metropolis_hastings(
    target_distribution_bimodal,
    proposal_distribution_normal,
    q_normal,
    x_0=10,
    iter=100000
)

plot_samples(samples, acceptance_rate)

mh_1d_plot(samples, target_distribution_bimodal, step=500, pause_time=0.5)

# TODO: 2d donut, banana 3d(shell & ring) (ring wirdth) auto-correction time**
# github