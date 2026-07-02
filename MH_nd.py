import numpy as np
import matplotlib.pyplot as plt

# starting point (need to fix the dimension to match M-H)
# x0 = np.array([10, 10]) # now just 2d
x0 = np.array([0,0,0])

def metropolis_hastings(target_distribution, proposal_distribution, q, x_0, iter):

    samples = np.zeros((iter, len(x_0)))
    samples[0] = x_0

    accepted = 0

    for i in range(1, iter):
        current_sample = samples[i - 1]

        proposal = proposal_distribution(current_sample)
        acceptance_ratio = (target_distribution(proposal) * q(current_sample, proposal)) / (target_distribution(current_sample) * q(proposal, current_sample))
        acceptance_prob = min(1, acceptance_ratio)

        if np.random.rand() < acceptance_prob:
            samples[i] = proposal
            accepted += 1
        else:
            samples[i] = current_sample

    return samples, accepted/(iter - 1)  # return samples and acceptance rate


# target distributions: 

# standard normal 2d
def target_distribution_normal_2d(x):
    return np.exp(-0.5 * (x[0]**2 + x[1]**2))

# bimodal 2d
def target_distribution_bimodal_2d(x):
    return (0.8 * np.exp(-0.5 * ((x[0] + 2) / 0.8)**2) * np.exp(-0.5 * ((x[1] + 2) / 0.8)**2)
        + 0.2 * np.exp(-0.5 * ((x[0] - 3) / 0.5)**2) * np.exp(-0.5 * ((x[1] - 3) / 0.5)**2))

# bimodal 3d
"""
the center should have two modes (-2,-2,-2) and (3,3,1), with different weights. 
the samples would only concentrate around the mode that x0 is closer to. 
(play around x0=[0,0,0], x0 = [-1,0,0], and x0=[5,0,0])
"""
def target_distribution_bimodal_3d(x):
    m1 = np.exp(-0.5 * (((x[0] + 2) / 0.8)**2 + ((x[1] + 2) / 0.8)**2 + ((x[2] + 2) / 0.8)**2))

    m2 = np.exp(-0.5 * (((x[0] - 3) / 0.6)**2 + ((x[1] - 3) / 0.6)**2 + ((x[2] - 1) / 0.6)**2))

    return 0.7 * m1 + 0.3 * m2


# donut 2d
def target_distribution_donut_2d(x, R=3, sigma=0.4):
    r = np.sqrt(x[0]**2 + x[1]**2)
    return np.exp(-0.5 * ((r - R) / sigma)**2)

# donut 3d
def target_distribution_donut_3d(x, R=3, r=1, sigma=0.25):
    rho = np.sqrt(x[0]**2 + x[1]**2)
    distance_from_surface = np.sqrt((rho - R)**2 + x[2]**2) - r
    return np.exp(-0.5 * (distance_from_surface / sigma)**2)


# proposal distributions: 

# standard normal 2d
def proposal_distribution_normal_2d(x):
    return np.random.normal(x, 1, size=x.shape)  # normal proposal distribution with mean x and std 1


# density functions: 

# standard normal 2d
def q_standard_normal_2d(proposing, current):
    return (1 / (2 * np.pi)) * np.exp(-0.5 * np.sum((proposing - current)**2))  # multivariate normal density function


# plotting

def mh_2d_plot(samples, target_distribution, step=50, bins = 100, pause_time=0.5):
    x_samples = samples[:, 0]
    y_samples = samples[:, 1]

    # Set plot range based on samples
    x_min, x_max = x_samples.min() - 1, x_samples.max() + 1
    y_min, y_max = y_samples.min() - 1, y_samples.max() + 1

    fig = plt.figure(figsize=(10, 8))

    graphs = fig.add_gridspec(2, 2, width_ratios = (4,1), height_ratios = (1,4), hspace = 0.05, wspace = 0.05)
    x_scatter = fig.add_subplot(graphs[1,0])
    x_hist = fig.add_subplot(graphs[0,0])
    y_hist = fig.add_subplot(graphs[1,1], sharey = x_scatter)
    
    
    for i in range(step, len(samples), step):

        current = samples[:i]

        x_current = current[:,0]
        y_current = current[:,1]

        x_scatter.clear()
        x_hist.clear()
        y_hist.clear()
        

        x_scatter.scatter(x_current, y_current, s=5, alpha=0.5, marker = 'o', color='orange', label = "current sample")
        x_scatter.set_xlabel("x")
        x_scatter.set_ylabel("y")
        x_scatter.set_xlim(x_min, x_max)
        x_scatter.set_ylim(y_min, y_max)
        x_scatter.legend()

        x_hist.hist(x_current, bins=bins, density=True, alpha=0.5, orientation='vertical', label = f"samples till now: ")
        y_hist.hist(y_current, bins=bins, density=True, alpha=0.5, orientation='horizontal')
        y_hist.set_xlabel("density")
        x_hist.set_xlim(x_min, x_max)
        y_hist.set_ylim(y_min, y_max)

        plt.pause(pause_time)

    plt.show()

def mh_3d_plot(samples, step=20, bins=80, pause_time=0.5):
    x_samples = samples[:, 0]
    y_samples = samples[:, 1]
    z_samples = samples[:, 2]

    x_min, x_max = x_samples.min() - 1, x_samples.max() + 1
    y_min, y_max = y_samples.min() - 1, y_samples.max() + 1
    z_min, z_max = z_samples.min() - 1, z_samples.max() + 1

    fig = plt.figure(figsize=(10, 8))

    graphs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), hspace=0.05, wspace=0.05)

    # 3D scatter plot
    plot_3d = fig.add_subplot(graphs[1, 0], projection='3d')
    # x_scatter = fig.add_subplot(graphs[1, 0])
    x_hist = fig.add_subplot(graphs[0, 0])
    y_hist = fig.add_subplot(graphs[1, 1])
    z_hist = fig.add_subplot(graphs[0, 1])

    for i in range(step, len(samples), step):
        current = samples[:i]

        x_current = current[:, 0]
        y_current = current[:, 1]
        z_current = current[:, 2]

        plot_3d.clear()
        x_hist.clear()
        y_hist.clear()
        z_hist.clear()


        plot_3d.scatter(
            x_current,
            y_current,
            z_current,
            s=3,
            alpha=0.5, 
            marker = "o",
            label = "samples",
            color = "orange")
        
        plot_3d.scatter(
            x_current[-1],
            y_current[-1],
            z_current[-1],
            s=100,
            color = "purple",
            label = "current",
            marker = "o"
        )
        
        plot_3d.set_xlabel("x")
        plot_3d.set_ylabel("y")
        plot_3d.set_zlabel("z")
        # plot_3d.set_title("3D MH")

        plot_3d.set_xlim(x_min, x_max)
        plot_3d.set_ylim(y_min, y_max)
        plot_3d.set_zlim(z_min, z_max)
        plot_3d.legend()

        # x marginal
        x_hist.hist(x_current, bins=bins, density=True, alpha=0.5, orientation = "vertical")
        x_hist.set_xlabel("x")
        x_hist.set_ylabel("density")
        # x_hist.set_xlim(x_min, x_max)
        x_hist.set_ylim(0, 1)

        #y marginal
        y_hist.hist(y_current, bins=bins, density=True, alpha=0.5, orientation = "horizontal")
        y_hist.set_xlabel("density")
        y_hist.set_ylabel("y")
        y_hist.set_ylim(y_min, y_max)

        #z marginal
        z_hist.hist(z_current, bins=bins, density=True, alpha=0.5, orientation = "vertical")
        z_hist.set_xlabel("z")
        z_hist.set_ylabel("density")
        z_hist.set_ylim(0, 1)

        fig.suptitle(f"Samples till now: {i}")

        plt.pause(pause_time)


    plt.tight_layout()
    plt.show()

samples, acceptance_rate = metropolis_hastings(
    target_distribution_bimodal_3d,
    proposal_distribution_normal_2d,
    q_standard_normal_2d,
    x_0=x0,
    iter=100000
)

# mh_2d_plot(
#     samples,
#     target_distribution_donut_2d,
#     step=500,
#     pause_time=0.2
# )

mh_3d_plot(samples, step=3000, bins=1000, pause_time=0.1)