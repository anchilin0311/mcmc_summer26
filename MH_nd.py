import numpy as np
import matplotlib.pyplot as plt

# starting point (need to fix the dimension to match M-H)
# x0 = np.array([10, 10]) # now just 2d
x0 = np.array([55,0,0])

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

#closer modes -> smaller tau
def target_distribution_bimodal_v2_3d(x):
    m1 = np.exp(-0.5 * (((x[0] + 1) / 0.8)**2 + ((x[1] + 1) / 0.8)**2 + ((x[2] + 1) / 0.8)**2))

    m2 = np.exp(-0.5 * (((x[0] - 1) / 0.6)**2 + ((x[1] - 1) / 0.6)**2 + ((x[2] - 1) / 0.6)**2))

    return 0.7 * m1 + 0.3 * m2 

def target_distribution_bimodal_v3_3d(x):
    center1 = [-50,-50,-50]
    center2 = [1,1,1]
    m1 = np.exp(-0.5 * (((x[0] - center1[0]) / 0.8)**2 + ((x[1] - center1[1]) / 0.8)**2 + ((x[2] + center1[2]) / 0.8)**2))

    m2 = np.exp(-0.5 * (((x[0] - center2[0]) / 0.6)**2 + ((x[1] - center1[1]) / 0.6)**2 + ((x[2] - center2[2]) / 0.6)**2))

    return 0.7 * m1 + 0.3 * m2 

def mode_percentage(samples, mode1, mode2):
    count1=0
    count2=0

    for s in samples:
        dist_to_mode1 = np.linalg.norm(s - mode1)
        dist_to_mode2 = np.linalg.norm(s - mode2)

        if dist_to_mode1 < dist_to_mode2:
            count1 += 1
        else: 
            count2 += 1

    print(f"percentage around {mode1} is {count1/len(samples)}" )
    print(f"percentage around {mode2} is {count2/len(samples)}")
    
    return count1/len(samples), count2/len(samples)


# donut 2d
def target_distribution_donut_2d(x, R=3, sigma=0.4):
    r = np.sqrt(x[0]**2 + x[1]**2)
    return np.exp(-0.5 * ((r - R) / sigma)**2)

# donut 3d
def target_distribution_donut_3d(x, R=50, r=5, sigma=.25):
    rho = np.sqrt(x[0]**2 + x[1]**2)
    distance_from_surface = np.sqrt((rho - R)**2 + x[2]**2) - r
    return np.exp(-0.5 * (distance_from_surface / sigma)**2)

# general radius and angle function
def donut_r_theta(samples):
    x = samples[:,0]
    y = samples[:,1]

    r = np.sqrt(x**2+y**2)
    theta = np.arctan2(y,x)

    return r, theta


#donut keep in 3d but has other random dimension in there (or a recantangle long and thin)
#fix ratio change dimension AND fix dimension change length & thickness. 
# cross entropy? 

# proposal distributions: 

# standard normal 2d
def proposal_distribution_normal_nd(x):
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
        x_hist.set_xlim(x_min, x_max)
        # x_hist.set_ylim(x_min, 1)

        #y marginal
        y_hist.hist(y_current, bins=bins, density=True, alpha=0.5, orientation = "horizontal")
        y_hist.set_xlabel("density")
        y_hist.set_ylabel("y")
        y_hist.set_ylim(y_min, y_max)

        #z marginal
        z_hist.hist(z_current, bins=bins, density=True, alpha=0.5, orientation = "vertical")
        z_hist.set_xlabel("z")
        z_hist.set_ylabel("density")
        z_hist.set_xlim(z_min, z_max)

        fig.suptitle(f"Samples till now: {i}")

        plt.pause(pause_time)


    plt.tight_layout()
    plt.show()


# autocorrelation

"""def autocorrelation(samples, step = 1):
    x = np.asarray(samples) - np.mean(samples)

    autocorrelation = []

    if step == 0, then correlation = 1
    if step != 0, then correlation = sum(x[i] * x[i + step]) / sum(x[i]**2)
    add that correlation to autocorrelation [] and return it as an array? but how to visualize it and do we have to do it every single step?
    since my plotting visualization is  multiple steps (like 30-300) 
    
    TODO: also how to do this autocorrelation in higher dimension? 

    cross entropy? cross autocorrelation?

    look into FFT for autocorrelation (faster speed)

    multimodal correlation time vs how far they are:)
    donut, R bigger

    
    sigma: too small & too large -> high auto correlation; 
           medium -> low autocorrelation
    R: larger -> slower angular exploring (aka. higher angular autocorrelation time)
    r: smaller (narrower tube) -> lower acceptance rate (aka. higher autocorrelation time)
       higher -> easier movement -> more space to explore (aka. may increase global mixing time)

"""

#autocorrelation
def autocorrelation_1d(samples, step = 1): 
    samples = np.asarray(samples)
    samples = samples - np.mean(samples)

    autocorrelation = []

    for s in range(step+1):
        if s == 0: 
            ac = 1.0
        else: 
            ac = np.sum(samples[:-s] * samples[s:]) / np.sum(samples*samples)
        autocorrelation.append(ac)

    return np.array(autocorrelation)

def plot_ac_1d(samples, step = 1):
    plt.figure(figsize=(8,6))
    plt.plot(range(step+1), autocorrelation_1d(samples, step = step))
    
    plt.xlabel("steps")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation plot")

    plt.legend()
    plt.show()

def plot_ac_nd(samples, step = 1):
    samples = np.asarray(samples)
    dim = samples.shape[1]

    plt.figure(figsize=(8,6))

    for d in range(dim):
        plt.plot(range(step+1), autocorrelation_1d(samples[:,d], step = step), label=f"dim {d+1}")
    
    plt.xlabel("steps")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation per dimension")

    plt.legend()
    plt.show()

#find the window and calculate tau TODO: redo this check!!
def ac_window(taus, c = 5):
    #use the sokal rule (stop at first M where M>=c*tau(M))
    taus = np.asarray(taus).reshape(-1)

    for m in range(1, len(taus)):
        if m >= c*taus[m]:
            return m
    # return len(taus)-1 #return last possible lag if never find a good stopping point
    return None #avliable autocorrelaiton sequence is too short to find a valid good window

def tau_1d(samples, c, max_lag = 1000):

    max_lag = min(max_lag, len(samples)-1)

    ac = autocorrelation_1d(samples, step=max_lag)
    ac = np.asarray(ac).reshape(-1)
    taus = np.zeros(len(ac))

    for m in range(len(ac)):
        taus[m]=1+2*np.sum(ac[1:m+1])

    window = ac_window(taus,c)

    if window is None:
        print(f'no valid window found through lag {max_lag}\n'
              f'last run estimate = {taus[-1]:.4f}')
        return np.nan
        
    tau = taus[window]
    return float(tau)

# autocorrelation of radius vs angle (and also z if >=3 dim)
def plot_donut_ac(samples, lag = 100):
    r,theta = donut_r_theta(samples)

    ac_r = autocorrelation_1d(r, lag)
    ac_cos = autocorrelation_1d(np.cos(theta), lag)
    ac_sin = autocorrelation_1d(np.sin(theta), lag)

    lags = np.arange(lag+1)

    plt.figure(figsize=(8,6))
    plt.plot(lags,ac_r, label="radius")
    plt.plot(lags, ac_cos, label="cos(theta)")
    plt.plot(lags, ac_sin, label="sin(theta)")

    if samples.shape[1] >=3:
        ac_z = autocorrelation_1d(samples[:,2], lag)
        plt.plot(lags, ac_z, label="z")

    plt.xlabel("lag")
    plt.ylabel("autocorrelation")
    plt.title("radius vs angle autocorrelation")

    plt.legend()
    plt.show()


samples, acceptance_rate = metropolis_hastings(
    target_distribution_donut_3d,
    proposal_distribution_normal_nd,
    q_standard_normal_2d,
    x_0=x0,
    iter=100000
)

theta = np.arctan2(samples[:, 1], samples[:, 0])
plot_ac_1d(theta, step=1000)

plot_ac_nd(samples,step = 1000)


# mh_3d_plot(samples, step=500, bins=90, pause_time=0.05)

#print tau
for d in range(samples.shape[1]):
    tau = tau_1d(samples[:,d],c=5, max_lag= 20000)
    print(f"dimension {d}: tau = {tau:.4f}")


#print mode percetage for bimodal
# mode_percentage(samples, [-2,-2,-2], [3,3,1])

# autocorrelation for 2d, 3d donuts (radius, theta, z)
# plot_donut_ac(samples, lag=100)