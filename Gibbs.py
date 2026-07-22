import numpy as np
import matplotlib.pyplot as plt

x0= np.array([5.0,5.0])

def gibbs_sampling_nd(cond_dist, x_0, iter):
    samples = np.zeros((iter, len(x_0)))
    samples[0] =x_0

    for i in range(1, iter):
        current = samples[i-1].copy()

        for dim in range(len(x_0)):
            current[dim]=cond_dist[dim](current)
        samples[i] = current

    return samples


#correlated 2d normal dist

rho = .8

#sample x conditional on current val of y
def sample_normal_2d_x_given_y(current):
    y = current[1]

    mean = rho*y
    sd = np.sqrt(1-rho**2)

    return np.random.normal(mean,sd)

def sample_normal_2d_y_given_x(current):
    x = current[0]

    mean = rho*x
    sd = np.sqrt(1-rho**2)

    return np.random.normal(mean,sd)

#plotting
def gibbs_2d_scatter(samples, step = 100, bins = 80, pause_time = .3):
    x_samples = samples[:,0]
    y_samples = samples[:,1]

    x_min = x_samples.min()-1
    x_max = x_samples.max()+1
    y_min = y_samples.min()-1
    y_max = y_samples.max()+1


    fig = plt.figure(figsize=(14,8))

    graphs = fig.add_gridspec(2,2,width_ratios = (4,1), height_ratios = (1,4))

    scatter_plot = fig.add_subplot(graphs[1,0])
    x_hist = fig.add_subplot(graphs[0,0])
    y_hist = fig.add_subplot(graphs[1,1])

    for s in range(step,len(samples) + step, step):
        s = min(s, len(samples))

        current = samples[:s]

        x_current = current[:,0]
        y_current = current[:,1]

        scatter_plot.clear()
        x_hist.clear()
        y_hist.clear()

        scatter_plot.scatter(x_current,y_current, s=5, alpha=.5)

        scatter_plot.scatter(x_current[-1], y_current[-1], s=100, marker = 'o', label='current')
        scatter_plot.set_xlabel('x')
        scatter_plot.set_ylabel('y')
        scatter_plot.set_xlim(x_min, x_max)
        scatter_plot.set_ylim(y_min, y_max)

        x_hist.hist(x_current, bins = bins, density=True, alpha = .5)
        y_hist.hist(y_current, bins=bins, density=True, alpha=.5, orientation='horizontal')

        x_hist.set_xlim(x_min, x_max)
        x_hist.set_ylabel('density')

        y_hist.set_ylim(y_min, y_max)
        y_hist.set_ylabel('density')

        fig.suptitle(f'gibbs sampling till {s} iterations')

        plt.pause(pause_time)

        if s == len(samples): break

    plt.show()

#run gibbs
cond_dist = [sample_normal_2d_x_given_y,sample_normal_2d_y_given_x]

samples = gibbs_sampling_nd(cond_dist=cond_dist, x_0=x0, iter = 100000)

gibbs_2d_scatter(samples, step = 100, bins = 80, pause_time= 0.3)