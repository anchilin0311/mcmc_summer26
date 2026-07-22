import numpy as np
import matplotlib.pyplot as plt

x0_2d= np.array([5.0,5.0])
x0_3d = np.array([5.0, 5.0, 5.0])

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

def sample_normal_3d_x_given_yz(current):
    y = current[1]
    z = current[2]

    mean = rho/(1+rho)*(y+z)
    sd = np.sqrt((1-rho)*(1+2*rho)/(1+rho))

    return np.random.normal(mean, sd)


def sample_normal_3d_y_given_xz(current):
    x = current[0]
    z = current[2]

    mean = rho/(1+rho)*(x+z) 
    sd = np.sqrt((1-rho)*(1+2*rho)/(1+rho)) 

    return np.random.normal(mean, sd)


def sample_normal_3d_z_given_xy(current):
    x = current[0]
    y = current[1]

    mean = rho/(1+rho)*(x+y)
    sd = np.sqrt((1-rho)*(1+2*rho)/(1+rho))

    return np.random.normal(mean, sd)



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

#autocorrelation with fft

def autocorrelation_1d(samples,max_lag=1000):
    samples = np.asarray(samples).reshape(-1)

    if len(samples) >= 2:
        max_lag = min(len(samples)-1, max_lag)

        center = samples - np.mean(samples)

        ac = np.zeros(1+max_lag)
        ac[0]= 1.0

        for i in range(1, 1+max_lag):
            ac[i] = np.sum(center[:-i] * center[i:])/np.sum(center**2)
        
        return ac
    

def integrated_ac(samples, max_lag=1000, c=5):
    ac = autocorrelation_1d(samples, max_lag=max_lag)

    taus = np.ones(len(ac))

    for i in range(1, len(ac)):
        taus[i] = taus[i-1]+2*ac[i]

    window = None

    for i in range(1, len(taus)):
        if i >= c*taus[i]:
            window = i
            break

    if window is None:
        print(f'no valid window found till lag{max_lag}')
        return np.nan, None, ac, taus
    
    tau = taus[window]
    return tau, window, ac, taus

def calc_tau_nd(samples, burn_in=0, max_lag=1000, c=5):
    samples= np.asarray(samples)

    if (burn_in >= len(samples)-1):
        print(f'too many burn-in')
    else:
        samples = samples[burn_in:]
        results = []

        for d in range(samples.shape[1]):
            tau, window, ac, taus = integrated_ac(samples[:, d], max_lag=max_lag, c=c)

            if tau > 0:
                ess = len(samples)/tau
            else:
                ess = np.nan
            
            results.append({'dim': d, 'tau': tau, 'window': window, 'effective_sample_size':ess, 'autocorrelation':ac, 'taus': taus})
            
            print(f'dimension {d}: tau = {tau:.3f}, window = {window}, effective sample size = {ess:.3f}')


        return results

#autocorrealtion plots
def ac_np_plot(taus, max_lag=100):
    fig, axes = plt.subplots(len(taus), 1, figsize=(10,3*len(taus)), sharex=True)

    if len(taus)==1: 
        axes = [axes]

    for tau, axis in zip(taus, axes):
        ac = tau['autocorrelation']
        max_lag = min(len(ac)-1, max_lag)
        
        axis.plot(np.arange(max_lag+1), ac[:max_lag+1], marker='o')
        axis.axvline(tau['window'], label=f'window = {tau['window']}')

        axis.set_ylabel(f'autocorrelation dim{tau['dim']}')
        axis.set_title(f'dim {tau['dim']}: tau = {tau['tau']:.3f}')
        axis.legend()
    
    axes[-1].set_xlabel('lag')

    fig.suptitle('autocorrelation by coordinate')
    fig.tight_layout()

    plt.show()



#run gibbs
#2d
cond_dist_2d = [sample_normal_2d_x_given_y,sample_normal_2d_y_given_x]
samples_2d = gibbs_sampling_nd(cond_dist=cond_dist_2d, x_0=x0_2d, iter = 100000)

# gibbs_2d_scatter(samples, step = 100, bins = 80, pause_time= 0.3)
tau_2d = calc_tau_nd(samples_2d, burn_in=100, max_lag=2000)
ac_np_plot(tau_2d, max_lag=40)

#3d
cond_dist_3d = [sample_normal_3d_x_given_yz, sample_normal_3d_x_given_yz, sample_normal_3d_z_given_xy]
samples_3d = gibbs_sampling_nd(cond_dist=cond_dist_3d, x_0 = x0_3d, iter = 100000)

tau_3d = calc_tau_nd(samples_3d, burn_in = 100, max_lag = 2000)
ac_np_plot(tau_3d, max_lag=40)



