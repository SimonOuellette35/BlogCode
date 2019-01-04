import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

# First we generate the data
def generateDataV1(N):

    x = [1.0]
    y = [1.0]

    corr = 0.25

    for _ in range(N):

        x_innovation = np.random.normal(0., 0.1)
        x_tmp = x[-1] + x_innovation
        y_tmp = y[-1] + corr * x_innovation + np.random.normal(0., 0.1)

        x.append(x_tmp)
        y.append(y_tmp)

    return np.array(x), np.array(y)

def generateDataV2(N):

    x = [1.0]
    y = [1.0]

    corr = 0.25

    corrs = [corr]
    for _ in range(N):

        x_innovation = np.random.normal(0., 0.1)
        x_tmp = x[-1] + x_innovation
        y_tmp = y[-1] + corr * x_innovation + np.random.normal(0., 0.1)

        x.append(x_tmp)
        y.append(y_tmp)

        corr += np.random.normal(0., 0.01)
        corrs.append(corr)

    return np.array(x), np.array(y), np.array(corrs)

x, y = generateDataV1(10000)

plt.plot(x)
plt.plot(y)
plt.show()

x2, y2, corrs = generateDataV2(10000)

plt.plot(x2)
plt.plot(y2)
plt.show()

plt.plot(corrs)
plt.show()

# Now we model the V1 data, and examine the stability of the correlation
import theano.tensor as tt
with pm.Model() as model1:

    def custom_likelihood(x_diffs, y_obs_last, y_obs):

        expected = y_obs_last - corr * x_diffs
        return pm.Normal.dist(mu=expected, sd=0.1).logp(y_obs)

    corr = pm.Uniform('corr', lower=-1., upper=1., shape=1000)
    corr = tt.repeat(corr, 10)

    pm.DensityDist('obs', custom_likelihood, observed={
        'x_diffs': (x[:-1] - x[1:]),
        'y_obs_last': y[:-1],
        'y_obs': y[1:]
    })

    mean_field = pm.fit(n=5000, method='advi', obj_optimizer=pm.adam(learning_rate=0.01))
    trace = mean_field.sample(1000)

estimated_corrs = np.mean(trace['corr'], axis=0)

plt.plot(estimated_corrs)
plt.show()

# Now we model the V2 data, and examine the stability of the correlation
with pm.Model() as model2:

    def custom_likelihood(x_diffs, y_obs_last, y_obs):
        expected = y_obs_last - corr * x_diffs
        return pm.Normal.dist(mu=expected, sd=0.1).logp(y_obs)

    corr = pm.Uniform('corr', lower=-10., upper=10., shape=1000)
    corr = tt.repeat(corr, 10)

    pm.DensityDist('obs', custom_likelihood, observed={
        'x_diffs': (x2[:-1] - x2[1:]),
        'y_obs_last': y2[:-1],
        'y_obs': y2[1:]
    })

    mean_field = pm.fit(n=5000, method='advi', obj_optimizer=pm.adam(learning_rate=0.01))
    trace = mean_field.sample(1000)

estimated_corrs = np.repeat(np.mean(trace['corr'], axis=0), 10)

plt.plot(estimated_corrs, color='r')
plt.plot(corrs, color='b')
plt.show()

# Clearly in V2 the correlation is not stable. Let's model it as a random walk and examine the stability of its std. dev.
with pm.Model() as model3:

    def custom_likelihood(x_diffs, y_obs_last, y_obs):
        expected = y_obs_last - corr * x_diffs
        return pm.Normal.dist(mu=expected, sd=0.1).logp(y_obs)

    step_size = pm.Uniform('step_size', lower=0.0001, upper=0.1, shape=999)
    corr = pm.GaussianRandomWalk('corr', mu=0, sd=step_size, shape=1000)
    corr = tt.repeat(corr, 10)

    pm.DensityDist('obs', custom_likelihood, observed={
        'x_diffs': (x2[:-1] - x2[1:]),
        'y_obs_last': y2[:-1],
        'y_obs': y2[1:]
    })

    mean_field = pm.fit(n=5000, method='advi', obj_optimizer=pm.adam(learning_rate=0.01))
    trace = mean_field.sample(1000)

estimated_corrs = np.repeat(np.mean(trace['corr'], axis=0), 10)
step_size = np.repeat(np.mean(trace['step_size'], axis=0), 10)

plt.plot(estimated_corrs, color='r')
plt.plot(corrs, color='b')
plt.show()

plt.plot(step_size)
plt.show()