# %%
import statsmodels.api as sm
from utils import SimData, UStats
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def run_simulation(nsim):
    results_list = []
    for _ in tqdm(range(nsim), desc="Processing", unit="iteration"):
        sim = SimData(**kwargs)
        # Fit linear regression model
        x_add_const = sm.add_constant(sim.x)
        model = sm.OLS(sim.y, x_add_const)
        results = model.fit()
        res = results.resid

        kernel_test = UStats().kernel_test(
            res,
            sim.x,
            const=0.1,
            N=kwargs["sample_size"],
            nfeatures=kwargs["n_features"],
        )
        results_list.append(kernel_test[0])
    return results_list


# %%
kwargs = {
    "sample_size": 2000,
    "n_features": 1,
    "eps_distri": "two-way-clustering",  # normal, ar, heteroscedasticity, two-way-clustering
    "x_scale": 5,
    "eps_scale": 1,
    "ar_params": [1, -0.1],
    "p": 1,
    "tau": 1,
    "hypothesis": "alternative",
    "n1": 100,  # n1*n2 must equal sample_size
    "n2": 20,
    "scale_alternative": 0.1,
}
nsim = 2000
ee = SimData(**kwargs).gen_epsilon()
print(ee.mean(), ee.var())

results = run_simulation(nsim)
results = np.array(results)
# Count the proportion of values in results that exceed 1.96 in absolute value
count = np.sum(np.abs(results) > 1.96) / len(results)
print(
    f"{kwargs['eps_distri']}, {kwargs['hypothesis']}, {kwargs['ar_params']}, {kwargs['n1']}, {kwargs['scale_alternative']} \n"
)
print(f"The propotion of rejection is {count}")


# %%
plt.figure()
plt.hist(results)
plt.show()

# %%
sim = SimData(**kwargs)
plt.plot(sim.epsilon)
# Plot ACF of res for first 6 lags
sm.graphics.tsa.plot_acf(sim.epsilon, lags=6)
# %%
