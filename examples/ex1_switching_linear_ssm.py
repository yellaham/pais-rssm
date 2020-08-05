import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
from monte_carlo_tools import pf

## TODO: Finalize tracking example before moving on to analyzing the penguin data
#       - Create SSM objects
#       - Create regime switching model,
#       - Generate synthetic data
#       - Solve the tracking problem by running the bootstrap regime switching filter

# Time length for the generated time-series
time_length = 50

# Different parameters for the candidate models
a = np.array([0.1, 0.1, 0.7, 0.7])
b = np.array([-0.5, 0.5, -0.5, 0.5])

# Initialize the candidate state space models and store into a list
candidate_models = []
for i in range(len(a)):
    # Create a linear state-space model
    candidate_models.append(pf.SSM(lambda x: np.random.normal(a[i]*x+b[i], 1),
                                   lambda x1, x2: sp.norm.logpdf(x1, a[i]*x2+b[i], 1),
                                   lambda x: np.random.normal(x, 1),
                                   lambda y, x: sp.norm.logpdf(y, x, 1)))

# Define the regime dynamics
regime_dynamics_rand = lambda model_idx, num_samp: np.random.choice(np.linspace(start=0, stop=3, num=4), num_samp,
                                                                    replace=True, p=np.ones(4)/4)
regime_dynamics_log_pdf = lambda model_idx: np.log(1/4)

# Create a multiple regime SSM and generate synthetic data
model = pf.MultiRegimeSSM(candidate_models, regime_dynamics_rand, regime_dynamics_log_pdf)
y, x, m_idx = model.generate_data(init_state=[0.25], T=time_length)

# Plot the generated observations
plt.plot(np.arange(1, time_length+1), y)
plt.plot(np.arange(0, time_length+1), x)
plt.show()