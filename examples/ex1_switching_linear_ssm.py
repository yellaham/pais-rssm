import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
from monte_carlo_tools import pf

# Time length for the generated time-series
time_length = 50

# Different parameters for the candidate models
a = np.array([0.1, 0.1, 0.1, 0.1])
b = np.array([-10, -5, 5, 10])

# Initialize the candidate state space models and store into a list
candidate_models = []
for i in range(len(a)):
    # Create a linear state-space model
    candidate_models.append(pf.SSM(lambda x: np.random.normal(a[i]*x+b[i], 1),
                                   lambda x1, x2: sp.norm.logpdf(x1, a[i]*x2+b[i], 1),
                                   lambda x: np.random.normal(x, 0.1),
                                   lambda y, x: sp.norm.logpdf(y, x, 0.1)))

# Define the regime dynamics
regime_dynamics_rand = lambda model_idx, num_samp: np.random.choice(np.linspace(start=0, stop=3, num=4), num_samp,
                                                                    replace=True, p=np.ones(4)/4)
regime_dynamics_log_pdf = lambda model_idx: np.log(1/4)

# Create a multiple regime SSM and generate synthetic data
model = pf.MultiRegimeSSM(candidate_models, regime_dynamics_rand, regime_dynamics_log_pdf)
y, x, m_idx = model.generate_data(init_state=[0.25], T=time_length)

# Run the Bootstrap regime switching filter to estimate the latent state trajectory
num_particles = 100
initial_particles = np.random.uniform(-1, 1, size=(1, num_particles))
output = pf.brspf(y, model, initial_particles)
print(m_idx)
print(sp.mode(output.model_indexes.T)[0])

# Plot the tracking of the latent states
plt.plot(np.arange(0, time_length+1), x)
plt.plot(np.arange(1, time_length+1), np.mean(output.particles, axis=2))
plt.show()