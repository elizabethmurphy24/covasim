import covasim as cv
import sciris as sc
import matplotlib.pyplot as plt
import numpy as np
import covasim.data as cvdata
from ete3 import Tree, TreeStyle, NodeStyle, faces, AttrFace
import pandas as pd
from scipy.integrate import odeint
import cvxpy as cp

# Assign each person to a region
# Track new infections by region each day
# Plot the results as heatmap

n_rows, n_cols = 2, 2
n_regions = n_rows * n_cols
pop_size = 20000
location = 'Zambia'
n_days = 180

pars = dict(
    pop_size = pop_size,
    pop_type = 'hybrid',
    n_days = n_days,
    location = location,
    pop_infected = 0,   # no automatic initial infections
    n_imports = 0,      # no imported infections
)

# checking if age & household size distributions exists for location='zambia'
try:
    age_data = cvdata.get_age_distribution(location)
    print(f"Age distribution for {location}:")
    print(age_data)
except ValueError as e:
    print(f"No age data for {location}, using default. ({e})")

try: 
    household_size = cvdata.get_household_size(location) 
    print(f"Household size distribution for {location}:") 
    print(household_size) 
except ValueError as e: 
    print(f"No household data for {location}, using default. ({e})")



sim = cv.Sim(pars)
sim.initialize()


# assign each person to a region
region_ids = np.repeat(np.arange(n_regions), pop_size // n_regions)
np.random.shuffle(region_ids)
sim.people.region = region_ids

sim.people.infect(inds=np.where(sim.people.region == 0)[0][:20]) # 20 initial infections in Region_0_0

# check initial infection and what region
seeded_inds = np.where(sim.people.exposed)[0]
print("Number of initially seeded infections:", len(seeded_inds))
print("Regions of initially seeded infections:", np.unique(sim.people.region[seeded_inds]))

# run sim
sim.run()


# compute new infections per region per day
new_cases = np.zeros((len(sim.results['new_infections']), n_regions))
for t in range(len(sim.results['new_infections'])):
    newly_infected = np.where(sim.people.date_infectious == t)[0]
    for r in range(n_regions):
        new_cases[t, r] = np.sum(sim.people.region[newly_infected] == r)



dates = sim.results['date']

# plot heatmap
plt.figure(figsize=(10, 4))
plt.imshow(new_cases.T, aspect='auto', cmap='Reds', origin='lower')
plt.colorbar(label='New infections per day')
plt.ylabel('Region')
plt.yticks(range(n_regions), [f"Region_{i}_{j}" for i in range(n_rows) for j in range(n_cols)])

tick_idx = np.linspace(0, len(dates)-1, 6, dtype=int)
plt.xticks(tick_idx, [dates[i].strftime('%b %d') for i in tick_idx], rotation=45)
plt.tight_layout()
plt.show()



# JUST region 0
region = 0
new_cases = np.zeros(len(sim.results['new_infections']))
for t in range(len(sim.results['new_infections'])):
    newly_infected = np.where(sim.people.date_infectious == t)[0]
    new_cases[t] = np.sum(sim.people.region[newly_infected] == region)


# wastewater signal
def eclipse_model(y, t, b, k, delta, p, mu, c):
    T, I1, I2, Vi, Vni = y
    dydt = [-b*Vi*T, b*Vi*T - k*I1, k*I1 - delta*I2, p*mu*I2 - c*Vi, p*(1.-mu)*I2 - c*Vni]
    return dydt

b = 5e-5
c = 10
k = 6
mu = 1e-4
p = 1e5
delta = 0.5
t = np.linspace(0, 40, 41)
y0 = [1.33e5, 0, 1/30, 0, 0]

# solve ode
sol = odeint(eclipse_model, y0, t, args=(b, k, delta, p, mu, c))
c = sol[:,3:].sum(axis=1)
c = c/c.sum()


# shedding load density plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t, c, linewidth=2)
ax.set_xlabel("Days since infection")
ax.set_ylabel("Shedding load density")

ww_signal = np.convolve(new_cases, c[::-1], mode='same')

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(dates, new_cases, label='New infections from covasim', color='tab:red')
ax.plot(dates, ww_signal, label='Simulated wastewater signal', color='tab:blue')
ax.legend()
plt.show()




# work on

# indiviual level shedding
n = len(sim.results['new_infections'])
infected_inds = np.where(sim.people.date_infectious >= 0)[0]
inds_in_region0 = infected_inds[sim.people.region[infected_inds] == region]
n_ind = len(inds_in_region0)

# individual shedding
ind_shed = np.zeros((n_ind, n))

for idx_pos, pid in enumerate(inds_in_region0):
    start_day = int(sim.people.date_infectious[pid])
    for tau in range(len(c)):
        t = start_day + tau
        if 0 <= t < n:
            ind_shed[idx_pos, t] += c[tau]

# aggregate signals
agg_ind_signal = ind_shed.sum(axis=0)

# plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dates, agg_ind_signal, label='Aggregated individual shedding', color='tab:red')
ax.plot(dates, ww_signal, label='Simulated wastewater signal', color='tab:blue')
ax.legend()
plt.show()