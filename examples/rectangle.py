import covasim as cv
import sciris as sc
import matplotlib.pyplot as plt
import numpy as np
import covasim.data as cvdata
from ete3 import Tree, TreeStyle, NodeStyle, faces, AttrFace
import pandas as pd
from scipy.integrate import odeint
import cvxpy as cp


n_rows, n_cols = 2, 2
n_regions = n_rows * n_cols
pop_size = 100000
location = 'Zambia'

pars = dict(
    pop_size = pop_size,
    pop_type = 'hybrid',
    n_days = 180,
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

# PLOT heatmap
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


# PLOT shedding load density
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




# individual viral load shedding
daily_shedding = np.zeros((pars['n_days']+1, sim.n))

# record rel_trans every day
def track_shedding(sim):
    t = sim.t
    if t < daily_shedding.shape[0]:
        daily_shedding[t, :] = sim.people.rel_trans
    return

# new simulation
sim = cv.Sim(pars, interventions=track_shedding)
sim.initialize()

# assign regions again
sim.people.region = region_ids

# reseed infections
initial_inds = np.where(sim.people.region == 0)[0][:20]
sim.people.infect(initial_inds)

# run sim
sim.run()

# extract region 0
region = 0
region_inds = np.where(sim.people.region == region)[0]

ww_signal_reltrans = daily_shedding[:, region_inds].sum(axis=1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(dates, ww_signal_reltrans, label='rel_trans', color='tab:blue')
ax.legend()
plt.show()



# EFFECTIVE REPRODUCTION NUMBER
import epyestim
import epyestim.covid19 as covid19

dates_ts = pd.to_datetime(sim.results['date'])

# covasim cases R_e
# cases = pd.Series(new_cases, index=dates_ts)
cases = pd.Series(sim.results['new_infections'].values, index=dates_ts)
sim_time_varying_r = covid19.r_covid(cases)

# covasim default output R_e
sim_re = sim.results['r_eff'].values
sim_re_series = pd.Series(sim_re, index=dates_ts)

# PLOT
fig, ax = plt.subplots(1,1, figsize=(11, 5))
ax.plot(sim_time_varying_r.index,sim_time_varying_r.loc[:,'Q0.5'],color='#ff7f0e',label='Covasim Cases')
ax.fill_between(sim_time_varying_r.index, sim_time_varying_r['Q0.025'], sim_time_varying_r['Q0.975'], color='#ff7f0e', alpha=0.2)
ax.plot(sim_re_series.index, sim_re_series.values, color="#4e0eff", label='Covasim R_e')
ax.set_ylabel('$R_e$')
ax.set_ylim([-0.2, 4])
ax.set_yticks([0, 1, 2, 3])
ax.axhline(y=1, color='red',linewidth=.7)
plt.legend(loc='lower right')
plt.show()


print(sim.results['r_eff'].values[:20])
# [3.21004993 3.50187266 3.57300642 3.5576068  3.54291047 3.51565718
# 3.45472311 3.46832504 3.55711422 3.51827703 3.32933567 3.32383208
# 3.49370891 3.30173734 2.72557355 2.37236892 2.36162143 2.21253799
# 1.81167899 1.57379071]

print(sim_time_varying_r['Q0.5'].head(20))