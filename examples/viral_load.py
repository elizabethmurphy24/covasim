import covasim as cv
import sciris as sc
import matplotlib.pyplot as plt
import numpy as np
import covasim.data as cvdata
import pandas as pd
import covasim.defaults as cvd
import covasim.utils as cvu


n_rows, n_cols = 2, 2
n_regions = n_rows * n_cols
pop_size = 10000
location = 'Zambia'

pars = dict(
    pop_size = pop_size,
    pop_type = 'hybrid',
    n_days = 180,
    location = location,
    pop_infected = 0,
    n_imports = 0,
)

sim = cv.Sim(pars)
sim.initialize()

# assign each person to a region
region_ids = np.repeat(np.arange(n_regions), pop_size // n_regions)
np.random.shuffle(region_ids)
sim.people.region = region_ids

sim.people.infect(inds=np.where(sim.people.region == 0)[0][:20]) # 20 initial infections in Region_0_0

region = 0
inds_region0 = np.where(sim.people.region == region)[0]

vl_store = np.zeros((sim.npts, len(inds_region0)))

for t in range(sim.npts):

    sim.step()

    frac_time = cvd.default_float(sim.pars['viral_dist']['frac_time'])
    load_ratio = cvd.default_float(sim.pars['viral_dist']['load_ratio'])
    high_cap = cvd.default_float(sim.pars['viral_dist']['high_cap'])
    date_inf = sim.people.date_infectious
    date_rec = sim.people.date_recovered
    date_dead = sim.people.date_dead
    viral_load = cvu.compute_viral_load(t, date_inf, date_rec, date_dead, frac_time, load_ratio, high_cap)
# store only individuals from Region 0
    vl_store[t, :] = viral_load[inds_region0]


days = np.arange(sim.npts)

fig, ax = plt.subplots(figsize=(10, 6))
ww_signal_region0 = vl_store.sum(axis=1)
plt.plot(days, ww_signal_region0, color='tab:red', label='Region 0 total viral load')
plt.xlabel('Day')
plt.ylabel('Viral load')
plt.show()