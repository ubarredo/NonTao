import datetime as dt
import random as rd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# MACHINES ##################################################

def set_machines(quan, lim):
    mac = pd.DataFrame(columns=('zone',
                                'machine',
                                'latitude',
                                'longitude'))
    ind = 0
    for k, v in quan.items():
        for n in range(v):
            mac.loc[ind] = (k,
                            'n' + (len(str(sum(quan.values()))) -
                                   len(str(ind + 1))) * '0' + str(ind + 1),
                            round(rd.uniform(lim[k][0], lim[k][1]), 6),
                            round(rd.uniform(lim[k][2], lim[k][3]), 6))
            ind += 1
    mac['latitude'] = mac['latitude'].map('{:,.6f}'.format)
    mac['longitude'] = mac['longitude'].map('{:,.6f}'.format)
    mac.to_csv('docs/nontao_machines.csv', index=False)
    return mac


quantities = {'1A': 10, '1B': 10, '2A': 10, '2B': 10, '2C': 10}
limits = {'1A': (42.846520, 42.841549, -2.679226, -2.668133),
          '1B': (42.846520, 42.841549, -2.668133, -2.662672),
          '2A': (42.855290, 42.846520, -2.673395, -2.662672),
          '2B': (42.855290, 42.850894, -2.679226, -2.673395),
          '2C': (42.850894, 42.846520, -2.679226, -2.673395)}

machines = set_machines(quantities, limits)


# DISPLAY ######################################################

def plot_display(mac, col):
    x = np.array(mac['longitude']).astype(float)
    y = np.array(mac['latitude']).astype(float)
    plt.scatter(x,
                y,
                edgecolor='black',
                linewidth='2',
                c=[col[i] for i in mac['zone']],
                s=400,
                alpha=.7)
    l1, l2, l3, l4 = np.min(x), np.max(x), np.min(y), np.max(y)
    plt.xlim(l1 - .1 * (l2 - l1), l2 + .1 * (l2 - l1))
    plt.ylim(l3 - .1 * (l4 - l3), l4 + .1 * (l4 - l3))
    plt.title("Machines per zone")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig('plots/nontao_display.png')


colors = {'1A': 'red', '1B': 'blue',
          '2A': 'green', '2B': 'orange', '2C': 'purple'}
plot_display(machines, colors)


# DATA #####################################################


def expand_df(df, spo, field='machine'):
    l = len(df)
    df = pd.DataFrame().append([df] * spo, ignore_index=True)
    df.sort_values(field, inplace=True)
    df['spot'] = np.array(list(range(spo)) * l) + 1
    df['occ'] = np.empty((len(df), 0)).tolist()
    return df


def check_spot(x1, x2, y):
    flat = np.array(y).flatten()
    check1 = not np.array([x1 <= i <= x2 for i in flat]).any()
    check2 = not np.array([i[0] <= x1 <= i[1] for i in y]).any()
    check3 = not np.array([i[0] <= x2 <= i[1] for i in y]).any()
    return np.array([check1, check2, check3]).all()


def sim_data(mac, spo, opt, dur, tra):
    temp = expand_df(mac, spo)
    dat = pd.DataFrame(columns=['zone',
                                'machine',
                                'latitude',
                                'longitude',
                                'spot',
                                'in',
                                'out'])
    fails = 0
    ind = 0
    while fails < tra:
        o = rd.choice(opt)
        delta = dt.timedelta(hours=rd.randint(o[0], o[1] - 1),
                             minutes=rd.randrange(60),
                             seconds=rd.randrange(60))
        start = dt.datetime.min + delta
        end = start + dt.timedelta(minutes=rd.choice(dur))
        a, b = [3600 * hms.hour + 60 * hms.minute + hms.second
                for hms in [start.time(), end.time()]]
        mask = temp['occ'].map(lambda occ: check_spot(a, b, occ))
        subset = temp[mask]
        if not subset.empty:
            fails = 0
            s = rd.choice(subset.index)
            temp.ix[s, 'occ'].append((a, b))
            dat.loc[ind] = (subset.ix[s, 'zone'],
                            subset.ix[s, 'machine'],
                            subset.ix[s, 'latitude'],
                            subset.ix[s, 'longitude'],
                            subset.ix[s, 'spot'],
                            start.time(),
                            end.time())
            ind += 1
        else:
            fails += 1
    dat.sort_values(['zone', 'machine', 'spot', 'in'], inplace=True)
    dat['spot'] = dat['spot'].astype(int)
    dat.to_csv('docs/nontao_data.csv', index=False)


spots = 4
options = [(10, 14), (16, 20)]
durations = [15, 30, 60, 90]
traffic = {'low': 1, 'medium': 10, 'high': 100}
sim_data(machines, spots, options, durations, traffic['medium'])
