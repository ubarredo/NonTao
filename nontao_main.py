import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


def string_time(hms):
    return dt.datetime.strptime(hms, '%H:%M:%S').time()


def select_data(dat, t):
    dat[['in', 'out']] = dat[['in', 'out']].applymap(string_time)
    return dat[(dat['in'] < t.time()) & (t.time() < dat['out'])]


def make_panel(mac, dat, t):
    spo = dat['spot'].max()
    dat = select_data(dat, t)
    l = len(mac)
    pan = pd.DataFrame().append([mac] * spo, ignore_index=True)
    pan.sort_values('machine', inplace=True)
    pan.reset_index(inplace=True, drop=True)
    pan['spot'] = np.array(list(range(spo)) * l) + 1
    pan['time'] = t.time()
    pan['wait'] = 0
    pan['free'] = 1
    for i in dat.index:
        ind = pan[(pan['machine'] == dat['machine'][i]) &
                  (pan['spot'] == dat['spot'][i])].index
        pan.ix[ind, 'free'] = 0
        x = dt.datetime.combine(dt.date.min, dat['out'][i])
        pan.ix[ind, 'wait'] = (x - t).total_seconds()
    pan.to_csv('docs/nontao_panel.csv', index=False)
    return pan


def calc_results(pan):
    res = pan.groupby(['zone',
                       'machine',
                       'latitude',
                       'longitude',
                       'time']).mean()
    res.drop('spot', axis=1, inplace=True)
    lat = np.array(res.index.get_level_values('latitude')).astype(float)
    long = np.array(res.index.get_level_values('longitude')).astype(float)
    c = np.column_stack((long, lat))
    w = np.array(res['wait'])
    f = np.array(res['free'])
    res.to_csv('docs/nontao_results.csv', index=True)
    return c, w, f


def knn_values(p, points, values, k):
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = np.sqrt(np.sum(np.power(p - points[i], 2)))
    ind = distances.argsort()
    return np.mean(values[ind[:k]])


def make_grid(points, values, k, h=50):
    xs = np.linspace(min(points[:, 0]), max(points[:, 0]), h)
    ys = np.linspace(min(points[:, 1]), max(points[:, 1]), h)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.zeros(xx.shape, dtype=float)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = np.array([x, y])
            grid[j, i] = knn_values(p, points, values, k)
    return xx, yy, grid


def plot_grid(points, values, k, colormap, title, filename):
    fig = plt.figure()
    xx, yy, grid = make_grid(points, values, k)
    plt.pcolormesh(xx, yy, grid, cmap=colormap, alpha=.5)
    plt.scatter(points[:, 0], points[:, 1],
                c=values, cmap=colormap,
                s=400, edgecolor='black', linewidth='1', alpha=.8)
    plt.xlim(np.min(xx), np.max(xx))
    plt.ylim(np.min(yy), np.max(yy))
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig('plots/nontao_%s.png' % filename)
    return fig


machines = pd.read_csv('docs/nontao_machines.csv', dtype=object)
data = pd.read_csv('docs/nontao_data.csv')
now = dt.datetime(1, 1, 1,
                  dt.datetime.now().hour,
                  dt.datetime.now().minute,
                  dt.datetime.now().second)
instant = dt.datetime.combine(dt.date.min, dt.time(10, 30))
# uncomment to use local time
# instant = now
panel = make_panel(machines, data, instant)
coordinates, wait_time, free_prob = calc_results(panel)
plot_grid(points=coordinates,
          values=wait_time,
          k=3,
          colormap=ListedColormap(['darkgreen',
                                   'green',
                                   'orange',
                                   'red',
                                   'darkred']),
          title="Mean wait time (s) at " + str(instant.time()),
          filename='wait_time')
plot_grid(points=coordinates,
          values=free_prob,
          k=3,
          colormap='Blues',
          title="Free probability at " + str(instant.time()),
          filename='free_prob')
