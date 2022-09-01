import numpy as np

from bbhnet.analysis.distributions.cluster import ClusterDistribution


def test_cluster_distribution():

    t_clust = 10
    distribution = ClusterDistribution("test", ["H", "L"], t_clust)

    t = np.arange(100).astype("float32")
    y = np.random.normal(size=len(t))

    shifts = [0, 1]

    # insert one maximum value every t_clust / 2 seconds
    # expect these to be the only remaining clustered events
    y[:: int(t_clust / 2)] = 100

    distribution.update(y, t, shifts)
    assert distribution.Tb == 100

    # ensure all event times are at least t_clust / 2 apart
    assert (np.diff(distribution.event_times) >= (t_clust / 2)).all()

    assert (distribution.events == 100).all()
    assert distribution.nb(99) == len(distribution.events)
    assert len(distribution.shifts) == len(distribution.events)
