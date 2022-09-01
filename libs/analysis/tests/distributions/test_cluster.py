import numpy as np
import pytest

from bbhnet.analysis.distributions.cluster import ClusterDistribution


@pytest.fixture(params=[2, 5, 10, 11, 29])
def t_clust(request):
    return request.param


def test_cluster_distribution(t_clust):

    distribution = ClusterDistribution("test", ["H", "L", "V"], t_clust)

    t = np.arange(100).astype("float32")
    y = np.random.normal(size=len(t))

    shifts = [0, 1, 1]

    # insert one maximum value every t_clust / 2 seconds
    y[:: int(t_clust / 2)] = 100

    distribution.update(y, t, shifts)
    assert distribution.Tb == 100

    # ensure all event times are at least t_clust / 2 apart
    assert (np.diff(distribution.event_times) <= (t_clust / 2)).all()

    assert (distribution.events == 100).all()
    assert distribution.nb(99) == len(distribution.events)
    print((distribution.shifts))
    assert len(distribution.shifts) == len(distribution.events)
