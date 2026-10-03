"""Unit tests for DBSCAN clustering (precomputed embedding distance)."""

import numpy as np
import unittest

from speciation.clustering import cluster, dbscan_precomputed
from speciation.config import SpeciationConfig
from speciation.distance import pairwise_embedding_matrix


def _unit(v):
    a = np.asarray(v, dtype=np.float64)
    return a / np.linalg.norm(a)


class TestDBSCANPrecomputed(unittest.TestCase):
    def test_two_tight_clusters_and_noise(self):
        a = _unit([1.0, 0.0, 0.0])
        b = _unit([0.99, 0.1, 0.0])
        c = _unit([0.0, 1.0, 0.0])
        d = _unit([0.1, 0.99, 0.0])
        noise = _unit([0.0, 0.0, 1.0])
        emb = np.stack([a, b, c, d, noise])
        dist = pairwise_embedding_matrix(emb)
        self.assertEqual(dist.shape, (5, 5))
        np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-8)
        np.testing.assert_allclose(dist, dist.T, atol=1e-8)

        labels = dbscan_precomputed(dist, eps=0.05, min_samples=2)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])
        self.assertEqual(int(labels[4]), -1)

    def test_all_noise_when_eps_tiny(self):
        emb = np.stack([_unit([1, 0]), _unit([0, 1]), _unit([1, 1])])
        dist = pairwise_embedding_matrix(emb)
        labels = dbscan_precomputed(dist, eps=1e-6, min_samples=2)
        self.assertTrue(np.all(labels == -1))

    def test_single_cluster_large_eps(self):
        emb = np.stack([_unit([1, 0]), _unit([0.9, 0.1]), _unit([0.8, 0.2])])
        dist = pairwise_embedding_matrix(emb)
        labels = dbscan_precomputed(dist, eps=1.0, min_samples=2)
        self.assertTrue(np.all(labels == 0))


class TestClusteringAPIConfig(unittest.TestCase):
    def test_config_accepts_dbscan(self):
        cfg = SpeciationConfig(clustering_method="dbscan", dbscan_eps=0.2, dbscan_min_samples=2)
        self.assertEqual(cfg.clustering_method, "dbscan")
        self.assertEqual(cfg.dbscan_eps, 0.2)

    def test_config_rejects_bad_method(self):
        with self.assertRaises(AssertionError):
            SpeciationConfig(clustering_method="kmeans")

    def test_cluster_unknown_raises(self):
        with self.assertRaises(ValueError):
            cluster("nope", config=SpeciationConfig())


if __name__ == "__main__":
    unittest.main()
