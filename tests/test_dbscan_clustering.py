"""Unit tests for Incremental DBSCAN clustering config / dispatch API."""

import unittest

from speciation.clustering import cluster
from speciation.config import SpeciationConfig


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
