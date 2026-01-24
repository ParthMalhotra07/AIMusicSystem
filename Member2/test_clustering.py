"""
Quick Clustering Test
Tests all clustering models on embeddings
"""

import sys
import numpy as np

# Add Member2 to path
sys.path.insert(0, '.')

from clustering import KMeansModel, HierarchicalModel, DBSCANModel
from clustering import ClusteringEvaluator, ClusterAnalyzer

print("=" * 80)
print("🎵 Quick Clustering Test")
print("=" * 80)

# Load embeddings from previous test
try:
    # Try to load real embeddings if available
    embeddings_path = 'output/test_ae_embeddings.npy'
    X = np.load(embeddings_path)
    print(f"\n✓ Loaded real embeddings from: {embeddings_path}")
    print(f"   Shape: {X.shape}")

    # Load song names if available
    try:
        import json
        with open('output/song_mapping.json', 'r') as f:
            mapping = json.load(f)
            song_names = mapping.get('song_names', None)
            print(f"✓ Loaded {len(song_names)} song names")
    except:
        song_names = [f"song_{i:02d}" for i in range(len(X))]
        print(f"⚠ Using generated song names")

except:
    # Generate synthetic data if embeddings not available
    print("\n⚠ No embeddings found, using synthetic data")
    np.random.seed(42)

    # Create 3 distinct clusters
    cluster1 = np.random.randn(20, 14) + np.array([5]*14)
    cluster2 = np.random.randn(15, 14) + np.array([-5]*14)
    cluster3 = np.random.randn(10, 14) + np.array([0]*14)

    X = np.vstack([cluster1, cluster2, cluster3])
    song_names = [f"song_{i:02d}" for i in range(len(X))]

    print(f"✓ Generated synthetic embeddings: {X.shape}")

# Determine optimal number of clusters
n_clusters = min(5, len(X) // 3)  # At least 3 samples per cluster
print(f"\n🎯 Using {n_clusters} clusters")

# Test 1: K-Means
print("\n" + "=" * 80)
print("1️⃣  Testing K-Means Clustering")
print("=" * 80)

kmeans = KMeansModel(n_clusters=n_clusters, n_init=10)
kmeans.fit(X, verbose=True)

evaluator = ClusteringEvaluator()
kmeans_metrics = evaluator.evaluate_all(X, kmeans.labels_, verbose=True)

analyzer = ClusterAnalyzer()
kmeans_summary = analyzer.get_cluster_summary(X, kmeans.labels_, song_names)
analyzer.print_summary(kmeans_summary, verbose=True)

# Test 2: Hierarchical
print("\n" + "=" * 80)
print("2️⃣  Testing Hierarchical Clustering")
print("=" * 80)

hierarchical = HierarchicalModel(n_clusters=n_clusters, linkage='ward')
hierarchical.fit(X, verbose=True)

hierarchical_metrics = evaluator.evaluate_all(X, hierarchical.labels_, verbose=True)

hierarchical_summary = analyzer.get_cluster_summary(X, hierarchical.labels_, song_names)
analyzer.print_summary(hierarchical_summary, verbose=False)

# Test 3: DBSCAN
print("\n" + "=" * 80)
print("3️⃣  Testing DBSCAN Clustering")
print("=" * 80)

# Estimate eps
eps = DBSCANModel.estimate_eps(X, k=5)

dbscan = DBSCANModel(eps=eps, min_samples=3)
dbscan.fit(X, verbose=True)

dbscan_metrics = evaluator.evaluate_all(X, dbscan.labels_, verbose=True)

dbscan_summary = analyzer.get_cluster_summary(X, dbscan.labels_, song_names)
analyzer.print_summary(dbscan_summary, verbose=False)

# Compare all methods
print("\n" + "=" * 80)
print("📊 Comparison Summary")
print("=" * 80)

results = {
    'K-Means': kmeans.labels_,
    'Hierarchical': hierarchical.labels_,
    'DBSCAN': dbscan.labels_
}

comparison = evaluator.compare_results(X, results)
evaluator.print_comparison(comparison)

# Final recommendation
print("\n" + "=" * 80)
print("✅ Test Complete!")
print("=" * 80)

best_method = max(
    ['K-Means', 'Hierarchical', 'DBSCAN'],
    key=lambda m: comparison[m]['silhouette_score']
)

print(f"\n🏆 Best Method: {best_method}")
print(f"   Silhouette Score: {comparison[best_method]['silhouette_score']:.4f}")
print(f"   Number of Clusters: {comparison[best_method]['n_clusters']}")

print("\n" + "=" * 80)
print("Next Steps:")
print("  1. Use embeddings from: python embedding/train.py")
print("  2. Train clustering: python clustering/train.py --embeddings <path>")
print("  3. Use cluster labels for recommendations (Member 3)")
print("=" * 80)
