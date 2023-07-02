import numpy as np

from scipy.sparse import coo_array, csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy


def single_linkage(points):
    # Run Delaunay triangulation to obtain a suitable input for Euclidean MST.
    tri = Delaunay(points)
    u, v = np.transpose(
        sorted(
            {
                (i, j)
                for sim in tri.simplices
                for ind, i in enumerate(sim)
                for j in sim[ind + 1 :]
            }
        )
    )
    # Obtain relevant Euclidean distances for Euclidean MST.
    data = ((points[u] - points[v]) ** 2).sum(axis=1) ** 0.5

    # Run Euclidean MST.
    X = coo_array((data, (u, v)), shape=(len(points), len(points)))
    Tcsr = coo_array(minimum_spanning_tree(csr_matrix(X)))

    # Run Union-Find to obtain single-linkage clustering.
    Z = []
    parent = list(range(len(points)))
    subtree_size = [1] * len(points)

    def representative(i):
        while parent[i] != i:
            parent[i] = i = parent[parent[i]]
        return i

    def union(i, j):
        i = representative(i)
        j = representative(j)
        parent[i] = j
        subtree_size[j] += subtree_size[i]

    for i in np.argsort(Tcsr.data):
        u = representative(Tcsr.row[i])
        v = representative(Tcsr.col[i])
        dist = Tcsr.data[i]
        n = subtree_size[u] + subtree_size[v]
        parent[u] = parent[v] = len(parent)
        parent.append(len(parent))
        subtree_size.append(n)
        u, v = sorted((u, v))
        Z.append((u, v, dist, n))

    return np.array(Z)


def main() -> None:
    seed = 42
    np.random.seed(seed)
    pts = np.random.random((20, 3))
    print(f"Random points (seed={seed}):")
    print(pts)
    a = single_linkage(pts)
    print("\nOutput of single_linkage():")
    print(a)
    b = scipy.cluster.hierarchy.linkage(pts)
    print("\nOutput of scipy.cluster.hierarchy.linkage():")
    print(b)
    assert np.array_equal(a, b)

    np.random.seed(seed)
    pts = np.random.random((64000, 2))
    print(single_linkage(pts)[-1])


if __name__ == "__main__":
    main()
