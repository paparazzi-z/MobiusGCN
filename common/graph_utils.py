from __future__ import absolute_import

import torch
import numpy as np
import scipy.sparse as sp


def fourier(L, algo='eigh', k=1):
    """Return the Fourier basis, i.e. the EVD of the Laplacian."""

    def sort(lamb, U):
        idx = lamb.argsort()
        return lamb[idx], U[:, idx]

    if algo == 'eig':
        lamb, U = np.linalg.eig(L.toarray())
        lamb, U = sort(lamb, U)
    elif algo == 'eigh':
        lamb, U = np.linalg.eigh(L.toarray())
    elif algo == 'eigs':
        lamb, U = sp.linalg.eigs(L, k=k, which='SM')
        lamb, U = sort(lamb, U)
    elif algo == 'eigsh':
        lamb, U = sp.linalg.eigsh(L, k=k, which='SM')
    elif algo == 'SVD':
        a, lamb, U = np.linalg.svd(L.toarray())

    return lamb, U


def laplacian(num_pts, edges, normalized=True):

    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    
    # Degree of vertexs
    d = adj_mx.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = sp.diags(d.A.squeeze(), 0)
        L = D - adj_mx
    else:
        d += np.spacing(np.array(0, adj_mx.dtype))
        d = 1 / np.sqrt(d)
        D = sp.diags(d.A.squeeze(), 0)
        I = sp.identity(d.size, dtype=adj_mx.dtype)
        L = I - D * adj_mx * D

    assert type(L) is sp.csr.csr_matrix
    
    eigenVal, eigenVec = fourier(L)
    eigenVal = torch.tensor(eigenVal, dtype=torch.float)
    eigenVec = torch.tensor(eigenVec, dtype=torch.float)
    
    return eigenVal, eigenVec

def laplacian_from_skeleton(skeleton):
    num_joints = skeleton.num_joints()
    edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, num_joints)), skeleton.parents())))
    return laplacian(num_joints, edges, normalized = True)

# def lmax(L, normalized=True):
#     """Upper-bound on the spectrum."""
#     if normalized:
#         return 2
#     else:
#         return sp.linalg.eigsh(
#                 L, k=1, which='LM', return_eigenvectors=False)[0]

def testlaplacian(num_pts, edges, normalized=True):

    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    
    # Degree of vertexs
    d = adj_mx.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = sp.diags(d.A.squeeze(), 0)
        L = D - adj_mx
    else:
        d += np.spacing(np.array(0, adj_mx.dtype))
        d = 1 / np.sqrt(d)
        D = sp.diags(d.A.squeeze(), 0)
        I = sp.identity(d.size, dtype=adj_mx.dtype)
        L = I - D * adj_mx * D

    print(L)
    assert type(L) is sp.csr.csr_matrix
    
    a, b, c = np.linalg.svd(L.toarray())
    a = torch.tensor(a, dtype=torch.float)
    b = torch.tensor(b, dtype=torch.float)
    c = torch.tensor(c, dtype=torch.float)
    
    return a, b, c

def test_laplacian_from_skeleton(skeleton):
    num_joints = skeleton.num_joints()
    edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, num_joints)), skeleton.parents())))
    return testlaplacian(num_joints, edges, normalized = True)