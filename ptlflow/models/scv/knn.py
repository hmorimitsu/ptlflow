try:
    import faiss
except ImportError:
    raise ImportError(
        'ERROR: faiss not found.'
        ' CSV requires faiss library to run.'
        ' Install with pip install faiss-gpu'
    )
import torch

res = faiss.StandardGpuResources()
res.setDefaultNullStreamAllDevices()


def swig_ptr_from_Tensor(x):
    """ gets a Faiss SWIG pointer from a pytorch tensor (on CPU or GPU) """
    assert x.is_contiguous()

    if x.dtype == torch.float32:
        return faiss.cast_integer_to_float_ptr(x.storage().data_ptr() + x.storage_offset() * 4)

    if x.dtype == torch.int64:
        return faiss.cast_integer_to_int_ptr(x.storage().data_ptr() + x.storage_offset() * 8)

    raise Exception("tensor type not supported: {}".format(x.dtype))


def search_raw_array_pytorch(res, xb, xq, k, D=None, I=None,
                             metric=faiss.METRIC_L2):
    """search xq in xb, without building an index"""
    assert xb.device == xq.device

    nq, d = xq.size()
    if xq.is_contiguous():
        xq_row_major = True
    elif xq.t().is_contiguous():
        xq = xq.t()    # I initially wrote xq:t(), Lua is still haunting me :-)
        xq_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')

    xq_ptr = swig_ptr_from_Tensor(xq)

    nb, d2 = xb.size()
    assert d2 == d
    if xb.is_contiguous():
        xb_row_major = True
    elif xb.t().is_contiguous():
        xb = xb.t()
        xb_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')
    xb_ptr = swig_ptr_from_Tensor(xb)

    if D is None:
        D = torch.empty(nq, k, device=xb.device, dtype=torch.float32)
    else:
        assert D.shape == (nq, k)
        assert D.device == xb.device

    if I is None:
        I = torch.empty(nq, k, device=xb.device, dtype=torch.int64)
    else:
        assert I.shape == (nq, k)
        assert I.device == xb.device

    D_ptr = swig_ptr_from_Tensor(D)
    I_ptr = swig_ptr_from_Tensor(I)

    args = faiss.GpuDistanceParams()
    args.metric = metric
    args.k = k
    args.dims = d
    args.vectors = xb_ptr
    args.vectorsRowMajor = xb_row_major
    args.numVectors = nb
    args.queries = xq_ptr
    args.queriesRowMajor = xq_row_major
    args.numQueries = nq
    args.outDistances = D_ptr
    args.outIndices = I_ptr
    faiss.bfKnn(res, args)

    return D, I


def knn_faiss_raw(fmap1, fmap2, k):

    b, ch, _ = fmap1.shape

    if b == 1:
        fmap1 = fmap1.view(ch, -1).t().contiguous()
        fmap2 = fmap2.view(ch, -1).t().contiguous()

        dist, indx = search_raw_array_pytorch(res, fmap2, fmap1, k, metric=faiss.METRIC_INNER_PRODUCT)

        dist = dist.t().unsqueeze(0).contiguous()
        indx = indx.t().unsqueeze(0).contiguous()
    else:
        fmap1 = fmap1.view(b, ch, -1).permute(0, 2, 1).contiguous()
        fmap2 = fmap2.view(b, ch, -1).permute(0, 2, 1).contiguous()
        dist = []
        indx = []
        for i in range(b):
            dist_i, indx_i = search_raw_array_pytorch(res, fmap2[i], fmap1[i], k, metric=faiss.METRIC_INNER_PRODUCT)
            dist_i = dist_i.t().unsqueeze(0).contiguous()
            indx_i = indx_i.t().unsqueeze(0).contiguous()
            dist.append(dist_i)
            indx.append(indx_i)
        dist = torch.cat(dist, dim=0)
        indx = torch.cat(indx, dim=0)
    return dist, indx