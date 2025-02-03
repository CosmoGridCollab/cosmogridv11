import os, sys, logging, collections, numpy as np 
from . import utils_logging
LOGGER = utils_logging.get_logger(__file__)

def drop_cols(a, *fieldnames_to_remove):
    a = a[ [ name for name in a.dtype.names if name not in fieldnames_to_remove ] ]
    return rewrite(a)


def rename_cols(a, src, dst):

    new_names = []
    for c in a.dtype.names:

        ind = [i for i,s in enumerate(src) if s==c]
        if len(ind)>0:
            new_names.append(dst[ind[0]])
        else:
            new_names.append(c)
                    
    a.dtype.names = new_names

    return a

def rewrite(a):

    names = [f for f in a.dtype.fields]
    types = [a.dtype.fields[f][0] for f in names]
    a_new = np.empty(len(a), dtype=dict(names=names, formats=types))
    for f in names:
        a_new[f] = a[f]

    return a_new


def dict_to_rec(d):

    keys = list(d.keys())
    arr = np.array([d[k] for k in keys])
    return arr_to_rec(arr, keys)    

def arr_to_rec(arr, cols, dtype=np.float64):

    arr = np.atleast_2d(arr)
    rec = np.empty(len(arr), dtype=np.dtype([(c,dtype) for c in cols]))
    for i, p in enumerate(rec.dtype.names):
        rec[p] = arr[:,i]
    return rec

def rec_to_arr(rec, cols=None, dtype=np.float64):

    if cols == None:
        cols = rec.dtype.names

    return np.array([rec[c] for c in cols], dtype=dtype).T

def arrstr(arr, fmt=': 2.3f'):

    assert arr.ndim<=2, "arrstr works with arrays up to 2 dims"
    arr = np.atleast_2d(arr)
    s = '['
    templ = '{%s}' % fmt
    for i, row in enumerate(arr):
        for j, val in enumerate(row):
            s += templ.format(val)
            if j <len(row)-1:
                s += ' '
        if i < len(arr)-1:
            s+='\n'
    s+=']'
    return s

def add_cols(rec, names, shapes=None, data=0):

    if type(names) is str:
        names = [names]

    # check if new data should be sliced
    slice_data = (isinstance(data, np.ndarray) and data.ndim == 2) or (isinstance(data, list))

    # create new recarray
    names = [str(name) for name in names]
    new_names = list(set(names)-set(rec.dtype.names))
    extra_dtype = get_dtype(new_names, shapes=shapes)
    newdtype = np.dtype(rec.dtype.descr + extra_dtype.descr)
    newdtype = np.dtype([newdtype.descr[i] for i in np.argsort(newdtype.names)])
    newrec = np.empty(rec.shape, dtype=newdtype)

    # add data to new recarray
    for field in rec.dtype.fields:
        newrec[field] = rec[field]
        
    for ni, name in enumerate(extra_dtype.names):

        if slice_data:
            newrec[name] = data[ni]

        else:
            newrec[name] = data

    return newrec



def get_dtype(columns, main='f8', shapes=None):

    list_name = []
    list_dtype = []

    if shapes is None:
        shapes = [() for _ in columns]

    for col in columns:
        if ':' in col:
            name, dtype = col.split(':')
        else:
            name, dtype = col, main

        list_name.append(str(name))
        list_dtype.append(str(dtype))

    dtype = np.dtype(list(zip(list_name, list_dtype, shapes)))

    return dtype

def describe(arr):

    print(f'shape  ={arr.shape}')
    print(f'memory = {arr.nbytes/1e6}mb')
    print(f'min    ={np.min(arr): 2.4e}')
    print(f'max    ={np.max(arr): 2.4e}')
    print(f'mean   ={np.mean(arr): 2.4e}')
    print(f'median ={np.median(arr): 2.4e}')
    print(f'sum    ={np.sum(arr): 2.4e}')

def zeros_rec(n, columns):

    return np.zeros(n, dtype=get_dtype(columns))


def merge_recs(recs):

    assert len(np.unique([len(r) for r in recs]))==1, "all recs should have the same lengths"

    dtype = []
    for r in recs:
        dtype += r.dtype.descr
    dtype = np.dtype(dtype)
    out = np.empty(len(recs[0]), dtype)

    for r in recs:
        for c in r.dtype.names:
            out[c] = r[c]

    return out


def unicode_to_ascii(rec):

    formats = []
    for col in rec.dtype.names:
        dt = rec[col].dtype
        if dt.kind == 'U':
            nchars = dt.itemsize // 4
            formats.append(f'S{nchars}')
        else:
            formats.append(dt.str)

    dtype_out = np.dtype(dict(names=rec.dtype.names, formats=formats))
    rec_new = np.empty(len(rec), dtype=dtype_out)
    for col in rec.dtype.names:
        rec_new[col] = rec[col]

    return rec_new


def ascii_to_unicode(rec):

    formats = []
    for col in rec.dtype.names:
        dt = rec[col].dtype
        if dt.kind == 'S':
            formats.append('U512')
        else:
            formats.append(dt.str)

    dtype_out = np.dtype(dict(names=rec.dtype.names, formats=formats))
    rec_new = np.empty(len(rec), dtype=dtype_out)
    for col in rec.dtype.names:
        rec_new[col] = rec[col]

    return rec_new


