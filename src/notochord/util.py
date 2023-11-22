from collections.abc import Mapping
import inspect
import urllib.request

import torch
from tqdm import tqdm

# from https://stackoverflow.com/a/53877507
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
##

def get_function_defaults(fn):
    """get dict of name:default for a function's arguments"""
    s = inspect.signature(fn)
    return {k:v.default for k,v in s.parameters.items()}

def get_class_defaults(cls):
    """get the default argument values of a class constructor"""
    d = get_function_defaults(getattr(cls, '__init__'))
    # ignore `self` argument, insist on default values
    try:
        d.pop('self')
    except KeyError:
        raise ValueError("""
            no `self` argument found in class __init__
        """)
    assert [v is not inspect._empty for v in d.values()], """
            get_class_defaults should be used on constructors with keyword arguments only.
        """
    return d

# mask for canonical order
def gen_perms(a):
    """yield all permutations of the given list"""
    if len(a)==1:
        yield a
    else:
        # for each position
        for i in range(len(a)):
            # for permuations of remaining positions
            for p in gen_perms(a[:i]+a[i+1:]):  
                yield a[i:i+1]+p 

def gen_masks(n, dtype=torch.float):
    """yield the autoregressive mask matrices of all permuations of n items"""
    for perm in gen_perms(list(range(n))):
        m = torch.zeros(n,n,dtype=dtype)
        for idx,i in enumerate(perm):
            for j in perm[:idx]:
                m[j,i] = 1
        yield perm, m


def deep_update(a, b):
    """
    in-place update a with contents of b, recursively for nested Mapping objects.
    """
    for k in b:
        if k in a and isinstance(a[k], Mapping) and isinstance(b[k], Mapping):
            deep_update(a[k], b[k])
        else:
            a[k] = b[k]


def arg_to_set(x):
    """convert None to empty set, iterable to set, or scalar to set with one item"""
    if x is None:
        return set()
    elif not hasattr(x, '__iter__'):
        return {x}
    else:
        return set(x)