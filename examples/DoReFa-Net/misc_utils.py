import os


def contains_lmdb(path):
    items = os.listdir(path)
    conds = [i.endswith('lmdb') for i in items]
    return True if True in conds else False
