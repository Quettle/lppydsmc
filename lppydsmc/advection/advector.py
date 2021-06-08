import numpy as np

def advect(arr, f, dt, args, scheme):
    # scheme: the scheme employed to update arr
    scheme(arr, f, dt, args)