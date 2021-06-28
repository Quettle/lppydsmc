import numpy as np

def advect(arr, f, dt, t, args, scheme):
    # scheme: the scheme employed to update arr
    scheme(arr, f, dt, t, args)