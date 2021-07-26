import numpy as np

def advect(arr, f, time_step, time, fn_args, scheme):
    # scheme: the scheme employed to update arr
    scheme(arr, f, time_step, time, fn_args)


