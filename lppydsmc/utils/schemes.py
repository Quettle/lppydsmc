import numpy as np

# ---------------------- Schemes -------------------- #
""" Available schemes for integration of the particles.

In each of the following schemes, *f* is the function that returns the derivated of *arr*.
based on arr = [x, y, vx, vy, vz] and *args*, that are to be provided by the user.
The user should be careful to make *f* as efficient as possible (vectorize it, good use of numpy etc.).

TODO : implement the leap_frog scheme. Issue however with the half time step ?
"""

def euler_explicit(arr, fn, dt, t, fn_args):
    der = fn(arr, t, **fn_args) # der is 5D : [vx, vy, ax, ay, az]
    arr[:, :] += dt*der
    return arr

def rk4(arr, fn, dt, t, fn_args):
    k1 = fn(arr, t, **fn_args) # 5D [vx, vy, ax, ay, az]
    k2 = fn(arr + dt/2 * k1, t+dt/2, **fn_args)
    k3 = fn(arr + dt/2 * k2, t+dt/2, **fn_args)
    k4 = fn(arr + dt * k3, t+dt, **fn_args)

    arr[:, :] += dt/6. * (k1 + 2*k2+ 2*k3 + k4)

    return arr


# problem here : 
# based on https://en.wikipedia.org/wiki/Leapfrog_integration
# we need the acceleration of the previous step too. However we don't know how to compute that
# so what we do is : returning arr and the acceleration so we can save it for the next time
# acc(n-1) should be saved in *args*
# def leap_frog(arr, f, dt, t, args):
#     der = f(arr, t, *args) # acc is 3D [ax, ay, az], acc = a(i), a(i+1)
#     arr[:,:2] += arr[:, 2:4]*dt + 0.5*der[0, 2:4]*dt*dt # pos(i+1) = pos(i) + v(i)*dt + 1/2 a(i) * dt^2
#     arr[:, 2:] += 0.5*(acc[0]+acc[1])*dt # v(i+1) = v(i) + 1/2 (a(i)+a(i+1))*dt
#     return arr, acc

# ----------------- dispatcher ------------------ #

def scheme_dispatcher(string):
    """ Simple dispatcher.

    Args:
        string (str): name of the scheme for the integration of the particles.

    Returns:
        function: the actual scheme to realize the integration.
    """
    if(string == 'rk4'):
        return rk4
    else:
        # default
        return euler_explicit