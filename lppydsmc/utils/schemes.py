import numpy as np
# ---------------------- Schemes -------------------- #

# In each of the following schemes, *f* is the function that returns the acceleration
# based on arr = [x, y, vx, vy, vz] and *args*, that are to be provided by the user.
# The user should be careful to make *f* as efficient as possible (and vectorize it and use only numpy).

def euler_explicit(arr, f, dt, args):
    acc = f(arr, dt, *args) # acc is 3D [ax, ay, az]
    arr[:, 2:] += dt*acc
    arr[:, :2] += arr[:, 2:4]*dt
    return arr

# problem here : 
# based on https://en.wikipedia.org/wiki/Leapfrog_integration
# we need the acceleration of the previous step too. However we don't know how to compute that
# so what we do is : returning arr and the acceleration so we can save it for the next time
# acc(n-1) should be saved in *args*
def leap_frog(arr, f, dt, args):
    acc = f(arr, dt, *args) # acc is 3D [ax, ay, az], acc = a(i), a(i+1)
    arr[:,:2] += arr[:, 2:4]*dt + 0.5*acc[0, :2]*dt*dt # pos(i+1) = pos(i) + v(i)*dt + 1/2 a(i) * dt^2
    arr[:, 2:] += 0.5*(acc[0]+acc[1])*dt # v(i+1) = v(i) + 1/2 (a(i)+a(i+1))*dt
    return arr, acc
