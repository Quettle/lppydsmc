import numpy as np

def get_quantity(debits, remains, dt):
        """ Compute the quantity of particles to inject and also returns the decimal part (useful to add to the next time step).

        Args:
            debits (np.ndarray): 1D-array of size (number of species) with the debits of particles for each species.
            remains (np.ndarray): 1D-array of floats, of size (number of species), with the remained for the previous time step.
            dt (float): time step

        Returns:
            np.ndarray, np.ndarray: remains (as float in [0,1)) and quantities (as integers) to inject, each of size (number of species). 
        """
        remains, qties = np.modf(debits*dt+remains) # returns remains, quantity_to_inject
        return remains, qties.astype(int)
        
def maxwellian(in_wall, in_vect, vel_std, inject_qty, dt, drift = 0):
        """ Compute velocities for a maxwellian injection at a random boundary represented by its inward-normal vector.
        
        Args:
            in_wall (np.ndarray): 1D-array of size (4) represented by its two extremities e.g. [x1,y1,x2,y2]
            in_vect (np.ndarray): 1D-array of size (2) e.g. +/- [y2-y1, x1-x2] (depending on the direction of injection - meaning the inward direction)
            vel_std (float): velocity standard deviation for each gaussian distribution for the other two directions (other than the injection one). 
            Usually computed using lppydsmc.utils.physics.gaussian.
            inject_qty (int): quantity to be injected (computed using *quantity(debits, remains, dt)*)
            dt (float): time step
            drift (float, optional): Drift to be added in the injection direction if necessary. Defaults to 0.

        Returns:
            np.ndarray: 2D-array of shape (inject_qty x 3), where one velocity is (vx,vy,vz).
        """

        # What we are basically doing is initializing velocity for a injection along +x
        # and then rotating the velocity so we have speed along the right vector.

        # rotating coefficients 
        k1, k2 = in_vect[0], in_vect[1] # ctheta, stheta

        # initializing velocity in the system : (in_vect, b, in_vect x b), direct system
        u = vel_std * np.sqrt(-2*np.log((1-np.random.random(size = inject_qty)))) + drift
        v = np.random.normal(loc = 0, scale = vel_std, size = inject_qty)
        w = np.random.normal(loc = 0, scale = vel_std, size = inject_qty) # = vz

        dpu = u*np.random.random(size = inject_qty)*dt # delta position in the new base

        # velocity in the right base
        vx = u*k1-v*k2  # i.e. : vx = vx*ctheta + vy*stheta
        vy = v*k1+u*k2  # i.e. : vy = vy*ctheta - vx*stheta
        dpx = dpu*k1
        dpy = dpu*k2

        vel = np.stack((vx, vy, w), axis = 1)

        # position
        pos = in_wall[:2]+np.stack((dpx, dpy), axis = 1) + np.random.random(size = (inject_qty,1))*(in_wall[2:]-in_wall[:2]) # radius*in_vect+

        return np.concatenate((pos, vel), axis = 1)

# TODO : may be make it available in the config files too.
def dirac(in_wall, in_vect, velocity, dt, inject_qty):
        # rotating coefficients 
        k1, k2 = in_vect[0], in_vect[1] # ctheta, stheta

        # initializing velocity in the system : (in_vect, b, in_vect x b), direct system
        u = velocity * np.ones(size = inject_qty)
        dpu = u*np.random.random(size = inject_qty)*dt # delta position in the new base
        
        # velocity in the right base
        vx = u*k1  # i.e. : vx = vx*ctheta + vy*stheta
        vy = u*k2  # i.e. : vy = vy*ctheta - vx*stheta
        dpx = dpu*k1
        dpy = dpu*k2
        vel = np.stack((vx, vy, np.zeros(size = inject_qty)), axis = 1)

        # position
        pos = in_wall[:2]+np.stack((dpx, dpy), axis = 1) + np.random.random(size = (inject_qty,1))*(in_wall[2:]-in_wall[:2]) # radius*in_vect+

        return np.concatenate((pos, vel), axis = 1)