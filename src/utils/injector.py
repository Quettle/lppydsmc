import numpy as np
# TODO : this functions may also be cythonized.

def inject(in_wall, in_vect, debit, vel_std, radius, dt, remains = 0):
        # Injecting for one particle and one wall 
        # Returns a list of particle trough the wall
        # Hypothesis :
        #   (- uniform distribution for position)
        #   - Exactly as I did before for now
        #   - gaussian distribution for velocity; no drift
        # in_vect shape is (2,). What we are basically doing is initializing velocity for a injection along +x
        # and then rotating the velocity so we have speed along the right vector.
        inject_qty_frac = dt * debit + remains
        inject_qty = int(inject_qty_frac)
        # ic(inject_qty)
        remains = inject_qty_frac-inject_qty
        
        # rotating coefficients 
        k1, k2 = in_vect[0], in_vect[1] # ctheta, stheta

        # initializing velocity in the system : (in_vect, b, in_vect x b), direct system
        u = vel_std * np.sqrt(-2*np.log((1-np.random.random(size = inject_qty))))
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

        return np.concatenate((pos, vel), axis = 1), remains