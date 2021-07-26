import numpy as np


# ------------------------------- DSMC reactions ---------------------------- #

def react(idx_colliding_particles, arrays, masses, types_dict, reactions, p = None, monitoring = False):
    """ Take the indexes of the walls-colliding particles of ONLY ONE SPECIES and process recombination-probability on them. By default, it will use a uniform law and a reaction is bound to happen.
    It is a very basic and approximative funtions.

    Args:
        idx_colliding_particles (np.ndarray): 1D-ndarray containing the indexes of the colliding particles in the corresponding array of arrays.
        arrays (list): list of arrays containing each the particles of the associated species, the size of each array is : number of particles x 5. 
        masses (np.ndarray): Array containing the masses of the species in the simulation. The order is described by *types_dict*.
        types_dict (dict): Dictionnary associating to species identifiers (e.g. 'I', 'I2', 'I+', 'I-', 'e-' etc.) their corresponding integer identifier
                           in *idx_colliding_particles*, *arrays* and *masses*.
        reactions (dict): dict giving the possible reactions for the current species. A reaction is giving by : 'i' :  'R : P1 + P2 + ...' where 'i' is 
                          an integer starting at 1 (0 is for 'no reaction').
        p (function, optional): function to compute the integer 'i' for each colliding particles. 0 if no reaction. Defaults to None. 
        monitoring (bool, optional): If monitoring is True, more computations is done and return value is different. Defaults to False.

    Returns:
        [np.ndarray, list, np.ndarray (optional)]: returns the index of the reacting particles (and should then be deleted), 
                                                   also returns the particle to add (array [x,y,vx,vy,vz]), and if monitoring is True, 
                                                   returns the integer array of happening reactions for every given particles.
    TODO : 
        - could use some optimization (initially this function was thought for recombination AND reaction between species consequent to DSMC collisions)
        - Add an example
    """
    nb_reactions = reactions['#']
    reactants = reactions['reactants']

    nb_parts = idx_colliding_particles.shape[0]

    if(p is None):
        p_ = 1/nb_reactions
        def p(size):
            proba = np.random.uniform(low=0.0, high=1.0, size = size)
            return (proba*p_+1).astype(int)

    happening_reactions = p(nb_parts)
    
    masses_reactants = [masses[types_dict[reactant]] for reactant in reactants]
    # list of arrays (pointers) towards the arrays containing all the particles of the given reacting particles 
    arrays_reactants = [arrays[types_dict[reactant]] for reactant in reactants] 

    particles_to_add = {}
    reacting_particles = []
    for i, r in enumerate(np.flip(happening_reactions)):
        if(r == 0): # if r==0, no reaction !
            continue

        idxes = idx_colliding_particles[i]

        reacting_particles.append(idxes)
        products = reactions[str(r)]

        masses_products = np.array([masses[types_dict[product]] for product in products])
        reduced_masses_products = 1/np.sum(masses_products) * masses_products 
        part = 1/np.sum(masses_reactants) * sum([masses_reactants[k]*arrays_reactants[k][idxes[k]] for k in range(len(reactants))])

        # NOTE : in a first approximation, we compute the linear momentum for the reactants and spread it over the products, 
        # taking into account the mass of each one. We could instead try something with an spreading angle etc. 
        # Here every product is leaving the same way.
        
        for k, product in enumerate(products):
            if(product in particles_to_add):
                particles_to_add[product].append(part*reduced_masses_products[k])
            else:
                particles_to_add[product] = [part*reduced_masses_products[k]]
    if(monitoring):
        return np.array(reacting_particles), particles_to_add, happening_reactions
    return np.array(reacting_particles), particles_to_add
        
# -------------------------------- Parsing functions to read the reactions ---------------------------------- #

# TODO : move it to the right place - this is more 'utils' or linked to 'config files reading' or something, right ?
def parse_file(file):
    with open(file, mode = 'r') as f:
        lines = f.readlines() # returns all lines as a list of strings
        return parse(lines)

def parse(lines):
    reactions = {}
    for line in lines :
        if(line[0]!= '#'):
            # then we suppose it is a reactant
            reacts, prods = line.split(':')
            reacts = reacts.strip()
            reactants, products = reacts.split(), prods.split() # split by white space
            # for reactant in reactants :
            #    # trying to get the st
            if(reactions.__contains__(reacts)):
                idx_reaction = reactions[reacts]['#'] + 1
                reactions[reacts.strip()][str(idx_reaction)] = parse_(products)
                reactions[reacts.strip()]['#'] = idx_reaction
            else:
                reactions[reacts.strip()] = {
                    '#' : 1,
                    'reactants' : parse_(reactants),
                    '1' : parse_(products)
                }
    return reactions

def parse_(item):
    # parsing item, which can be reactants or products
    # it is a list like ['I','I', '+', 'I-'] for example
    # we just need to delete the +
    item_ = []
    for i in item:
        if(i=='+' or i=='-'):
            continue
        item_.append(i)
    return item_
            

