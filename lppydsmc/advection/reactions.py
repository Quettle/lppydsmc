import numpy as np


# ------------------------------- DSMC reactions ---------------------------- #

def react(idx_reacting_particles, arrays, masses, types_dict, reactions, p = None): # INPLACE
    # reactions is a dict listing all available reactions
    # each of them having probability p
    # else we suppose uniform distribution
    # p returns the idx of the reactions that happen in reactions or 0 if there is no reaction
    nb_reactions = reactions['#']
    reactants = reactions['reactants']

    nb_parts = idx_reacting_particles.shape[0]

    if(p is None):
        p_ = 1/nb_reactions
        def p(size):
            proba = np.random.uniform(low=0.0, high=1.0, size = size)
            return (proba*p_+1).astype(int)

    happening_reactions = p(nb_parts)

    masses_reactants = [masses[types_dict[reactant]] for reactant in reactants]
    arrays_reactants = [arrays[types_dict[reactant]] for reactant in reactants] # list of arrays (pointers) towards the arrays containing all the particles of the given reacting particles 

    particles_to_add = {}
    reacting_particles = []
    for i,r in enumerate(np.flip(happening_reactions)):
        if(r == 0):
            continue
        idxes = idx_reacting_particles[i]

        reacting_particles.append(idxes)
        products = reactions[r]

        masses_products = np.array([masses[types_dict[product]] for product in products])
        reduced_masses_products = 1/np.sum(masses_products) * masses_products 
        part = 1/np.sum(masses_reactants) * sum([masses_reactants[k]*arrays_reactants[k][idxes[k]] for k in range(len(reactants))])
        # NOTE : in a first approximation, we compute the linear momentum for the reactants and spread it over the products, takin into account the mass of each one
        # we could instead try something with an spreading angle etc. here every product is leaving the same way.
        for k, product in enumerate(products):
            if(product in particles_to_add):
                particles_to_add[product].append(part*reduced_masses_products[k]) # this is a list
            else:
                particles_to_add[product] = [part*reduced_masses_products[k]]

            # arrays[types_dict[product]].add(linear_momentum_reactants*reduced_masses_products[k]) # TODO : this does not work as we need a container here... I hope to not use a container as it is a special thing...
            # probably going to try to return it instead.
        # NOTE
        # deleting in the previous array
        # only issue is : when we have a several reactants, their indexes in their respective array are not sorted for all (we can sort only on one array)
        # thus posing problems with future collisions ... as the wrong particles will be selected.
        # no other way of doing it I think, except may be deleting only afterwards ?
        # yes, that is what we are going to do.
        # ONCE every particle has been added in the other arrays, we delete all reactants
    return np.array(reacting_particles), particles_to_add

# ----------------------------------- OUSDOAU ------------------------- #
def basic(arr, count, law): 
    """ Returns an array containing the indexes of the particles that reacted with the walls according to the law *law*.

    Args:
        arr (ndarray, float): the array of particles, shape : number of particles x 5 (x,y,vx,vy,vz)
        count (ndarray, int): array of integers containing the number of times the associated particle in *arr* collided
        law (function): returns a probability of reactions ...
    """

    # NOTE : we can sample only for the particles that collided - which would give less computations
    # however it requires a loop (or a copy)
    # or we can draw for the whole array of particles which is crearly suboptimal
    idx_reactions = []
    mean_proba = 0
    s = 0
    for k, c in enumerate(count):
        if(c>0):
            s+= 1
            proba_reaction = law(arr[k], c) # TODO we probably have the wall to consider in the future
            mean_proba += proba_reaction
            rdm_uniform_draw = np.random.random() # we can maybe draw them all out (as we are anyway counting the number of colliding particles outside the function)
            if(proba_reaction > rdm_uniform_draw):
                idx_reactions.append(k)

    return np.array(idx_reactions, dtype = int), mean_proba/s if s!=0 else 0

def angle_dependance(arr, count, alpha, law): 
    """ Returns an array containing the indexes of the particles that reacted with the walls according to the law *law*.

    Args:
        arr (ndarray, float): the array of particles, shape : number of particles x 5 (x,y,vx,vy,vz)
        count (ndarray, int): array of integers containing the number of times the associated particle in *arr* collided
        alpha (ndarray, float) : array of float containing the angle between the speed and the normal to the wall it collided
        law (function): returns a probability of reactions ...
    """

    # NOTE : we can sample only for the particles that collided - which would give less computations
    # however it requires a loop (or a copy)
    # or we can draw for the whole array of particles which is crearly suboptimal
    idx_reactions = []
    mean_proba = 0
    s = 0
    for k, c in enumerate(count):
        if(c>0):
            s+= 1
            proba_reaction = law(arr[k], c, alpha[k])
            mean_proba += proba_reaction
            rdm_uniform_draw = np.random.random() # we can maybe draw them all out (as we are anyway counting the number of colliding particles outside the function)
            if(proba_reaction > rdm_uniform_draw):
                idx_reactions.append(k)

    return np.array(idx_reactions, dtype = int), mean_proba/s if s!=0 else 0


# -------------------------------- reactions ---------------------------------- #

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
            

