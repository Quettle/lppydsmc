# I should do better than that.
def size_one_save(N, target_space = None, nb_iterations = None, saving_period = None, with_processing = False, as_csv = True):
    # from pprint import pprint
    V = 6
    if(as_csv):
        W = (6*22+1+2)/6 # if float32 are saved as str, which is the case in csv, then, we can expect something of this magniture. 
        # I should maybe make sure I am using scientific notation.
        if(with_processing): # for norm and norm squared
            W+=22*2/6
    else:
        W = (6*4+1+2)/6
        if(with_processing):
            W+=4*2/6
    # which yields :
    size = N*(V*W+4)/1024**2
    
    out = {
        'N':N,
        'Size one save (MB)':size
    }
    
    if(nb_iterations != None):
        if(saving_period != None):
            size *= nb_iterations//saving_period
            out['Nb iterations']=nb_iterations
            out['Saving period']=saving_period
            out['Nb saves (induced)']=nb_iterations//saving_period
            out['Total size (induced)(MB)']=size
        else:
            size *= nb_iterations
            out['Nb iterations']=nb_iterations
            out['Saving period']=1
            out['Nb saves (induced)']=nb_iterations
            out['Total size (induced) (MB)']=size
    else:
        if(saving_period != None and target_space != None):
            nb_saves = int(target_space/size)
            nb_iterations = nb_saves*saving_period
            out['Nb iterations (induced)']=nb_iterations
            out['Saving period']=saving_period
            out['Nb saves (induced)']=nb_saves
            out['Total size (MB)']=target_space
    # pprint(out)
    return out
