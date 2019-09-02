from dask import delayed, compute
import numpy as np
import itertools
from scipy.stats import rankdata


# Rank predictions, one column at a time.
# Structure: shape 0 = classifier, shape 1 = event, shape 2 = class
# Evaluate predictions performances
#@delay
def reduction(preds,truth, perf_f,nbr_to_select=6,filter="perf"):
    nbr_clf = preds.shape[0]
    nbr_obs = preds.shape[1]
    nbr_class = preds.shape[2]
    
    def rank_pred(pred): return np.sort(np.max(pred,axis=1))
    def perf_func(pred,truth): return perf_f(np.argmax(pred,axis=1),truth)
    def cd_calc(last_rpred,ranked_pred): return np.sum(np.square(last_rpred - ranked_pred))
    def cd_bag_calc(ranked_preds, pairs): return [
                                                [np.sum(np.square(ranked_preds[pair[0]] - ranked_preds[pair[0]])),
                                                 pair[0],
                                                 pair[1]]
                                                for pair in pairs]
    def cd_mat_calc(coords_cd, cd_mat): 
        for coord_cd in coords_cd:
            cd_mat[coord_cd[0],coord_cd[1]] = cd_mat[coord_cd[1],coord_cd[0]] = coord_cd[2]
        return cd_mat
    def cds_calc(cd_mat): return np.sum(cd_mat, axis = 0)
    #returns filtered_preds, filter = ["perf", "cds", "HH", "SR"]
    def reduct(preds, cds_preds, perfs, criteria = "perf"):
    	def perf_inds(perfs): return np.argsort(perfs)
    	def cds_inds(cds_preds): return np.argsort(cds_preds)
    	def HH(perfs,cds_preds,nbr_to_select):
    		if nbr_to_select%2 == 1: 
    			nbr_to_select = nbr_to_select-1
    		return np.concatenate((perf_inds(perfs)[:int(nbr_to_select/2)],cds_inds(cds_preds)[:int(nbr_to_select/2)]))

    	if criteria == "perf":
    		inds = perf_inds(perfs)[:nbr_to_select]
    	elif criteria == "cds":
    		inds = cds_inds(cds_preds)[:nbr_to_select]
    	elif criteria == "HH":
    		inds = HH(perfs,cds_preds,nbr_to_select)

    	return preds[inds,:,:]

    rank_pred = delayed(rank_pred)
    perf_func = delayed(perf_func)
    cd_calc = delayed(cd_calc)
    cd_bag_calc = delayed(cd_bag_calc)
    cd_mat_calc = delayed(cd_mat_calc)
    cds_calc = delayed(cds_calc)
    reduct = delayed(reduct)

    ranked_preds = []
    perfs = []
    cd_mat = np.array([[0.]*preds.shape[0]]*preds.shape[0])
    coords_cd = []
    for i in range(len(preds)):
        ranked_pred = rank_pred(preds[i])
        perf = perf_func(preds[i], truth)

        ranked_preds.append(ranked_pred)
        perfs.append(perf)

    pairs = list(itertools.combinations(range(nbr_clf),2))
    nbr_pair_per_bag = int(len(pairs)/30)
    nbr_bag_pairs = int(len(pairs)//nbr_pair_per_bag)
    pairs_left = int(len(pairs)%nbr_pair_per_bag)
    cd_bag = delayed([])
    for j in range(nbr_bag_pairs-1):
        cd_bag.extend(cd_bag_calc(ranked_preds,pairs[j*nbr_pair_per_bag:(j+1)*nbr_pair_per_bag]))
        if j == nbr_bag_pairs-2 and pairs_left > 0:
            cd_bag.extend(cd_bag_calc(ranked_preds,pairs[nbr_pair_per_bag*nbr_bag_pairs:]))
    cd_mat = cd_mat_calc(cd_bag, cd_mat)
    cds_preds = cds_calc(cd_mat)

    reduced = reduct(preds, cds_preds, perfs,criteria=filter)

    return reduced.compute()

# Calculate all combinations with all operations
def expansion(preds):
	nbr_clf = preds.shape[0]
	nbr_obs = preds.shape[1]
	nbr_class = preds.shape[2]

	#Use all merging functions
	def avg(preds): return np.mean(preds, axis = 0)
	#Convert to scores equivalent to ranks
	#preds.shape=[clf_nbr,events,class]
	def ranks_scores(preds):
		nbr_obs = preds.shape[1]
		nbr_class = preds.shape[2]
		ranks = []
		for pred in preds:
			rank = np.zeros(pred.shape)
			rank_class = np.argsort(pred,axis=1)
			# Scores rank/nbr_elements
			rank = np.array(rankdata(np.max(pred,axis=1),method='min'))/pred.shape[0]
			# class for 3 classes prb: 3/3*best_val,  2/3*best_val,  1/3*best_val
			rank = rank*(rank_class.T+1)/nbr_class
			rank = rank.T
			ranks.append(rank)

		ranks = np.array(ranks)
		return ranks

	def merges(preds,ranks): 
		n_preds = []
		if(preds.shape[0]<2): 
			n_preds.extend(np.reshape(preds,(1,nbr_obs,nbr_class)))
			n_preds.extend(np.reshape(ranks,(1,nbr_obs,nbr_class)))
		else:
			#Score Avg
			n_preds.extend(np.reshape(avg(preds),(1,nbr_obs,nbr_class)))
			#Rank Avg
			n_preds.extend(np.reshape(avg(ranks),(1,nbr_obs,nbr_class)))
		return np.array(n_preds)

	#Create all combinations indexes
	combis = []
	for n in range(nbr_clf-1):
		combis.extend(list(itertools.combinations(range(nbr_clf),n+1)))
	#Create rank table
	ranks = ranks_scores(preds)
	
	new_preds = []
	#Execute delayed merge
	for combi in combis:
		new_preds.extend(merges(preds[combi,:,:],ranks[combi,:,:]))
	
	new_preds = new_preds
	
	return np.array(new_preds)

# Evaluate / repeat
def graph(preds,truth,perf_f,nbr_layers=3,nbr_to_select=6,filter="perf"):
	if(preds.shape[0] > nbr_to_select):
		preds = reduction(preds,truth, perf_f,nbr_to_select,filter)
	for n in range(layer):
		preds = reduction(expansion(preds),truth, perf_f,nbr_to_select,filter)

	return reduction(preds, truth, perf_f,1,filter)
