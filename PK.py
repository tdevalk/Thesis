import numpy as np
import pandas as pd
import itertools
from pgmpy.estimators import BaseEstimator

class PKalgorithm:

    def __init__(self, new_data, new_vars, EM, Model):
        self.newdata = new_data #WHAT TO DO WITH NAN VALUES HERE?
        # Remove data that does not contain new vars, since imputation is not possible
        # Complete other data using current model
        self.vars = list(self.newdata.columns)
        self.new_vars = new_vars
        self.EM = EM
        self.model = Model
        self.EM.set_model(self.model)
        for var in self.new_vars:
            self.newdata = self.newdata[self.newdata[var].notnull()]
        self.weights = self.EM.compute_weights(self.newdata, latent_assignments={"Treat. response": ["Complete", "Complete", "Partial", "Stabile",
                                                                  "Progressive", "Recurrence", "Death", "Other"]})
        print("weight has been computated")
        self.bic = BaseEstimator(self.weights, state_names=EM.state_names)
        self.new_state_names = {x: self.bic.state_names[x] for x in self.new_vars}

    def merge(self, suff, old_state_names):
        vars = list(old_state_names.keys()) #All variables included in old suff stat
        counts1 = self.bic.state_counts(vars[0], self.new_vars+vars[1:], weighted=True).stack(vars[1:]+self.new_vars)   #new suff stat
        counts2 = suff.unstack().reorder_levels(vars) #Old suff stat
        result = counts1.copy()        #End product after merge initialized


        # Generate all possible combined value assignments
        allNames = vars #Ensure order is correct
        old_combinations = list(itertools.product(*(old_state_names[Name] for Name in allNames)))
        allNames = self.new_vars
        new_combinations = list(itertools.product(*(self.new_state_names[Name] for Name in allNames)))
        # calculate new suff stat for each combination
        for old_combi in old_combinations:
            for new_combi in new_combinations:
                new_count = counts1[old_combi+new_combi]
                old_count = counts2[old_combi]
                normalizer = np.sum([counts1[old_combi+alt_combi] for alt_combi in new_combinations])
                if normalizer == 0:
                    result[old_combi+new_combi] = 0
                else:
                    result[old_combi+new_combi] = ((old_count/normalizer)*new_count) + new_count
        # Fundamental problem: first index should be alphabetical first index, not just the one it was conditioned upon
        alphabetical_order = vars+self.new_vars
        alphabetical_order.sort()
        return result.unstack(alphabetical_order[1:])
        # return result.unstack(vars[1:]+self.new_vars)

    def revise_model(self, model, suff_stats, state_names):
        new_suff = {}
        for variable_set in suff_stats:
            suff = self.merge(suff_stats[variable_set], {x: state_names[x] for x in variable_set})
            new_suff.update(suff)

        #Update all suff stats
        #Check value of adding new var to each parent set



