#!/usr/bin/env python
import itertools
from itertools import permutations, combinations
from collections import deque

import networkx as nx
import numpy as np
import pandas as pd
from math import log
from tqdm.auto import trange
from pgmpy.estimators import BicScore
from itertools import chain
from joblib import Parallel, delayed
from pgmpy.estimators import ParameterEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork

from pgmpy.estimators import (
    StructureScore,
    StructureEstimator,
    K2Score,
    ScoreCache,
    BDeuScore,
    BDsScore,
)
from pgmpy.base import DAG
from pgmpy.global_vars import SHOW_PROGRESS

from PK import PKalgorithm

class FG_estimator(StructureEstimator):
    def __init__(self, data, use_cache=True, **kwargs):
        """
        Class for heuristic searches for DAGs, to update
        network structure using new data. `estimate` attempts to find a model with optimal score
        given some new data.

        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states (or values)
            that the variable can take. If unspecified, the observed values in the data set
            are taken to be the only possible states.

        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
            This sets the behavior of the `state_count`-method.

        use_caching: boolean
            If True, uses caching of score for faster computation.
            Note: Caching only works for scoring methods which are decomposible. Can
            give wrong results in case of custom scoring methods.

        References
        ----------
        Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
        Section 18.4.3 (page 811ff)
        """
        self.use_cache = use_cache
        self.suff = None
        super(FG_estimator, self).__init__(data, **kwargs)

    def legal_operations(
        self,
        new_data,
        model,
        score_method,
        structure_score,
        tabu_list,
        max_indegree,
        black_list,
        white_list,
        fixed_edges,
    ):
        """Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip a single edge. For details on scoring
        see Koller & Friedman, Probabilistic Graphical Models, Section 18.4.3.3 (page 818).
        If a number `max_indegree` is provided, only modifications that keep the number
        of parents for each node below `max_indegree` are considered. A list of
        edges can optionally be passed as `black_list` or `white_list` to exclude those
        edges or to limit the search.
        """
        score = score_method.local_score
        tabu_list = set(tabu_list)

        # Step 1: Get all legal operations for adding edges.
        potential_new_edges = (
            set(permutations(self.variables, 2))
            - set(model.edges())
            - set([(Y, X) for (X, Y) in model.edges()])
        )

        for (X, Y) in potential_new_edges:
            # Check if adding (X, Y) will create a cycle.
            if not nx.has_path(model, Y, X):
                operation = ("+", (X, Y))
                if (
                    (operation not in tabu_list)
                    and ((X, Y) not in black_list)
                    and ((X, Y) in white_list)
                ):
                    old_parents = model.get_parents(Y)
                    old_parents.sort()
                    new_parents = old_parents + [X]
                    new_parents.sort()
                    if not score_method.can_evaluate(Y, new_parents):
                        self.suff = score_method.add_to_suff(new_data, Y, new_parents)
                    if len(new_parents) <= max_indegree:
                        score_delta = score(Y, new_parents) - score(Y, old_parents)
                        score_delta += structure_score("+")
                        yield (operation, score_delta)

        # Step 2: Get all legal operations for removing edges
        for (X, Y) in model.edges():
            operation = ("-", (X, Y))
            if (operation not in tabu_list) and ((X, Y) not in fixed_edges):
                old_parents = model.get_parents(Y)
                new_parents = old_parents[:]
                new_parents.remove(X)
                if not score_method.can_evaluate(Y, new_parents):
                    self.suff = score_method.add_to_suff(new_data, Y, new_parents)
                score_delta = score(Y, new_parents) - score(Y, old_parents)
                score_delta += structure_score("-")
                yield (operation, score_delta)

        # Step 3: Get all legal operations for flipping edges
        for (X, Y) in model.edges():
            # Check if flipping creates any cycles
            if not any(
                map(lambda path: len(path) > 2, nx.all_simple_paths(model, X, Y))
            ):
                operation = ("flip", (X, Y))
                if (
                    ((operation not in tabu_list) and ("flip", (Y, X)) not in tabu_list)
                    and ((X, Y) not in fixed_edges)
                    and ((Y, X) not in black_list)
                    and ((Y, X) in white_list)
                ):
                    old_X_parents = model.get_parents(X)
                    old_Y_parents = model.get_parents(Y)
                    new_X_parents = old_X_parents + [Y]
                    new_Y_parents = old_Y_parents[:]
                    new_Y_parents.remove(X)
                    if len(new_X_parents) <= max_indegree:
                        if not score_method.can_evaluate(Y, new_Y_parents):
                            self.suff = score_method.add_to_suff(new_data, Y, new_Y_parents)
                        if not score_method.can_evaluate(X, new_X_parents):
                            self.suff = score_method.add_to_suff(new_data, X, new_X_parents)
                        score_delta = (
                            score(X, new_X_parents)
                            + score(Y, new_Y_parents)
                            - score(X, old_X_parents)
                            - score(Y, old_Y_parents)
                        )
                        score_delta += structure_score("flip")
                        yield (operation, score_delta)

        # Could add removal of unused suff stats for memory efficiency

    def check_outcoming_edges(
        self,
        new_data,
        model,
        score_method,
        structure_score,
        tabu_list,
        max_indegree,
        black_list,
        white_list,
        fixed_edges,
        new_vars,
    ):
        """Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip a single edge. For details on scoring
        see Koller & Friedman, Probabilistic Graphical Models, Section 18.4.3.3 (page 818).
        If a number `max_indegree` is provided, only modifications that keep the number
        of parents for each node below `max_indegree` are considered. A list of
        edges can optionally be passed as `black_list` or `white_list` to exclude those
        edges or to limit the search.
        """
        score = score_method.local_score
        tabu_list = set(tabu_list)

        # Step 1: Get all legal operations for adding edges.
        # potential_new_edges = (
        #         set(permutations(self.variables, 2))
        #         - set(model.edges())
        #         - set([(Y, X) for (X, Y) in model.edges()])
        # )
        outgoing_new_edges = (
                set(itertools.product(new_vars, self.variables))
                - set(model.edges())
                - set([(Y, X) for (X, Y) in model.edges()])
        )

        incoming_new_edges = (
                set(itertools.product(self.variables, new_vars))
                - set(model.edges())
                - set([(Y, X) for (X, Y) in model.edges()])
        )

        new_edges = outgoing_new_edges.union(incoming_new_edges)

        for (X, Y) in new_edges:
            # Check if adding (X, Y) will create a cycle.
            if not nx.has_path(model, Y, X):
                operation = ("+", (X, Y))
                if (
                        (operation not in tabu_list)
                        and ((X, Y) not in black_list)
                        and ((X, Y) in white_list)
                ):
                    old_parents = model.get_parents(Y)
                    old_parents.sort()
                    new_parents = old_parents + [X]
                    new_parents.sort()
                    # print(f' trying to evaluate: {(Y, new_parents)}')
                    if not score_method.can_evaluate(Y, new_parents):
                        print(f"Cannot evaluate {(Y, new_parents)}")
                        self.suff = score_method.add_to_suff(new_data, Y, new_parents)
                    if len(new_parents) <= max_indegree:
                        score_delta = score(Y, new_parents) - score(Y, old_parents)
                        score_delta += structure_score("+")
                        if score_delta > 0:
                            yield (operation, score_delta)


    def variable_addition_update(
        self,
        new_data,
        new_vars,
        structure_scoring_method=None,
        parameter_scoring_method=None,
        start_dag=None,
        fixed_edges=set(),
        tabu_length=100,
        max_indegree=None,
        black_list=None,
        white_list=None,
        show_progress=True,
        ):
        PK = PKalgorithm(new_data, new_vars)
        new_suff = {}
        for variable_set in self.suff:
            suff = PK.merge(self.suff[variable_set], {x: self.state_names[x] for x in variable_set})
            key = list(variable_set) + new_vars
            key.sort()
            key = tuple(key)
            new_suff[key] = suff
        self.suff.update(new_suff)
        for var in new_vars:
            self.state_names[var] = PK.new_state_names[var]
            structure_scoring_method.state_names[var] = PK.new_state_names[var]

        # Add nodes to network, without edges
        model = start_dag
        for var in new_vars:
            model.add_node(var)
            self.variables = self.variables+new_vars


        # Step 1.3: Check fixed_edges
        if not hasattr(fixed_edges, "__iter__"):
            raise ValueError("fixed_edges must be an iterable")
        else:
            fixed_edges = set(fixed_edges)
            start_dag.add_edges_from(fixed_edges)
            if not nx.is_directed_acyclic_graph(start_dag):
                raise ValueError(
                    "fixed_edges creates a cycle in start_dag. Please modify either fixed_edges or start_dag."
                )

        # Step 1.4: Check black list and white list
        black_list = set() if black_list is None else set(black_list)
        white_list = (
            set([(u, v) for u in self.variables for v in self.variables])
            if white_list is None
            else set(white_list)
        )

        # Step 1.5: Initialize max_indegree, tabu_list, and progress bar
        if max_indegree is None:
            max_indegree = float("inf")
        tabu_list = deque(maxlen=tabu_length)

        best_operation = True
        while best_operation != None:
            #get all legal parent set additions
            best_operation, best_score_delta = max(
                self.check_outcoming_edges(
                    new_data,
                    start_dag,
                    structure_scoring_method,
                    structure_scoring_method.structure_prior_ratio,
                    tabu_list,
                    max_indegree,
                    black_list,
                    white_list,
                    fixed_edges,
                    new_vars,
                ),
                key=lambda t: t[1],
                default=(None, None),
            )

            if best_operation != None:
                # perform all operations that result in score increase:
                print(f"Changed model by {best_operation} , leading to score increase "
                      f"of {best_score_delta}")
                model.add_edge(*best_operation[1])
                tabu_list.append(("-", best_operation[1]))

            parameter_scoring_method.set_suff_stats(self.suff)
            new_cpds = MLE_FG(data=new_data, suff=self.suff,
                              model=model).get_parameters(n_jobs=-1, weighted=False)
            model.cpds = new_cpds
            parameter_scoring_method.set_model(BayesianNetwork(model))

        return model

    def update(
        self,
        new_data,
        structure_scoring_method=None,
        parameter_scoring_method=None,
        start_dag=None,
        fixed_edges=set(),
        tabu_length=100,
        max_indegree=None,
        black_list=None,
        white_list=None,
        data_per_search=100,
        show_progress=True,
    ):

        """
        Performs local hill climb search to estimates the `DAG` structure that
        has optimal score, according to the scoring method supplied. Starts at
        model `start_dag` and proceeds by step-by-step network modifications
        until a local maximum is reached. Only estimates network structure, no
        parametrization.

        Parameters
        ----------
        scoring_method: str or StructureScore instance
            The score to be optimized during structure estimation.  Supported
            structure scores: k2score, bdeuscore, bdsscore, bicscore. Also accepts a
            custom score but it should be an instance of `StructureScore`.

        start_dag: DAG instance
            The starting point for the local search. By default a completely
            disconnected network is used.

        fixed_edges: iterable
            A list of edges that will always be there in the final learned model.
            The algorithm will add these edges at the start of the algorithm and
            will never change it.

        tabu_length: int
            If provided, the last `tabu_length` graph modifications cannot be reversed
            during the search procedure. This serves to enforce a wider exploration
            of the search space. Default value: 100.

        max_indegree: int or None
            If provided and unequal None, the procedure only searches among models
            where all nodes have at most `max_indegree` parents. Defaults to None.
        black_list: list or None
            If a list of edges is provided as `black_list`, they are excluded from the search
            and the resulting model will not contain any of those edges. Default: None
        white_list: list or None
            If a list of edges is provided as `white_list`, the search is limited to those
            edges. The resulting model will then only contain edges that are in `white_list`.
            Default: None

        epsilon: float (default: 1e-4)
            Defines the exit condition. If the improvement in score is less than `epsilon`,
            the learned model is returned.

        max_iter: int (default: 1e6)
            The maximum number of iterations allowed. Returns the learned model when the
            number of iterations is greater than `max_iter`.

        Returns
        -------
        model: `DAG` instance
            A `DAG` at a (local) score maximum.
        """
        # Step 1.2: Check the start_dag
        if start_dag is None:
            start_dag = DAG()
            start_dag.add_nodes_from(self.variables)
        elif not isinstance(start_dag, DAG) or not set(start_dag.nodes()) == set(
            self.variables
        ):
            raise ValueError(
                "'start_dag' should be a DAG with the same variables as the data set, or 'None'."
            )

        # Step 1.3: Check fixed_edges
        if not hasattr(fixed_edges, "__iter__"):
            raise ValueError("fixed_edges must be an iterable")
        else:
            fixed_edges = set(fixed_edges)
            start_dag.add_edges_from(fixed_edges)
            if not nx.is_directed_acyclic_graph(start_dag):
                raise ValueError(
                    "fixed_edges creates a cycle in start_dag. Please modify either fixed_edges or start_dag."
                )

        # Step 1.4: Check black list and white list
        black_list = set() if black_list is None else set(black_list)
        white_list = (
            set([(u, v) for u in self.variables for v in self.variables])
            if white_list is None
            else set(white_list)
        )

        # Step 1.5: Initialize max_indegree, tabu_list, and progress bar
        if max_indegree is None:
            max_indegree = float("inf")

        tabu_list = deque(maxlen=tabu_length)
        current_model = start_dag.copy()

        update_steps = len(new_data)/data_per_search
        if update_steps < 1:
            print("New data sample to small for update step!")

        if show_progress and SHOW_PROGRESS:
            iteration = trange(int(update_steps))
        else:
            iteration = range(int(update_steps))


        # Step 2: For each iteration, find the best scoring operation and
        #         do that to the current model. If no legal operation is
        #         possible, sets best_operation=None.
        for ind in iteration:
            update_data = new_data[ind*data_per_search:(ind+1)*data_per_search]
            parameter_scoring_method.set_suff_stats(self.suff)
            parameter_scoring_method.set_model(BayesianNetwork(current_model))
            weights = parameter_scoring_method.compute_weights(update_data, latent_card={})
            self.suff = structure_scoring_method.update_suff(weights, weighted=True, decay=0.01)

            best_operation, best_score_delta = max(
                self.legal_operations(
                    new_data,
                    current_model,
                    structure_scoring_method,
                    structure_scoring_method.structure_prior_ratio,
                    tabu_list,
                    max_indegree,
                    black_list,
                    white_list,
                    fixed_edges,
                ),
                key=lambda t: t[1],
                default=(None, None),
            )
            #some check that the model can remain unchanged
            if best_operation is None or best_score_delta <= 0:
                print("The current best model remains unchanged!")
            else:
                print(f"Changed model by {best_operation} , leading to score increase "
                      f"of {best_score_delta}")
                if best_operation[0] == "+":
                    current_model.add_edge(*best_operation[1])
                    tabu_list.append(("-", best_operation[1]))
                elif best_operation[0] == "-":
                    current_model.remove_edge(*best_operation[1])
                    tabu_list.append(("+", best_operation[1]))
                elif best_operation[0] == "flip":
                    X, Y = best_operation[1]
                    current_model.remove_edge(X, Y)
                    current_model.add_edge(Y, X)
                    tabu_list.append(best_operation)

            new_cpds = MLE_FG(data=self.data, suff=self.suff,
                model=current_model).get_parameters(n_jobs=-1, weighted=False)
            current_model.cpds = new_cpds

        # Step 3: Return if no more improvements or maximum iterations reached.
        return current_model


class SuffStatBicScore(StructureScore):
    def __init__(self, data, **kwargs):
        """
        Class for Bayesian structure scoring for BayesianNetworks with
        Dirichlet priors.  The BIC/MDL score ("Bayesian Information Criterion",
        also "Minimal Descriptive Length") is a log-likelihood score with an
        additional penalty for network complexity, to avoid overfitting.  The
        `score`-method measures how well a model is able to describe the given
        data set.

        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states (or values)
            that the variable can take. If unspecified, the observed values in the data set
            are taken to be the only possible states.

        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
            This sets the behavior of the `state_count`-method.

        References
        ---------
        [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
        Section 18.3.4-18.3.6 (esp. page 802)
        [2] AM Carvalho, Scoring functions for learning Bayesian networks,
        http://www.lx.it.pt/~asmc/pub/talks/09-TA/ta_pres.pdf
        """
        self.suff = {}
        self.N = len(data)
        self.bic = BicScore(data)
        super(SuffStatBicScore, self).__init__(data, **kwargs)

    def calculate_sufficient_stats(self, model):
        for node in model.nodes():
            parents = model.get_parents(node)
            vars = parents + [node]
            candidate_parents = [item for item in list(model.nodes()) if item not in vars]
            #Extra parents
            for candidate in candidate_parents:
                potential_parents = parents.copy()
                potential_parents += [candidate]
                key = potential_parents + [node]
                key.sort()
                key = tuple(key)
                for alternative_key in self.suff.copy().keys():
                    if set(alternative_key).issubset(key) and not alternative_key == key:
                           self.suff.pop(alternative_key)
                if not any(set(key).issubset(alternative_key) for alternative_key in
                        self.suff.keys()):
                    counts = self.bic.state_counts(key[0], key[1:])
                    self.suff[key] = counts
        print(f"The total number of stored sufficient statistic tables is {len(self.suff)}")

    def update_suff(self, new_data, weighted=False, decay=0):
        self.N += len(new_data)
        new_bic = BicScore(new_data, state_names=self.bic.state_names)
        for key in self.suff:
            old_suff = self.suff[key]
            added_suff = new_bic.state_counts(key[0], key[1:], weighted=weighted)
            self.suff[key] = old_suff * (1-decay) + added_suff
        print(f"The stored sufficient statistics were updated!")
        return self.suff

    def can_evaluate(self, variable, parents):
        for key in self.suff.keys():
            vars = parents + [variable]
            if set(vars).issubset(key):
                return True
        return False

    def add_to_suff(self, new_data, variable, parents):
        print(f"A new suff stat was added for var: {variable} and parents: {parents}")
        new_bic = BicScore(new_data, state_names=self.bic.state_names)
        key = parents + [variable]
        key.sort()
        key = tuple(key)
        counts = new_bic.state_counts(key[0], key[1:])
        self.suff[key] = counts
        return self.suff

    def simplify_suff(self, variable, parents):
        # print("Simplify suff called")
        for alternative_keys in self.suff.keys():
            # print(f"Alternative suff under consideration: {alternative_keys}")
            key = parents + [variable]
            key.sort()
            key = tuple(key)
            # print(f'key under eval: {key}')
            if set(key).issubset(alternative_keys):
                # print("Subset detected, starting reduction")
                if variable != alternative_keys[0]:
                    # print("Retarget necessary")
                    retargetted_suff = self.suff[alternative_keys].stack(level=variable).unstack(
                        level=alternative_keys[0])
                    # print(f"retargetted: {retargetted_suff}")
                else:
                    retargetted_suff = self.suff[alternative_keys]
                if parents != []:
                    reduced_suff = retargetted_suff.groupby(axis=1, level=parents).sum()
                else:
                    reduced_suff = pd.DataFrame(retargetted_suff.sum(axis=1))
                # print(f'reduction leads to: {reduced_suff}')
                return reduced_suff
        # print("Failing to simplify a sufficient statistic!")
        return False

    def local_score(self, variable, parents):
        'Computes a score that measures how much a \
        given variable is "influenced" by a given list of potential parents.'
        par = list(parents)
        # print(f"Evaluating score for variable {variable} with parents {par}")
        var_states = self.state_names[variable]
        var_cardinality = len(var_states)
        # print(f"States of variable are {var_states}, thus cardinality is {var_cardinality}")
        # print(f"parents for which state_counts are searched: {par}")
        par.sort()
        key = tuple([variable]) + tuple(par)
        # print(f"parents for which state_counts are searched: {par}")
        state_counts = self.simplify_suff(variable, par)
        # print(f"The suff stat is {state_counts}")
        if not type(state_counts) == pd.DataFrame:
            print(f"this suff stat {key} is not in the list!")
        sample_size = self.N
        num_parents_states = float(state_counts.shape[1])
        # print(f'The parents have {num_parents_states} states')
        counts = np.asarray(state_counts)
        log_likelihoods = np.zeros_like(counts, dtype=float)

        # Compute the log-counts
        np.log(counts, out=log_likelihoods, where=counts > 0)
        # print(f"the log_counts are {log_likelihoods}")

        # Compute the log-conditional sample size
        log_conditionals = np.sum(counts, axis=0, dtype=float)
        np.log(log_conditionals, out=log_conditionals, where=log_conditionals > 0)
        # print(f"the log conditionals are {log_conditionals}")

        # Compute the log-likelihoods
        log_likelihoods -= log_conditionals
        log_likelihoods *= counts
        # print(f"log likelihoods are {log_likelihoods}")

        normalization = np.sum(counts)
        score = np.sum(log_likelihoods)
        # print(f"Score is than {log_likelihoods}")
        score -= 0.5 * log(normalization) * num_parents_states * (var_cardinality - 1)
        # print(f"Score is then changed to {score}")
        # print(normalization)
        return score/normalization

class MLE_FG(ParameterEstimator):
    def __init__(self, suff, model, data, **kwargs):
        """
        Class used to compute parameters for a model using Maximum Likelihood Estimation.

        Parameters
        ----------
        model: A pgmpy.models.BayesianNetwork instance

        data: pandas DataFrame object
            DataFrame object with column names identical to the variable names of the network.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states
            that the variable can take. If unspecified, the observed values
            in the data set are taken to be the only possible states.

        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.NaN` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
        """
        if not isinstance(model, BayesianNetwork):
            raise NotImplementedError(
                "Maximum Likelihood Estimate is only implemented for BayesianNetwork"
            )
        elif set(model.nodes()) > set(data.columns):
            raise ValueError(
                f"Maximum Likelihood Estimator works only for models with all observed variables. Found latent variables: {model.latents}."
            )
        self.suff = suff
        super(MLE_FG, self).__init__(model, data, **kwargs)

    def get_parameters(self, n_jobs=-1, weighted=False):
        """
        Method to estimate the model parameters (CPDs) using Maximum Likelihood
        Estimation.

        Parameters
        ----------
        n_jobs: int (default: -1)
            Number of jobs to run in parallel. Default: -1 uses all the processors.

        weighted: bool
            If weighted=True, the data must contain a `_weight` column specifying the
            weight of each datapoint (row). If False, assigns an equal weight to each
            datapoint.

        Returns
        -------
        parameters: list
            List of TabularCPDs, one for each variable of the model

        n_jobs: int
            Number of processes to spawn

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.estimators import MaximumLikelihoodEstimator
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 4)),
        ...                       columns=['A', 'B', 'C', 'D'])
        >>> model = BayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D')])
        >>> estimator = MaximumLikelihoodEstimator(model, values)
        >>> estimator.get_parameters()
        [<TabularCPD representing P(C:2) at 0x7f7b534251d0>,
        <TabularCPD representing P(B:2 | C:2, A:2) at 0x7f7b4dfd4da0>,
        <TabularCPD representing P(A:2) at 0x7f7b4dfd4fd0>,
        <TabularCPD representing P(D:2 | C:2) at 0x7f7b4df822b0>]
        """

        parameters = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(self.estimate_cpd)(node, weighted) for node in self.model.nodes()
        )

        return parameters

    def simplify_suff(self, variable, parents):
        for alternative_keys in self.suff.keys():
            key = parents + [variable]
            key.sort()
            key = tuple(key)
            if set(key).issubset(alternative_keys):
                if variable != alternative_keys[0]:
                    retargetted_suff = self.suff[alternative_keys].stack(level=variable).unstack(
                        level=alternative_keys[0])
                else:
                    retargetted_suff = self.suff[alternative_keys]
                if parents != []:
                    reduced_suff = retargetted_suff.groupby(axis=1, level=parents).sum()
                    if not isinstance(reduced_suff.columns, pd.MultiIndex):
                        reduced_suff.columns = pd.MultiIndex.from_arrays(
                            [reduced_suff.columns]
                        )
                else:
                    reduced_suff = retargetted_suff.sum(axis=1).reindex(self.state_names[
                                                                            variable]).to_frame()
                return reduced_suff
        print("Failing to simplify a sufficient statistic!")
        return False

    def estimate_cpd(self, node, weighted=False):
        """
        Method to estimate the CPD for a given variable.

        Parameters
        ----------
        node: int, string (any hashable python object)
            The name of the variable for which the CPD is to be estimated.

        weighted: bool
            If weighted=True, the data must contain a `_weight` column specifying the
            weight of each datapoint (row). If False, assigns an equal weight to each
            datapoint.

        Returns
        -------
        CPD: TabularCPD

        """
        parents = sorted(list(self.model.get_parents(node)))
        state_counts = self.simplify_suff(node, parents)
        # print(f"columns for suff: {state_counts1}")
        # state_counts = self.state_counts(node, weighted=weighted)
        # print(f"columns for state_counts: {state_counts}")

        # if a column contains only `0`s (no states observed for some configuration
        # of parents' states) fill that column uniformly instead
        state_counts.loc[:, (state_counts == 0).all()] = 1

        parents_cardinalities = [len(self.state_names[parent]) for parent in parents]
        node_cardinality = len(self.state_names[node])

        # Get the state names for the CPD
        state_names = {node: list(state_counts.index)}
        if parents:
            state_names.update(
                {
                    state_counts.columns.names[i]: list(state_counts.columns.levels[i])
                    for i in range(len(parents))
                }
            )

        cpd = TabularCPD(
            node,
            node_cardinality,
            np.array(state_counts),
            evidence=parents,
            evidence_card=parents_cardinalities,
            state_names={var: self.state_names[var] for var in chain([node], parents)},
        )
        cpd.normalize()
        return cpd


