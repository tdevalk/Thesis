from pgmpy.estimators import BDeuScore, K2Score, BicScore, ExpectationMaximization
from itertools import product, chain
import numpy as np
import pandas as pd

class Fixed_EM(ExpectationMaximization):
    def __init__(self, model, data, **kwargs):
        super(ExpectationMaximization, self).__init__(model, data, **kwargs)
        self.model_copy = self.model.copy()

    def _compute_weights(self, latent_card):
        cache = []

        data_unique = self.data.drop_duplicates()
        n_counts = self.data.fillna("Missing").groupby(list(self.data.columns), dropna=False).size(
        ).to_dict()

        for i in range(data_unique.shape[0]):
            missing = latent_card.copy()
            missing_vars = data_unique.iloc[i][data_unique.iloc[i].isnull()].keys()
            missing.update({var: self.data[var].dropna().unique() for var in missing_vars if var not in missing.keys()})
            combies = list(product(*[value for value in missing.values()]))
            missing_combinations = np.array(combies, dtype=object)
            df = data_unique.iloc[[i] * missing_combinations.shape[0]].reset_index(
                drop=True
            )
            for index, latent_var in enumerate(missing.keys()):
                df[latent_var] = missing_combinations[:, index]

            weights = df.apply(lambda t: self._get_likelihood(dict(t)), axis=1)
            df["_weight"] = ((weights / weights.sum()) * n_counts[
                tuple(data_unique.iloc[i].fillna("Missing"))
            ])
            cache.append(df)

        full_assignments = pd.concat(cache, copy=False)
        df = full_assignments.groupby(full_assignments.columns.tolist()[:-1]).sum().reset_index()
        normalized = df["_weight"] / df["_weight"].sum() * len(self.data)
        df["_weight"] = normalized
        return df