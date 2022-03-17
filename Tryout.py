import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import networkx as nx
# import netgraph # for movable nodes
from scipy.stats import shapiro
from pgmpy.estimators import BDeuScore, K2Score, BicScore, ExpectationMaximization
from pgmpy.models import BayesianModel, BayesianNetwork
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator
from FriedmanGoldszmidt import FG_estimator, SuffStatBicScore, MLE_FG



def rename_values(data: pd.DataFrame):
    data["gesl"].replace([1, 2], ["man", "vrouw"], inplace=True)
    data["tumsoort"].replace([501300, 502200, 503200], ["invasief", "ductaal", "lubolair"],
                             inplace=True)
    data["her2_stat"].replace([0.00000, 1.00000, 2.00000, 3.00000, 4.00000, 7.00000, 9.00000],
                              ["negatief", "negatief", "onduidelijk", "positief", np.nan,
                               np.nan, "onbekend"],
                              inplace=True)
    data["er_stat"].replace([0.00000, 1.00000, 9.00000], ["negatief", "positief", "onbekend"],
                            inplace=True)
    data["pr_stat"].replace([0.00000, 1.00000, 9.00000], ["negatief", "positief", "onbekend"],
                            inplace=True)
    data["uitgebr_chir_code"].replace(["131C50", "132C50", "141C50", "142C50", "315000", "190000",
                                       np.nan],
                                      ["ja", "ja", "ja", "ja", "ja", "ja", "nee"],
                                      inplace=True)
    data["vit_stat"].replace([0, 1], ["levend", "overleden"], inplace=True)
    for treatment in ["chemo", "target", "rt", "horm"]:
        data[treatment].replace([0.00000, 1.00000, 2.00000, 3.00000, 4.00000],
                                ["nee", "ja", "ja", "ja", "ja", ],
                                inplace=True)

def calc_survival(data: pd.DataFrame):
    data["1_year_survival"] = np.where(data["vit_stat_int"] > 365, "ja",
        np.where(data["vit_stat"] == "overleden", "nee", "onbekend"))
    data["5_year_survival"] = np.where(data["vit_stat_int"] > (365*5), "ja",
        np.where(data["vit_stat"] == "overleden", "nee", "onbekend"))
    data.drop(["vit_stat", "vit_stat_int"], axis=1, inplace=True)

def inspect_age_var(data: pd.DataFrame):
    # data.hist(column="leeft")
    # plt.show()
    #Since age is a peaked distribution, we use simple quantiled discretization
    data["disc_leeft"] = pd.qcut(data["leeft"], 3, labels=["jong", "gemiddeld", "oud"])
    data.drop("leeft", axis=1, inplace=True)

def simplify_stadia(data: pd.DataFrame):
    data["stadium"].replace(["1A", "1B", "2A", "2B", "3A", "3B", "3C",
                                       np.nan],
                                      ["1", "1", "2", "2", "3", "3", "3", "Niet te bepalen"],
                                      inplace=True)

# data = pd.read_csv("NKR_IKNL_breast_syntheticdata.csv", sep=";")
# char_var = ["leeft", "gesl"]
# tumor_var = ["tumsoort", "stadium", "her2_stat", "er_stat", "pr_stat"]
# treatment_var = ["uitgebr_chir_code", "chemo", "target", "horm", "rt"]
# outcome_var = ["vit_stat", "vit_stat_int"]
# data = pd.DataFrame(data, columns=char_var+tumor_var+treatment_var+outcome_var)
# rename_values(data)
# calc_survival(data)
# inspect_age_var(data)
# simplify_stadia(data)

# calculate_sufficient_stats(data)

data = pd.read_csv("asia10K.csv")
data = pd.DataFrame(data, columns=["Smoker", "LungCancer", "X-ray"])
test_data = data[:2000]
new_data = data[2000:]

#Create initial model from data
bic = BicScore(test_data)
hc = HillClimbSearch(test_data)
model = hc.estimate(scoring_method=bic)

#Create Bic scorer that works with suff stats instead of data to evaluate structures
bic_FG = SuffStatBicScore(test_data)
bic_FG.calculate_sufficient_stats(model)
bic_FG.score(model)

FG = FG_estimator(test_data)
updated_model = FG.update(new_data=new_data, scoring_method=bic_FG, start_dag=model, tabu_length=0,
                          data_per_search=100)

BN = BayesianNetwork(updated_model)
mle_FG = MLE_FG(bic_FG.suff, BN, test_data)
param = mle_FG.get_parameters()
BN.add_cpds(*param)
EM = ExpectationMaximization

# print(model.edges())
# fig = plt.figure()
# graph = nx.draw(best_model, with_labels=True, pos=nx.spring_layout(best_model))
# plt.show()
