import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
# import networkx as nx
# import netgraph # for movable nodes
from scipy.stats import shapiro
from pgmpy.estimators import BDeuScore, K2Score, BicScore, ExpectationMaximization
from pgmpy.models import BayesianModel, BayesianNetwork
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator
from FriedmanGoldszmidt import FG_estimator, SuffStatBicScore, MLE_FG, SuffStatBDeuScore
from EM_imputation import ExpectationMaximizationImputation

def rename_values(data: pd.DataFrame):
    data["Stage"].replace([1, 2, 3, 4], ["Stage 1", "Stage 2", "Stage 3", "Stage 4"], inplace=True)
    data["Gender"].replace([0, 1], ["Female", "Male"], inplace=True)
    data["Comorbidity"].replace([1, 2], ["No", "Yes"], inplace=True)
    data["Treat. mutation"].replace(np.NaN, "Not treatable", inplace=True)
    data["Histology"].replace([0, 1, 2, 3, 4, 5], ["NSCLC Non-adeno non-squamous", "NSCLC adeno", "NSCLC adeno", "Squamous", "SCLC", np.NaN],
                             inplace=True)
    data["M0 Ecog"].replace([0, 1, 2], ["Unsymptomatic", "Symptomatic", ">50% care"], inplace=True)
    data["T-stage"].replace([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                            [np.NaN, "Tin situ", "T1", "T1", "T1", "T1", "T2", "T2", "T3",
                             "T4", np.NaN, "T2", "T1", "T0"], inplace=True)
    data["N-stage"].replace([1, 2, 3, 4, 5, 6], ["N0", "N1", "N2", "N3", np.NaN, np.NaN], inplace=True)
    data["M-stage"].replace([1, 2, 3, 4, 5, 6], ["M0", "M1a", "M1b", "M1c", "M1", np.NaN], inplace=True)
    data["Treatment"].replace([1, 2, 3, 4, 5, 6], ["Immuno", "Immuno+", "Chemo(+radio)",
                                                      "Radio", "Targetted", "Chirurgical"], inplace=True)
    data["Prev. treatment"].replace([0, 1, 2, 3, 4, 6], ["No", "Chirurgical", "Chemoradio",
                                                      "Radio", "Systemic (chemo)", "Other"], inplace=True)
    data["Treat. response"].replace([0, 1, 2, 3, 4, 5, 6, 7], ["Complete", "Complete", "Partial", "Stabile",
                                                                  "Progressive", "Recurrence", "Death", "Other"],
                                       inplace=True)

def calc_survival(data: pd.DataFrame):
    datetime_str = '03/15/22 13:55:26'
    end_of_trial = datetime.datetime.strptime(datetime_str, '%m/%d/%y %H:%M:%S')
    data["survival days"] = np.where(pd.isnull(data["Date of death"]), end_of_trial - data["Start date"],
                                     data["Date of death"] - data["Start date"])
    data["M3 survival"] = np.where(data["survival days"] > datetime.timedelta(days = 15*7), "Yes",
        np.where(pd.isnull(data["Date of death"]), np.NaN, "No"))
    data["M6 survival"] = np.where(data["survival days"] > datetime.timedelta(days = 6*30.5), "Yes",
        np.where(pd.isnull(data["Date of death"]), np.NaN, "No"))

def disc_age(data: pd.DataFrame):
    # data.hist(column="leeft")
    # plt.show()
    #Since age is a peaked distribution, we use simple quantiled discretization
    data["Age"] = pd.qcut(data["Age"], 3, labels=["Young", "Average", "Old"]).astype("object")

def disc_HRQOL(data: pd.DataFrame, delta=True):
    if delta:
        data["delta HRQOL"] = data["M3 HRQOL"] - data["M0 HRQOL"]
        data["M3 HRQOL"] = np.where(data["delta HRQOL"] > 10, "Declined",
                 np.where((data["survival days"] < datetime.timedelta(days = 15*7)), "Death", np.where(pd.isnull(data["M3 HRQOL"]), np.NaN, "Stable")))
        data["M0 HRQOL"], bins = pd.qcut(data["M0 HRQOL"], 3, labels=["Low", "Average", "High"], retbins=True)
    else:
        data["M0 HRQOL"], bins = pd.qcut(data["M0 HRQOL"], 3, labels=["Low", "Average", "High"], retbins=True)
        data["M0 HRQOL"] = data["M0 HRQOL"].astype("object")
        data["M3 HRQOL"] = pd.qcut(data["M3 HRQOL"], 3, labels=["Low", "Average", "High"]).astype("object")
        # data["M3 HRQOL"] = pd.cut(data["M3 HRQOL"], bins=bins, labels=["Low", "Average", "High"]).astype("object")

def simplify_stadia(data: pd.DataFrame):
    data["stadium"].replace(["1A", "1B", "2A", "2B", "3A", "3B", "3C",
                                       np.nan],
                                      ["1", "1", "2", "2", "3", "3", "3", "Niet te bepalen"],
                                      inplace=True)

names = ["ID", "Institute", "Intervention", "Stage", "Age", "Gender", "Comorbidity",
         "Group", "Active", "M0 Ecog", "T-stage", "N-stage", "M-stage", "Histology",
         "Treat. mutation", "Start date", "Treatment", "Prev. treatment", "M0 HRQOL",
         "M3 HRQOL", "Treat. response", "Completion", "Dropout reason", "Date of death"]
dtypes = ["object", "object", "float64", "float64", "float64", "float64", "float64", "float64", "float64",
          "float64", "float64", "float64", "float64", "float64", "object", "object", "float64", "float64",
          "float64", "float64", "float64", "float64", "float64", "object"]
parse_dates = ["Start date", "Date of death"]
dtypes = dict(zip(names, dtypes))
data = pd.read_csv("220303_request_IRIS.csv", sep=";", header=0,
                   names=names, dtype=dtypes, parse_dates=parse_dates,
                   infer_datetime_format=True, dayfirst=True)
rename_values(data)
data=data[data["Histology"] != "SCLC"]
data=data[(data["Stage"] == "Stage 3") | (data["Stage"] == "Stage 4")]
calc_survival(data)
char_var = ["Age", "Gender", "Comorbidity"]
# tumor_var = ["T-stage", "N-stage", "M-stage", "Histology", "Treat. mutation"]
tumor_var = ["T-stage", "N-stage", "M-stage", "Histology"]
# treatment_var = ["Treatment", "Prev. treatment"]
treatment_var = ["Treatment"]
treatment_effect_var = ["Treat. response"]
outcome_var = ["M0 Ecog", "M0 HRQOL", "M3 HRQOL", "M3 survival", "M6 survival"]
disc_age(data)
disc_HRQOL(data)
data = pd.DataFrame(data, columns=char_var+tumor_var+treatment_var+treatment_effect_var+outcome_var)
# data = pd.DataFrame(data, columns=char_var)



test_data = data[:400]
test_data = pd.DataFrame(test_data, columns=test_data.columns[:-2])
new_data = data[400:]
new_data = pd.DataFrame(new_data, columns=new_data.columns[:-2])
extra_var_data = pd.DataFrame(data)

blacklist = pd.read_csv("BlacklistCSV.csv", sep=";")
blacklist = list(zip(blacklist["From"], blacklist["To"]))
whitelist = pd.read_csv("WhitelistCSV.csv", sep=";")
whitelist = list(zip(whitelist["From"], whitelist["To"]))
test_data_whitelist = [x for x in whitelist if not (x[0] in data.columns[-2:] or x[1] in data.columns[-2:])]

#Create initial model from data
# bic = BicScore(test_data)
bdeu = BDeuScore(test_data, equivalent_sample_size=15)
hc = HillClimbSearch(test_data)
model = hc.estimate(scoring_method=bdeu, epsilon=0, black_list=blacklist, fixed_edges=test_data_whitelist)
bn = BayesianNetwork(model)
bn.fit(test_data, estimator=MaximumLikelihoodEstimator)

#Create BDeu scorer that works with suff stats instead of data to evaluate structures
bdeu_FG = SuffStatBDeuScore(test_data)
bdeu_FG.calculate_sufficient_stats(bn, blacklist)
bdeu_FG.score(bn)

#Create Expectation Maximization scorer that works with suff stats instead of data to evaluate
# parameters
EMI_FG = ExpectationMaximizationImputation(bn, test_data)
EMI_FG.set_suff_stats(bdeu_FG.suff)
#
#
FG = FG_estimator(test_data)
updated_model = FG.update(new_data=new_data, structure_scoring_method=bdeu_FG,
                          parameter_scoring_method=EMI_FG, start_dag=bn, tabu_length=0,
                          data_per_search=35, black_list=blacklist)
# print(updated_model.cpds[0])
updated_model = FG.variable_addition_update(extra_var_data, ["M3 survival", "M6 survival"], start_dag=updated_model, structure_scoring_method=bdeu_FG,
                            parameter_scoring_method=EMI_FG, black_list=blacklist)




