import numpy as np
import pandas as pd
import datetime
# import matplotlib.pyplot as plt
# import networkx as nx
# import netgraph # for movable nodes
from scipy.stats import shapiro
from pgmpy.estimators import BDeuScore, K2Score, BicScore, ExpectationMaximization
from pgmpy.models import BayesianModel, BayesianNetwork
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator
from FriedmanGoldszmidt import FG_estimator, SuffStatBicScore, MLE_FG
from EM_imputation import ExpectationMaximizationImputation

def rename_values(data: pd.DataFrame):
    data["Stage"].replace([1, 2, 3, 4], ["Stage 1", "Stage 2", "Stage 3", "Stage 4"], inplace=True)
    data["Gender"].replace([0, 1], ["Female", "Male"], inplace=True)
    data["Comorbidity"].replace([1, 2], ["No", "Yes"], inplace=True)
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
    data["Previous treatment"].replace([0, 1, 2, 3, 4, 5], ["No", "Chirurgical", "Chemoradio",
                                                      "Radio", "Systemic (chemo)", "Other"], inplace=True)
    data["Treatment response"].replace([0, 1, 2, 3, 4, 5, 6, 7], ["Curative", "Complete", "Partial", "Stabile",
                                                                  "Progressive", "Recurrence", "Toxic", "Other"],
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
    data["Age"] = pd.qcut(data["Age"], 3, labels=["Young", "Average", "Old"])

def simplify_stadia(data: pd.DataFrame):
    data["stadium"].replace(["1A", "1B", "2A", "2B", "3A", "3B", "3C",
                                       np.nan],
                                      ["1", "1", "2", "2", "3", "3", "3", "Niet te bepalen"],
                                      inplace=True)

names = ["ID", "Institute", "Intervention", "Stage", "Age", "Gender", "Comorbidity",
         "Group", "Active", "M0 Ecog", "T-stage", "N-stage", "M-stage", "Histology",
         "Mutation", "Start date", "Treatment", "Previous treatment", "M0 HRQOL",
         "M3 HRQOL", "Treatment response", "Completion", "Dropout reason", "Date of death"]
dtypes = ["object", "object", "float64", "float64", "float64", "float64", "float64", "float64", "float64",
          "float64", "float64", "float64", "float64", "float64", "object", "object", "float64", "float64",
          "float64", "float64", "float64", "float64", "float64", "object"]
parse_dates = ["Start date", "Date of death"]
dtypes = dict(zip(names, dtypes))
data = pd.read_csv("220303_request_IRIS.csv", sep=";", header=0,
                   names=names, dtype=dtypes, parse_dates=parse_dates,
                   infer_datetime_format=True, dayfirst=True)
rename_values(data)
calc_survival(data)
char_var = ["Age", "Gender", "Comorbidity"]
tumor_var = ["T-stage", "N-stage", "M-stage", "Histology", "Mutation"]
treatment_var = ["Treatment", "Previous treatment"]
treatment_effect_var = ["Treatment response"]
outcome_var = ["M0 Ecog", "M0 HRQOL", "M3 HRQOL", "M3 survival", "M6 survival"]
data = pd.DataFrame(data, columns=char_var+tumor_var+treatment_var+treatment_effect_var+outcome_var)
disc_age(data)

#
# data = pd.read_csv("asia10K.csv")
# data = pd.DataFrame(data, columns=["Smoker", "LungCancer", "X-ray"])
# test_data = data[:2000]
# new_data = data[2000:]
# new_data[:500]["Smoker"] = np.NaN

test_data = data[:400]


#Create initial model from data
bic = BicScore(test_data)
hc = HillClimbSearch(test_data)
model = hc.estimate(scoring_method=bic)
bn = BayesianNetwork(model)
bn.fit(test_data, estimator=MaximumLikelihoodEstimator)

#Create Bic scorer that works with suff stats instead of data to evaluate structures
bic_FG = SuffStatBicScore(test_data)
bic_FG.calculate_sufficient_stats(bn)
bic_FG.score(bn)

#Create Expectation Maximization scorer that works with suff stats instead of data to evaluate
# parameters
# EMI_FG = ExpectationMaximizationImputation(bn, test_data)
# EMI_FG.set_suff_stats(bic_FG.suff)
#
#
# FG = FG_estimator(test_data)
# updated_model = FG.update(new_data=new_data, structure_scoring_method=bic_FG,
#                           parameter_scoring_method=EMI_FG, start_dag=bn, tabu_length=0,
#                           data_per_search=100)

# BN = BayesianNetwork(model)
# BN.fit(test_data, estimator=MaximumLikelihoodEstimator)
# mle_FG = MLE_FG(bic_FG.suff, BN, test_data)
# param = mle_FG.get_parameters()
# BN.add_cpds(*param)
# EM = ExpectationMaximization

# new_data["Smoker"][:500] = np.NaN
# new_data["X-ray"][400:600] = np.NaN
#
# bn = BayesianNetwork(model)
# bn.fit(new_data, estimator=ExpectationMaximizationImputation, complete_samples_only=False)
# print("end")



