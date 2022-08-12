import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
# import networkx as nx
# import netgraph # for movable nodes
from scipy.stats import shapiro
from pgmpy.estimators import BDeuScore, K2Score, BicScore, ExpectationMaximization
from EM_fixed import Fixed_EM
from pgmpy.models import BayesianModel, BayesianNetwork
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.inference import CausalInference
from FriedmanGoldszmidt import FG_estimator, SuffStatBicScore, MLE_FG, SuffStatBDeuScore
from EM_imputation import ExpectationMaximizationImputation
from Radidux_prepare import get_RADIDUX_data

def rename_values(data: pd.DataFrame):
    data["Stage"].replace([1, 2, 3, 4], ["Stage 1", "Stage 2", "Stage 3", "Stage 4"], inplace=True)
    data["Sex"].replace([0, 1], ["Female", "Male"], inplace=True)
    data["Comorbidity"].replace([1, 2], ["No", "Yes"], inplace=True)
    data["Treat. mutation"].replace(np.NaN, "Not treatable", inplace=True)
    data["Histology"].replace([0, 1, 2, 3, 4, 5], ["NSCLC Non-adeno non-squamous", "NSCLC adeno", "NSCLC adeno", "Squamous", "SCLC", np.NaN],
                             inplace=True)
    data["M0 PS"].replace([0, 1, 2], ["Unsymptomatic", "Symptomatic", ">50% care"], inplace=True)
    data["T-stage"].replace([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                            [np.NaN, "Tin situ", "T1", "T1", "T1", "T1", "T2", "T2", "T3",
                             "T4", np.NaN, "T2", "T1", "T0"], inplace=True)
    data["N-stage"].replace([1, 2, 3, 4, 5, 6], ["N0", "N1", "N2", "N3", np.NaN, np.NaN], inplace=True)
    # data["M-stage"].replace([1, 2, 3, 4, 5, 6], ["M0", "M1a", "M1b", "M1c", "M1", np.NaN], inplace=True)
    data["M-stage"].replace([1, 2, 3, 4, 5, 6], ["M0", "M1", "M1", "M1", "M1", np.NaN], inplace=True)
    data["Treatment"].replace([1, 2, 3, 4, 5, 6], ["Immuno", "Immuno+", "Chemo(+radio)",
                                                      "Radio", "Targetted", "Chirurgical"], inplace=True)
    # data["Treatment"].replace([1, 2, 3, 4, 5, 6], ["Dummy 1", "Dummy 1", "Dummy 1",
    #                                                   "Dummy 2", "Dummy 2", "Dummy 2"], inplace=True)
    # data["Prev. treatment"].replace([0, 1, 2, 3, 4, 6], ["Dummy 1", "Dummy 1", "Dummy 1",
    #                                                   "Dummy 2", "Dummy 2", "Dummy 2"], inplace=True)
    # data["Treat. response"].replace([0, 1, 2, 3, 4, 5, 6, 7], ["Dummy 1", "Dummy 1", "Dummy 1", "Dummy 1",
    #                                                   "Dummy 2", "Dummy 2", "Dummy 2", "Dummy 2"], inplace=True)
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

def disc_age(data: pd.DataFrame, bins=np.array([])):
    if bins.any():
        data["Age"] = pd.cut(data["Age"], bins=bins, labels=["Young", "Average", "Old"]).astype("object")
    else:
        data["Age"], bins = pd.qcut(data["Age"], 3, labels=["Young", "Average", "Old"], retbins=True)
        data["Age"] = data["Age"].astype("object")
    return bins

def disc_HRQOL(data: pd.DataFrame, bins=np.array([])):
    deltaM3 = data["M3 HRQOL"] - data["M0 HRQOL"]
    data["M3 HRQOL"] = np.where(deltaM3 > 10, "Declined",
             np.where(data["M3 survival"] == "No", "Death", np.where(pd.isnull(data["M3 HRQOL"]), None, "Stable")))
    if "M6 HRQOL" in data.columns:
        deltaM6 = data["M6 HRQOL"] - data["M0 HRQOL"]
        data["M6 HRQOL"] = np.where(deltaM6 > 10, "Declined",
                                    np.where(data["M6 survival"] == "No", "Death",
                                             np.where(pd.isnull(data["M6 HRQOL"]), None, "Stable")))
    if bins.any():
        data["M0 HRQOL"] = pd.cut(data["M0 HRQOL"], bins=bins, labels=["Low", "Average", "High"])
    else:
        data["M0 HRQOL"], bins = pd.qcut(data["M0 HRQOL"], 3, labels=["Low", "Average", "High"], retbins=True)
    data["M0 HRQOL"] = data["M0 HRQOL"].astype("object")
    return bins

def simplify_stadia(data: pd.DataFrame):
    data["stadium"].replace(["1A", "1B", "2A", "2B", "3A", "3B", "3C",
                                       np.nan],
                                      ["1", "1", "2", "2", "3", "3", "3", "Niet te bepalen"],
                                      inplace=True)

def inspect_exemplar_patients(model, outcomes):
    vars = model.nodes
    ci = CausalInference(model)
    patient1 = {"Age": "Young", "Sex": "Female", "Comorbidity": "No", "M-stage": "M0",
                "N-stage": "N3", "Histology": "NSCLC adeno", "M0 PS": "Unsymptomatic"}
    treatments1 = [{"Treatment": "Chemo(+radio)"}, {"Treatment": "Radio"}]
    patient2 = {"Age": "Average", "Sex": "Male", "Comorbidity": "No", "M-stage": "M0",
                "T-stage": "T4", "Histology": "Squamous", "M0 PS": "Unsymptomatic"}
    treatments2 = [{"Treatment": "Chemo(+radio)"}, {"Treatment": "Immuno"}]
    patient3 = {"Age": "Old", "Sex": "Male", "Comorbidity": "Yes", "N-stage": "N2",
                "T-stage": "T3", "Histology": "NSCLC adeno", "M0 PS": "Symptomatic"}
    treatments3 = [{"Treatment": "Chemo(+radio)"}, {"Treatment": "Immuno+"}]
    patient4 = {"Age": "Young", "Sex": "Male", "Comorbidity": "No", "N-stage": "N3",
                "T-stage": "T4", "M-stage": "M1", "Histology": "NSCLC adeno", "M0 PS": "Unsymptomatic"}
    treatments4 = [{"Treatment": "Chemo(+radio)"}, {"Treatment": "Immuno"}]
    patient5 = {"Age": "Young", "Sex": "Female", "Comorbidity": "No", "T-stage": "T1",
                "M-stage": "M1", "Histology": "NSCLC adeno", "M0 PS": "Symptomatic"}
    treatments5 = [{"Treatment": "Chemo(+radio)"}, {"Treatment": "Chirurgical"}]
    patients = [patient1, patient2, patient3, patient4, patient5]
    treatments = [treatments1, treatments2, treatments3, treatments4, treatments5]
    couples = list(zip(patients, treatments))
    for patient, treatments in couples:
        for treatment in treatments:
            for outcome in outcomes:
                patient_rel = {key: value for key, value in patient.items() if key in model.nodes}
                print(f"for patient: {patient}")
                print(f"for treatment: {treatment}")
                print(f"predict: {outcome}")
                if outcome == "M3 HRQOL":
                    print(ci.query(variables=[outcome], do=treatment, evidence=patient_rel.update({"M3 survival": "Yes"})))
                elif outcome == "M6 HRQOL":
                    print(ci.query(variables=[outcome], do=treatment, evidence=patient_rel.update({"M6 survival": "Yes"})))
                else:
                    print(ci.query(variables=[outcome], do=treatment, evidence=patient_rel))

# Load in data
names = ["ID", "Institute", "Intervention", "Stage", "Age", "Sex", "Comorbidity",
         "Group", "Active", "M0 PS", "T-stage", "N-stage", "M-stage", "Histology",
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

# Preprocess data
rename_values(data)
data=data[data["Histology"] != "SCLC"]
data=data[(data["Stage"] == "Stage 3") | (data["Stage"] == "Stage 4")]
data=data[data["T-stage"] != "T0"]
data=data[(data["Treat. response"] != "Other") | (data["Treat. response"] != "Recurrence")]
calc_survival(data)
char_var = ["Age", "Sex"]
tumor_var = ["T-stage", "N-stage", "M-stage", "Histology"]
treatment_var = ["Treatment"]
treatment_effect_var = ["Treat. response"]
outcome_var = ["M0 PS", "M0 HRQOL", "M3 HRQOL", "M3 survival", "M6 survival"]
bins_age = disc_age(data)
bins_HRQOL = disc_HRQOL(data)
Sympro_data = pd.DataFrame(data, columns=char_var+tumor_var+treatment_var+treatment_effect_var+outcome_var)
train = Sympro_data

# Get revision data
Radidux_data = get_RADIDUX_data()
disc_age(Radidux_data, bins_age)
disc_HRQOL(Radidux_data, bins_HRQOL)
Radidux_data = Radidux_data.sample(frac=1).reset_index(drop=True)
added_vars = ["M6 HRQOL", "Comorbidity"]
scenario1 = Radidux_data.drop(added_vars, axis=1)
scenario2 = Radidux_data

#Load blacklist and whitelist
blacklist = pd.read_csv("BlacklistCSV.csv", sep=";")
blacklist = list(zip(blacklist["From"], blacklist["To"]))
scen2_whitelist = pd.read_csv("WhitelistCSV.csv", sep=";")
scen2_whitelist = list(zip(scen2_whitelist["From"], scen2_whitelist["To"]))
whitelist = [x for x in scen2_whitelist if not (x[0] in added_vars or x[1] in added_vars)]
scen1_blacklist = [x for x in blacklist if not (x[0] in added_vars or x[1] in added_vars)]

#Create initial model from data
bdeu = BDeuScore(train, equivalent_sample_size=10)
hc = HillClimbSearch(train)
model = hc.estimate(scoring_method=bdeu, epsilon=0, black_list=blacklist, fixed_edges=whitelist)
bn = BayesianNetwork(model)
bn.fit(train, estimator=Fixed_EM, atol=0.001, max_iter=10)

#Create BDeu scorer that works with suff stats instead of data to evaluate structures
bdeu_FG = SuffStatBDeuScore(train)
bdeu_FG.calculate_sufficient_stats(bn, blacklist)

#Create Expectation Maximization scorer that works with suff stats instead of data to evaluate
# parameters
EMI_FG = ExpectationMaximizationImputation(bn, train)
EMI_FG.set_suff_stats(bdeu_FG.suff)

# Create FG object to run algorithms
FG = FG_estimator(train)
FG.suff = EMI_FG.suff_stats

# Category 1 revision
updated_model = FG.update(new_data=scenario1, structure_scoring_method=bdeu_FG,
                          parameter_scoring_method=EMI_FG, start_dag=bn, tabu_length=0,
                          data_per_search=25, black_list=blacklist, fixed_edges=whitelist)

# Category 2 revision
updated_model = FG.variable_addition_update(scenario2, added_vars, start_dag=bn, structure_scoring_method=bdeu_FG,
                            parameter_scoring_method=EMI_FG, black_list=blacklist, fixed_edges=scen2_whitelist)

# Perform causal inference on exemplar patients
inspect_exemplar_patients(updated_model, outcomes=["M3 survival", "M6 survival", "M3 HRQOL"])