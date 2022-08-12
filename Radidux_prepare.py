import pandas as pd
import numpy as np
import datetime
import optbinning as bin
from entropymdlp.mdlp import MDLP

def rename_values(data: pd.DataFrame):
    data["Sex"].replace([0, 1], ["Male", "Female"], inplace=True)
    data["Comorbidity"].replace([0, 1], ["No", "Yes"], inplace=True)
    data["Histology"].replace([1, 2, 3, 8], ["NSCLC adeno", "NSCLC Non-adeno non-squamous", "Squamous", np.NaN],
                             inplace=True)
    data["M0 PS"].replace([0, 1, 2], ["Unsymptomatic", "Symptomatic", ">50% care"], inplace=True)
    data["T-stage"].replace([1, 2, 3, 4, 9],
                            ["T1", "T2", "T3", "T4", np.NaN], inplace=True)
    data["N-stage"].replace([0, 1, 2, 3, 9], ["N0", "N1", "N2", "N3", np.NaN], inplace=True)
    data["M-stage"].replace([0, 9], ["M0", "M0"], inplace=True)
    data["Stage"].replace([1, 2, 3, 4], ["Stage 1", "Stage 2", "Stage 3", "Stage 4"], inplace=True)
    data["M3 survival"].replace([0, 1], ["No", "Yes"], inplace=True)
    data["M6 survival"].replace([0, 1], ["No", "Yes"], inplace=True)
    data["M9 survival"].replace([0, 1], ["No", "Yes"], inplace=True)
    data["M12 survival"].replace([0, 1], ["No", "Yes"], inplace=True)
    data["Treatment"] = "Chemo(+radio)"
    # data["Treatment"] = "Dummy 1"
    data["Treat. response"] = np.NaN

def get_RADIDUX_data():
    names = ["ID", "Sex", "os1", "os2", "M3 survival", "M6 survival", "M9 survival",
             "M12 survival", "Age", "T-stage", "N-stage", "M-stage", "Stage",
             "Comorbidity", "M0 PS", "Histology", "Histology_unknown", "?4",
             "?5", "?6", "EFGR score", "M0 HRQOL", "M3 HRQOL", "M6 HRQOL", "M9 HRQOL"]
    dtypes = ["float64" for x in names]
    dtypes = dict(zip(names, dtypes))
    data = pd.read_csv("radidux.csv", sep=";", header=0,
                       names=names, dtype=dtypes)
    rename_values(data)
    char_var = ["Age", "Sex", "Comorbidity"]
    # tumor_var = ["T-stage", "N-stage", "M-stage", "Histology", "Treat. mutation"]
    tumor_var = ["T-stage", "N-stage", "M-stage", "Histology"]
    # treatment_var = ["Treatment", "Prev. treatment"]
    treatment_var = ["Treatment"]
    treatment_effect_var = ["Treat. response"]
    outcome_var = ["M0 PS", "M0 HRQOL", "M3 HRQOL", "M6 HRQOL", "M3 survival", "M6 survival"]
    Radidux_data = pd.DataFrame(data, columns=char_var + tumor_var + treatment_var + treatment_effect_var + outcome_var)
    return Radidux_data

def disc_HRQOL(data: pd.DataFrame, delta=True):
    if delta:
        data["delta HRQOL"] = data["M3 HRQOL"] - data["M0 HRQOL"]
        data["M3 HRQOL"] = np.where(data["delta HRQOL"] > 10, "Declined",
                 np.where(data["M3 survival"]==False, "Death", np.where(pd.isnull(data["M3 HRQOL"]), np.NaN, "Stable")))
        data["M0 HRQOL"] = pd.qcut(data["M0 HRQOL"], 3, labels=["Low", "Average", "High"], retbins=False).astype("object")
    else:
        data["M0 HRQOL"], bins = pd.qcut(data["M0 HRQOL"], 3, labels=["Low", "Average", "High"], retbins=True)
        data["M0 HRQOL"] = data["M0 HRQOL"].astype("object")
        data["M3 HRQOL"] = pd.qcut(data["M3 HRQOL"], 3, labels=["Low", "Average", "High"]).astype("object")
        # data["M3 HRQOL"] = pd.cut(data["M3 HRQOL"], bins=bins, labels=["Low", "Average", "High"]).astype("object")

# def disc_age(data:pd.DataFrame, bins):
    # bin_data = pd.DataFrame(data, columns=["M0 HRQOL", "Age", "M6 survival"])
    # bin_data = bin_data.dropna()
    # bin_data = bin_data.reset_index(drop=True)
    # bin_data["M0 HRQOL"].replace(["Low", "Average", "High"], [0, 1, 2], inplace=True)
    # binner = bin.MDLP()
    # binner.fit(bin_data["Age"], bin_data["M0 HRQOL"])
    # mdlp = MDLP()
    # bins = mdlp.cut_points(x= bin_data["Age"].values, y=bin_data["M6 survival"].values)