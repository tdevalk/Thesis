import pandas as pd
import numpy as np

names = ["ID", "Gender", "?1", "?2", "M3 survival", "M6 survival", "M9 survival",
         "M12 survival", "Age", "T-stage", "N-stage", "M-stage", "Stage",
         "Comorbidity", "?3", "Histology type", "Histology", "?4",
         "?5", "?6", "EFGR score", "M0 HRQOL", "M3 HRQOL", "M6 HRQOL", "M9 HRQOL"]
dtypes = ["float64" for x in names]
dtypes = dict(zip(names, dtypes))
data = pd.read_csv("radidux.csv", sep=";", header=0,
                   names=names, dtype=dtypes)