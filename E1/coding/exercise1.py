import pandas as pd

df = pd.read_csv("StudentPerformance.shuf.test.csv", sep =";", lineterminator="\n", encoding="utf-8",error_bad_lines=False)
