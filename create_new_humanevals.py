import pandas as pd

human_eval = pd.read_csv('humaneval.csv').head(10)
human_eval.to_csv('original.csv')
human_eval.to_csv('control.csv')
human_eval.to_csv('print.csv')