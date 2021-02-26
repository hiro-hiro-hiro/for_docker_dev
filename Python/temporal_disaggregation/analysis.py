import pandas as pd
from timedisagg.td import TempDisagg

expected_dataset = pd.read_csv("sample_data.csv")

td_obj = TempDisagg(conversion="sum", method="chow-lin-maxlog")
final_disaggregated_output = td_obj(expected_dataset)

print(final_disaggregated_output.head())
