import pandas as pd
from timedisagg.td import TempDisagg
from typing_extensions import final

expected_dataset = pd.read_csv("./sample_data.csv")
print(expected_dataset)

# %%
#import pdb; pdb.set_trace()

import pandas as pd
import matplotlib.pyplot as plt
from timedisagg.td import TempDisagg

expected_dataset = pd.read_csv("./sample_data.csv")
#import pdb; pdb.set_trace()
td_obj = TempDisagg(conversion="sum", method="chow-lin-maxlog")
final_disaggregated_output = td_obj(expected_dataset)
print(final_disaggregated_output.head())
#print(final_disaggregated_output)
# %%
#plt.plot(expected_dataset["X"], expected_dataset["y"], ".:")
plt.plot(final_disaggregated_output["y"], final_disaggregated_output["y_hat"], ".:")


# %%
import timedisagg
timedisagg.__file__

# %%
