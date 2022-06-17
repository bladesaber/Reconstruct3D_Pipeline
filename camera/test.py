import pandas as pd
import numpy as np

a = np.array([1,2,3,4,5,5,5,1,1,2])
b = pd.DataFrame(a, columns=['a'])
print(b['a'].value_counts())
print(b.value_counts())