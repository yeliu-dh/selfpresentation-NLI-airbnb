## vif + model+ plot + latex

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# vif
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices


# ols mod
import statsmodels.api as sm
import statsmodels.formula.api as smf

# my utils

