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
# from utils.preprocess_listings import save_csv_as_latex


def check_vif (df, x_vars, y_vars):
    # y not in x:
    if y_vars in x_vars:
        x_vars.remove(y_vars)
    
        
    # cat/num
    cat_vars = [var for var in x_vars if not pd.api.types.is_numeric_dtype(df[var])]
    num_vars = [var for var in x_vars if pd.api.types.is_numeric_dtype(df[var])]

    print(f"[INFO] categorial vars: {cat_vars};\n"
            f"numeric vars :{num_vars}\n")

    cat_vars_str = '+'.join([f"C({c})" for c in cat_vars])
    num_vars_str = '+'.join(num_vars)

    formula = f"({y_vars} ~ {cat_vars_str} + {num_vars_str})"
    print(f"[INFO] formula : {formula}\n")
    
    # formula=('booking_rate_l30d ~'
    #         'number_of_reviews + review_scores_rating +' 
    #         ' years_since_host + C(lang) + len + C(professional_host) + C(host_is_superhost)+'
    #         'price + C(room_type) + minimum_nights + C(instant_bookable)+'
    #         'ouverture_mean + authenticité_mean + sociabilité_mean + auto_promotion_mean + exemplarité_mean'
    # )


    y, X = dmatrices(formula, data=df, return_type='dataframe')
    vif_df = pd.DataFrame()
    vif_df['Variables']=X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_df['niveau_colinearite'] = vif_df["VIF"].apply(
        lambda x: " " if x <= 1 else
                "*" if x <= 5 else
                "**" if x <= 10 else
                "***"
    )
    display(vif_df)  
    
    return


def model (df, vars, scores_fillna=False,
           outpath_folder='mod_results'):
    os.makedirs(outpath_folder, exist_ok=True)
    formula=''
    
    model_basic=smf.ols('booking_rate_l30d ~'
            'C(host_identity_verified)+ C(host_has_profile_pic) +' 
            'number_of_reviews + review_scores_rating + C(has_rating) +' 
            ' years_since_host + C(has_host_about)+ C(lang) + len +'
            'C (host_response_time) + host_response_rate + C(has_response_rate) + C(professional_host) +'
            'price + C(room_type) + minimum_nights + C(instant_bookable)'
            , data=df).fit()
    
    
    return 


