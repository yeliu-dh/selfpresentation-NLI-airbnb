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

def write_formula(df_input, x_vars, y_var, tactics_vars=None, group_col=None):
    df=df_input.copy()
    
    # y not in x:
    if y_var in x_vars:
        x_vars.remove(y_var)
    if group_col !=None and group_col in x_vars:
        x_vars.remove(group_col)
        
        
    # cat/num
    cat_vars = [var for var in x_vars if not pd.api.types.is_numeric_dtype(df[var])]
    num_vars = [var for var in x_vars if pd.api.types.is_numeric_dtype(df[var])]

    print(f"[INFO] {len(cat_vars)} categorial vars: {cat_vars};\n"
            f"{len(num_vars)} numeric vars :{num_vars}\n")


    cat_vars_str = '+'.join([f"C({c})" for c in cat_vars])
    num_vars_str = '+'.join(num_vars)
    tactics_vars_str=" " + '+'.join(tactics_vars)#避免无tactics输入
    
    if group_col !=None and tactics_vars.strip():
        tactics_vars_str=f"C({group_col})* ({tactics_vars_str})"
    formula = f"({y_var} ~ {cat_vars_str} + {num_vars_str}+ {tactics_vars_str})"
    

    print(f"[INFO] formula : {formula}\n")
    
    return formula 

def check_vif (df_input, x_vars, y_var, tactics_vars, group_col):
    df=df_input.copy()
    
    formula=write_formula(df_input, x_vars, y_var, tactics_vars=tactics_vars, group_col=group_col)

    
    try :
        y, X = dmatrices(formula, data=df, return_type='dataframe')
    except Exception as e :
        print(f"[ERROR] in check_vif : \n{e}")
        
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



def save_summary_as_latex(summary, output_folder, tex_filename):
    latex_str =summary.as_latex()
    outpath_summary=os.path.join(output_folder, tex_filename)

    with open(outpath_summary, "w") as f:
        f.write(latex_str)
        print(f"[SAVE] model summary saved to {outpath_summary}!")
    return 


def build_model (df_input, x_vars, tactics_vars, to_fillna0=False,
                y_var='booking_rate_l30d', group_col=None, 
                outpath_folder='mod_results', 
                tex_filename='ols_summary.tex'):
    df=df_input.copy()
    os.makedirs(outpath_folder, exist_ok=True)
    
    print(f"[CHECK] isna False in all cols : {df[x_vars].isna().value_counts(dropna=False)}\n")
    if to_fillna0==True:
        df[x_vars]=df[x_vars].fillna(0)

    # if group_col!=None:
        
        
    formula=write_formula(df, x_vars, y_var)    
    model_basic=smf.ols(formula, data=df).fit()
    summary=model_basic.summary()
    print(summary)
    save_summary_as_latex(summary, output_folder=outpath_folder, 
                          tex_filename=tex_filename)
    
    return 

