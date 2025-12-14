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
from utils.io import save_csv_as_latex



def write_formula(df, x_vars, y_var, key_vars=None, group_col=None):
    # y not in x:
    if y_var in x_vars:
        x_vars.remove(y_var)
    if group_col !=None and group_col in x_vars:
        x_vars.remove(group_col)
        
    # cat/num
    cat_vars = [var for var in x_vars if not pd.api.types.is_numeric_dtype(df[var])]
    num_vars = [var for var in x_vars if pd.api.types.is_numeric_dtype(df[var])]
    # print(f"[INFO] {len(cat_vars)} categorial vars: {cat_vars};\n"
    #         f"{len(num_vars)} numeric vars :{num_vars}\n")

    cat_vars_str = ' + '.join([f"C({c})" for c in cat_vars])
    num_vars_str = ' + '.join(num_vars)

    formula = f"{y_var} ~ {cat_vars_str.strip()} + {num_vars_str.strip()}"

    # init    
    key_vars_str=" "
    if key_vars != None:
        key_vars_str=' + '.join(key_vars)#避免无tactics输入
    
    if group_col !=None and key_vars_str.strip():
        # print(f'Interaction item : {group_col}!')
        # print(f"BEFOR:{key_vars_str}")
        if not pd.api.types.is_numeric_dtype(df[group_col]):# group_col cat
            key_vars_str=f"C({group_col}) * ({key_vars_str})"
        else : 
            key_vars_str=f"{group_col} * ({key_vars_str})"
        # print(f"AFTER:{key_vars_str}\n")

    if key_vars_str.strip():#如果key_vars_str exists:
        formula +=f"+ {key_vars_str.strip()}"
        # f"({y_var} ~ {cat_vars_str.strip()} + {num_vars_str.strip()} + {key_vars_str.strip()})"
    
    print(f"[INFO] formula :\n {formula}\n")
    
    return formula 



def check_vif (df, x_vars, y_var, key_vars, group_col):

    formula=write_formula(df=df, x_vars=x_vars, y_var=y_var, key_vars=key_vars, group_col=group_col)
        
    try :
        y, X = dmatrices(formula, data=df, return_type='dataframe')
    except Exception as e :
        print(f"[ERROR] in check_vif : \n{e}")
        
    vif_df = pd.DataFrame()
    vif_df['Variables']=X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_df['Niveau_colinearite'] = vif_df["VIF"].apply(
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




def build_model (df_input, x_vars, key_vars=None, to_fillna0=False,
                y_var='booking_rate_l30d', group_col=None, 
                outpath_folder='mod_results', 
                tex_filename='ols_summary.tex', save=False):
    df=df_input.copy()
    
    os.makedirs(outpath_folder, exist_ok=True)
    
    # collect all vars :
    all_vars=x_vars+[y_var]
    if key_vars!=None:
        all_vars+=key_vars
        
    print(f"[CHECK] isna False in all {len(all_vars)} vars : {df[all_vars].isna().value_counts(dropna=False)}\n")
    
    if to_fillna0==True and key_vars!=None:
        df[key_vars]=df[key_vars].fillna(0)
        
    formula=write_formula(df=df, x_vars=x_vars, y_var=y_var, key_vars=key_vars, group_col=group_col)
    
    model=smf.ols(formula, data=df).fit()
    summary=model.summary()
    print(summary)
    if save==True:
        save_summary_as_latex(summary, output_folder=outpath_folder, 
                          tex_filename=tex_filename)
    
    return df, formula, model


def plot_key_var(df_input, x_vars, y_var, tactics_vars, 
                     tactic_fr, group_col=None, 
                     ax=None, show=True):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import statsmodels.formula.api as smf
    import re
    
    """
    直接输入formula保证变量一致，因为df、tactics、formula的名字改变，直接重新建模
    目前只使用host_is_superhost的分组！
    
    """
    df=df_input.copy()
    df[tactics_vars]=df[tactics_vars].fillna(0)
    formula=write_formula(df=df, x_vars=x_vars, y_var=y_var, key_vars=tactics_vars, group_col=group_col)

    #-----------------------处理变量名！（无法识别accent）-----------------------    
    rename_tactics_map={"ouverture_weighted":"openness",
                        "authenticité_weighted":"authenticity",
                        "sociabilité_weighted":"sociability",
                        "auto_promotion_weighted":"self_promotion",
                        "exemplarité_weighted": "exemplification"}    
    # new df : 
    df.rename(columns=rename_tactics_map, inplace=True)
    
    # new tactic 
    tactic=rename_tactics_map[tactic_fr]
    if tactic not in df.columns :
        print(f"[WARNING] {tactic} NOT in df!!!")

    # new formula :
    for var, new_var in rename_tactics_map.items():
        formula=formula.replace(var, new_var)
            
    # new model :
    model=smf.ols(formula, data=df).fit()
    
    
    
    #------------------ 取控制变量的默认值--------------------
    default_vals = {}
    for col in model.model.exog_names:
        if ':' in col or col == 'Intercept':
            continue
        var_name = col.split('[')[0].replace('C(', '').replace(')', '')
        if var_name in df.columns:
            if df[var_name].dtype == 'O' or df[var_name].nunique() < 10:
                default_vals[var_name] = df[var_name].mode()[0]
            else:
                default_vals[var_name] = df[var_name].mean()
    vars_in_formula = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', formula)
    default_vals = {k: v for k, v in default_vals.items() if k in vars_in_formula}

    # #----------------------预测线------------------------
    # tactic_range = np.linspace(df[tactic].min(), df[tactic].max(), 100)
    # groups = ["f","t"]  # non superhost / superhost;若是其他变量则需要改变取值
    
    # rows = []
    # for g in groups:
    #     for val in tactic_range:
    #         row = default_vals.copy()
    #         row[tactic] = val
    #         row[group_col] = g
    #         rows.append(row)

    # predict_df = pd.DataFrame(rows)

    # for col in predict_df.columns:
    #     if col in df.columns and df[col].dtype == 'O':
    #         predict_df[col] = predict_df[col].astype(df[col].dtype)

    # predict_df['predicted_booking_rate'] = model.predict(predict_df)
    # # y 直接取自formula！！
    
    # preds = model.get_prediction(predict_df)
    # pred_summary = preds.summary_frame(alpha=0.05)
    # predict_df['ci_lower'] = pred_summary['mean_ci_lower']
    # predict_df['ci_upper'] = pred_summary['mean_ci_upper']

    # #----------------------plot-----------------------------
    # if ax is None:
    #     fig, ax = plt.subplots(figsize=(6, 5))
    # else:
    #     fig = ax.figure

    # for g in groups:
    #     group_df = predict_df[predict_df[group_col] == g]
    #     label = f'Superhôtes' if g == "t" else f'Autres'
    #     # line
    #     ax.plot(group_df[tactic], group_df['predicted_booking_rate'], label=label)
    #     # confiance interval
    #     ax.fill_between(group_df[tactic], group_df['ci_lower'], group_df['ci_upper'], alpha=0.2)



    #----------------------预测线------------------------ 
    tactic_range = np.linspace(df[tactic].min(), df[tactic].max(), 100)

    rows = []

    # 情况 A：有分组变量（如 superhost）
    if group_col is not None:
        groups = df[group_col].unique()

        for g in groups:
            for val in tactic_range:
                row = default_vals.copy()
                row[tactic] = val
                row[group_col] = g
                rows.append(row)

    # 情况 B：无分组变量，只画一条线
    else:
        for val in tactic_range:
            row = default_vals.copy()
            row[tactic] = val
            rows.append(row)

    predict_df = pd.DataFrame(rows)

    # 类型对齐（非常重要）
    for col in predict_df.columns:
        if col in df.columns:
            predict_df[col] = predict_df[col].astype(df[col].dtype)

    # 模型预测
    predict_df['predicted_booking_rate'] = model.predict(predict_df)
    preds = model.get_prediction(predict_df)
    pred_summary = preds.summary_frame(alpha=0.05)

    predict_df['ci_lower'] = pred_summary['mean_ci_lower']
    predict_df['ci_upper'] = pred_summary['mean_ci_upper']


    #----------------------plot-----------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    # 情况 A：有分组，多条线
    if group_col is not None:
        for g in groups:
            group_df = predict_df[predict_df[group_col] == g]
            label = str(g)
            if group_col=="host_is_superhost":
                label= f'Superhôtes' if g == "t" else f'Autres'

            ax.plot(group_df[tactic], group_df['predicted_booking_rate'],
                    label=label)
            ax.fill_between(group_df[tactic],
                            group_df['ci_lower'], group_df['ci_upper'],
                            alpha=0.2)

        ax.legend()


    # 情况 B：无分组，单条线
    else:
        ax.plot(predict_df[tactic], predict_df['predicted_booking_rate'], linewidth=2)
        ax.fill_between(
            predict_df[tactic],
            predict_df['ci_lower'],
            predict_df['ci_upper'],
            alpha=0.2
        )
        
    
    #------------------- sig-----------------------
    if group_col!=None:
        term = f"C({group_col})[T.t]:{tactic}"
    else : 
        term = tactic    
    pval = model.pvalues.get(term, None)
    if pval is not None:
        # 根据显著性等级选择标注
        if pval < 0.001:
            sig = '***'
        elif pval < 0.01:
            sig = '**'
        elif pval < 0.05:
            sig = '*'
        else:
            sig = ""

    tactic_fr_map={"ouverture_weighted":"ouverture",
            "authenticité_weighted":"authenticité",
            "sociabilité_weighted":"sociabilité",
            "auto_promotion_weighted":"auto_promotion",
            "exemplarité_weighted": "exemplarité"}    

    ax.set_xlabel(f"{tactic_fr_map.get(tactic_fr,None)}")
    ax.set_ylabel('Taux de réservation prédit')
    ax.set_title(f"Figure d'interaction :{tactic_fr_map.get(tactic_fr,None)} × Superhôte {sig}")
    ax.legend()
    
    if show and ax is None:
        plt.show()

    return fig, ax



def layout_plots(df_input, x_vars, y_var, tactics_vars, 
                 group_col=None,
                 output_folder=None, filename=None):
    import os
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    
    # --------------------layout-------------------
    # 使用更细的列分辨率（2 行 × 6 列），方便把第二行两个图放在中间列
    fig = plt.figure(figsize=(20,9))
    # gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.3, wspace=0.5)
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.4, wspace=1)
    
    axes = []
    
    # 第一行三个：分别占两列宽 (0:2, 2:4, 4:6)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])
   
    axes.extend([ax1, ax2, ax3])

    # 第二行两个：让它们居中，分别占中间的两列 (1:3 和 3:5)
    ax4 = fig.add_subplot(gs[1, 1:3])
    ax5 = fig.add_subplot(gs[1, 3:5])
    # ax4 = fig.add_subplot(gs[1, 0:1])  # 左图
    # ax5 = fig.add_subplot(gs[1, 2:3])  # 右图
    axes.extend([ax4, ax5])

    # 确保 tactics_vars 长度为 5（或小于等于 len(axes)）
    n_to_plot = min(len(tactics_vars), len(axes))

    #-------------------plot----------------------------
    # 注意：不要在这里重新定义 ax（不要写 ax = axes[i]）
    for ax, tactic_fr in zip(axes[:n_to_plot], tactics_vars[:n_to_plot]):
        # 传入 ax，且在这里不显示单图（plot_interaction 内 show=False）
        plot_key_var(
            df_input=df_input,
            x_vars=x_vars,
            y_var=y_var,
            tactics_vars=tactics_vars,
            tactic_fr=tactic_fr,
            group_col=group_col,
            ax=ax,
            show=False
        )

    # 如果 tactics < 5，删除多余 axes（可选）
    if len(tactics_vars) < len(axes):
        for extra_ax in axes[len(tactics_vars):]:
            fig.delaxes(extra_ax)
    # plt.tight_layout()

    # save
    os.makedirs(output_folder, exist_ok=True)
    if filename is None:
        if group_col!=None:
            filename = "tactics_interaction_plots.jpg"
        else : 
            filename="tactics_plots.jpg"    
    outpath_plots = os.path.join(output_folder, filename)
    # plt.title('')?
    plt.savefig(outpath_plots, dpi=300, bbox_inches='tight')
    
    print(f"[SAVE] plots saved to {outpath_plots}!")
    
    # show
    plt.show()
    
    return fig, axes

