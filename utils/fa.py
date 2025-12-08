import pandas as pd
import os, sys, importlib
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


## fa
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer import FactorAnalyzer
from pingouin import cronbach_alpha
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D




def run_check_fa(df, labels,output_folder="fa_results", 
                n_factors=5, rotation="oblimin", 
                fa_names=None,
                get_factor_scores_by='both',
                run_scores_fa_on="mean",
                get_data_heatmap=True, get_cronbachs_a=True,
                get_barplot=False, get_pca3d=False, 
                get_table=True,low_comm_t=0.5):
    
    
    # 英语提示，但是图片标题用法语！
    #output
    os.makedirs(output_folder, exist_ok=True)
    print(f"[INFO] all results saved to folder {output_folder}!\n"
          f"INITIALISE df, or MERGE go WRONG!!!")

    ###============================RUN==============================###
    print(f"============================RUN FA=================================")
    #-------------------------- check data -----------------------------
    missing_labels = [col for col in labels if col not in df.columns]
    if missing_labels:
        print(f"[WARNING] These labels are missing: {missing_labels}\n")

    else :
        df_dropna=df[df[labels].sum(axis=1)!=0]
        data=df_dropna[labels]
        print(f"[INFO] len scores in df notna:{len(data)}\n")
    
    # ---------------------------kmo-----------------------------------
    kmo_all, kmo_model = calculate_kmo(data)
    print(f"[INFO] kmo_model score (> 0.7) :{kmo_model}\n")

    # ---------------------------fa fit -------------------------------
    if rotation =="oblimin":
        print(f"[INFO] oblimin allows cross-loadings!\n")
    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method='minres')
    fa.fit(data)
    ### output impo :fa, data

    print("---------------------------heatmap of zsc-------------------------------")
    if get_data_heatmap==True: 
        loadings = fa.loadings_
        print(f"[INFO] good loadings > 0.5")
        
        if fa_names :
            loadings_df=pd.DataFrame(loadings, index=data.columns, columns=fa_names)
        else :
            loadings_df=pd.DataFrame(loadings, index=data.columns)  
    
        plt.figure(figsize=(10,8))
        sns.heatmap(loadings_df, annot=True, cmap='coolwarm')
        plt.title('Charges factorielles des items')
        # plt.xticks(rotation=45)
        plt.tight_layout()
        outpath_heatmap=os.path.join(output_folder, 'heatmap_fa.jpeg')
        plt.savefig(outpath_heatmap, dpi=300)
        plt.show()
        print(f"[SAVE] heatmap saved to {outpath_heatmap}!\n")

    print("---------------------------------fa scores ---------------------------------")
    # 从原始 df 中筛选非空行 (df_dropna) 做 FA
    # 得到 FA scores (df_scores) (无id索引)！
    # 拼接回 df_dropna → 得到 df_tactics
    # 按 "host_about" 去重 → df_bio_tactics
    # 最后再 merge 回原始 df → listings_tactics
    
    # init，空的df不会拼接上
    df_scores_mean=pd.DataFrame()
    df_scores_weighted=pd.DataFrame()
    
    if get_factor_scores_by  in ["mean","both"]:
        print(f"[INFO] calulate factor score by taking the average of its corresponding items, ref to a pre-defined map.\n")
        fa_item_map={
            "ouverture":['open to different cultures','cosmopolitan','international view','cultural exchange'],
            "authenticité":['personal life','life experiences','divers interests','hobbies','enjoy life'],
            'sociabilité':['meet new people', 'welcoming', 'friendly','sociable', 'interpersonal interaction'],
            'auto_promotion':['thoughtful service', 'attentive to needs','willing to help','responsive'],
            'exemplarité':["fan of Airbnb","Airbnb community",'love Airbnb', 'travel with Airbnb']
        }
        fa_scores_mean = pd.DataFrame(index=data.index)
        for f, items in fa_item_map.items():
            fa_scores_mean[f'{f}_mean'] = data[items].mean(axis=1)
        df_scores_mean=pd.DataFrame(fa_scores_mean).reset_index(drop=True)

    if get_factor_scores_by in ["transform","both"]:
        fa_scores_weighted = fa.transform(data)
        df_scores_weighted=pd.DataFrame(fa_scores_weighted).reset_index(drop=True)
        if fa_names:
            df_scores_weighted.columns=[f"{col}_weighted" for col in fa_names]
    
    if get_factor_scores_by not in ['mean','transform', 'both']:
        print(f"[WARNING] choose from 'mean' or 'transform'!!")

    df_scores=pd.concat([df_scores_mean, df_scores_weighted],axis=1)       



    print("------------------------heatmap of fa scores(tactcis)----------------------------")
    # init fa
    fa_scores = FactorAnalyzer(n_factors=2, rotation='oblimin', method='minres')
    
    get_heatmap_scores=False
    if run_scores_fa_on=='mean' and not df_scores_mean.empty:        
        fa_scores.fit(df_scores_mean)
        df_scores_hm=df_scores_mean
        get_heatmap_scores=True
    
    elif run_scores_fa_on=='weighted' and not df_scores_weighted.empty:
        fa_scores.fit(df_scores_weighted)
        df_scores_hm=df_scores_weighted

        get_heatmap_scores=True
    else :
        get_heatmap_scores=False
        
    if get_heatmap_scores==True:
        loadings_scores = fa_scores.loadings_
        loadings_scores_df=pd.DataFrame(loadings_scores, index=df_scores_hm.columns)
        sns.heatmap(loadings_scores_df, annot=True, cmap='coolwarm')
        plt.title(f'Charge factorielles des tactiques (scores {run_scores_fa_on})')
        plt.tight_layout()
        scores_mean_heatmap_outpath=os.path.join(output_folder,f'heatmap_fa_scores_{run_scores_fa_on}.jpg')
        plt.savefig(scores_mean_heatmap_outpath,dpi=300)
        plt.show()
        print(f"[SAVE] heatmap of mean scores saved to {scores_mean_heatmap_outpath}!\n")



    print("----------------------------merge data to listings---------------------------------")
    df_dropna = df_dropna.reset_index(drop=True)
    df_scores=df_scores.reset_index(drop=True)
    df_tactics=pd.concat([df_dropna, df_scores], axis=1)# horizontal
    
    df_bio_tactics=df_tactics.drop_duplicates(subset="host_about")
    cols_tosave=["host_about"]+df_scores.columns.tolist()
    df_bio_tactics=df_bio_tactics[cols_tosave]
    listings_tactics=df.merge(df_bio_tactics, left_on='host_about', right_on='host_about', how="left")
    
    outpath_listings_tactics=os.path.join(output_folder, 'lisitngs_tactics.csv')
    listings_tactics.to_csv(outpath_listings_tactics, index=False)
    display(listings_tactics.head())
    print(f"[INFO] scores shape (len(data), 1/2*n_factors) :{df_scores.shape}\n"
            f"len(df_dropna) should == len(df_scores)==len(df_tactics): {len(df_dropna)}, {len(df_scores)}, {len(df_tactics)}\n"
            f"len(df)==len(listings_tactics):{len(df)},{len(listings_tactics)}\n")
    print(f"[SAVE] listings with tactics scores by '{get_factor_scores_by}'way saved to {outpath_listings_tactics}!!\n")
            
        
    
    print("###====================================CHECK=======================================###")   

    if get_cronbachs_a==True:
        print("---------------------------cronbach's alpha-------------------------------")
        print(f"[INFO] good alpha > 0.7\n")
        print(f"Cronbach's alpha checks internal consistency, whether items mesure a lantent construct !\n")
        
        # predifine mapping dict
        labels2tac={
            "openness":['open to different cultures','cosmopolitan','international view','cultural exchange'],
            "authenticity":['personal life','life experiences','divers interests','hobbies','enjoy life'],
            'sociability':['meet new people', 'welcoming', 'friendly','sociable', 'interpersonal interaction'],
            'self_promotion':['thoughtful service', 'attentive to needs','willing to help','responsive'],
            'exemplification':["fan of Airbnb","Airbnb community",'love Airbnb', 'travel with Airbnb']
        }
        dict_alpha={}
        for tactic, labels in labels2tac.items():
            alpha, _ = cronbach_alpha(data[labels])
            # print(f"- {tactic}: {alpha}")          
            dict_alpha[tactic]=alpha
        dict_alpha_df = pd.Series(dict_alpha, name='cronbach_alpha').reset_index()
        dict_alpha_df.columns = ['tactique', 'cronbach_alpha']
        display(dict_alpha_df)
        outpath_alpha=os.path.join(output_folder, "cronbachs_alpha.csv")
        dict_alpha_df.to_csv(outpath_alpha, index=False)
        print(f"[SAVE] cronbach's alpha on items saved to {outpath_alpha}!\n")


        
        
    if get_barplot==True:        
        print(f"\n-----------------------------barplot comm&uniq-------------------------------")
        communalities = fa.get_communalities()
        uniqueness = fa.get_uniquenesses()
        print(f"[INFO] mean COMM (> 0.6):{communalities.mean()}\n"
            f"[INFO] mean UNIQ : {uniqueness.mean()}\n")
        
        
        comm_df = pd.DataFrame({
            'Communality': communalities,
            'Uniqueness': uniqueness
        }, index=data.columns)
        comm_df = comm_df.sort_values("Communality")

        plt.figure(figsize=(10,6))
        colors = sns.color_palette("viridis", n_colors=2)  # 两个指标

        comm_df.plot(kind='bar', stacked=False, color=colors)# edgecolor='black'
        plt.title("Communalités et spécificités des items", fontsize=16)
        plt.ylabel("Proportion de la Variance Expliquée")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Metric')
        plt.tight_layout()
        outpath_barplot=os.path.join(output_folder,"comm_uniq_barplot.jpg")
        plt.savefig(outpath_barplot, dpi=300)
        plt.show()
        print(f"[SAVE] barplot saved to {outpath_barplot}!\n")

        # # 画图
        # # plt.figure(figsize=(10, 6))
        # comm_df.sort_values("Communality").plot(kind='bar', figsize=(12,6))
        # plt.title("Communality & Uniqueness of Items")
        # plt.xticks(rotation=45, ha='right')
        # plt.tight_layout()
        # # plt.savefig('../figs/communalities_uniqueness.jpeg', dpi=300)
        # plt.show()
    
    if get_pca3d==True:
        print("# -----------------------------3D PCA-------------------------------")
        pca_3d=PCA(n_components=3)
        loadings_3d=pca_3d.fit_transform(loadings)
        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111, projection='3d')
        
        # 缩放向量
        norms = np.linalg.norm(loadings_3d, axis=1)
        loadings_3d_scaled = loadings_3d / norms[:, np.newaxis] #0.8
        print("[INFO] Explained variance ratios 3D (>0.6):", pca_3d.explained_variance_ratio_.sum())

        # 画箭头
        for i, var in enumerate(data.columns):
            ax.quiver(0, 0, 0,
                    loadings_3d_scaled[i, 0],
                    loadings_3d_scaled[i, 1],
                    loadings_3d_scaled[i, 2],
                    color='red', arrow_length_ratio=0.1)

            ax.text(loadings_3d_scaled[i, 0]*1.1,
                    loadings_3d_scaled[i, 1]*1.1,
                    loadings_3d_scaled[i, 2]*1.1,
                    var, fontsize=10)

        # 坐标轴限制
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

        plt.title('3D PCA Correlation Circle (Scaled)')
        plt.tight_layout()
        outpath_pca=os.path.join(output_folder, 'pca_3d.jpg')
        plt.savefig(outpath_pca,dpi=300)
        plt.show()
        print(f"[SAVE] PCA 3D saved to {outpath_pca}!\n")
        
        

    if get_table==True:
        print("# ------------------------------------tab------------------------------------")
        loadings = fa.loadings_
        communalities = fa.get_communalities()
        uniqueness = fa.get_uniquenesses()

        # gamma 和 sigma
        abs_loadings = np.abs(loadings)
        gamma = abs_loadings.max(axis=1)                   # 最大因子载荷
        sigma = np.sort(abs_loadings, axis=1)[:, -2]       # 第二大因子载荷

        # get DataFrame
        desired_order = None  # 先定义，保证一定存在
        if fa_names:
            fa_names=fa_names
            desired_order=['gamma','sigma','ouverture','authenticité','sociabilité',"auto_promotion","exemplarité",'communalité','spécificité']
            
        else :
            fa_names=[f"Factor{i+1}" for i in range(0, loadings.shape[1])]
            
        columns = ['gamma', 'sigma'] + \
                fa_names + \
                ["communalité","spécificité"]
                # ['communality', 'uniqueness']

        df_fa = pd.DataFrame(
            np.column_stack([gamma, sigma, loadings, communalities, uniqueness]),
            index=data.columns,
            columns=columns
        )
        df_fa.round(3)
        
        # reorder 
        # desired_order=['gamma','sigma','ouverture','authenticité','sociabilité',"auto_promotion","exemplarité",'communalité','spécificité']
        if desired_order:
            df_fa=df_fa[desired_order]
            
        display(df_fa)
        
        #info
        print(f"[INFO] mean COMM (> 0.6):{communalities.mean()}\n"
                    f"[INFO] mean UNIQ : {uniqueness.mean()}\n")
        low_comm_t=0.5
        low_comm_items = df_fa[df_fa['communalité'] <low_comm_t ]
        print(f"[WARNING] {len(low_comm_items)}/{len(df_fa)} low communality items (< {low_comm_t}):\n{';'.join(low_comm_items.index)}\n")

        # save
        outpath_tab=os.path.join(output_folder, 'table_fa.csv')
        df_fa.to_csv(outpath_tab, index=False)
        print(f"[SAVE] table saved to {outpath_tab}!\n")
        
        # -----------------------------tab2latex-------------------------------                
        latex_code = df_fa.round(3).to_latex(
            caption="Tableau de l'analyse factorielle",
            label="tab:fa_table",
            index=True,        # 是否保留行索引（item 名称）
            escape=False       # False 可以保留 LaTeX 特殊字符，比如 _ 等
        )
        outpath_tab_latex=os.path.join(output_folder, 'table_fa_latex.tex')
        with open(outpath_tab_latex, 'w') as f:
            f.write(latex_code)
        print(f"[SAVE] table latex saved to {outpath_tab_latex}!\n")   
        print(f"[WARNING] remember to rename factors by heatmap of zsc data!!!")            
      
        
        
    return 