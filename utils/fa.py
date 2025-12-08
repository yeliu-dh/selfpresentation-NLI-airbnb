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




def run_check_fa(df, labels, n_factors=5, rotation="oblimin", get_factor_scores_by='mean',
                output_folder="fa_results", 
                get_heatmap=True, get_cronbachs_a=True,
                get_barplot=False, get_pca3d=False, 
                get_table=True,):
    
    
    # 英语提示，但是图片标题用法语！
    #output
    os.makedirs(output_folder, exist_ok=True)

    ###============================RUN==============================###

    #-------------------------- check data -----------------------------
    if not labels in df.columns:
        print(f"[WARNING] labels not in df.columns!!\n")
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


    # ---------------------------fa scores -------------------------------
    if get_factor_scores_by=="mean":
        print()
              


    elif get_factor_scores_by=="transform":
        scores_fa = fa.transform(data)
        print(f"[INFO] scores shape (len(data),n_factors) :{scores_fa.shape}\n"
              f"scores got by transforme are")
    
        
    else :
        print(f"[WARNING] choose from 'mean' or 'transform'!!")

        
        
        
    
    
    
    
    ###============================CHECK==============================###
    
    # ---------------------------heatmap-------------------------------
    if get_heatmap==True:    
        loadings = fa.loadings_
        print(f"[INFO] good loadings > 0.5")
        
        plt.figure(figsize=(10,8))
        loadings_df=pd.DataFrame(loadings, index=data.columns)
        sns.heatmap(loadings_df, annot=True, cmap='coolwarm')
        plt.title('Factors loadings des items')
        # plt.xticks(rotation=45)
        plt.tight_layout()
        outpath_heatmap=os.path.join(output_folder, 'heatmap_fa.jpeg')
        plt.savefig(outpath_heatmap, dpi=300)
        plt.show()
        print(f"[SAVE] heatmap saved to {outpath_heatmap}!\n")
            
        
    # ---------------------------cronbach's alpha-------------------------------
    if get_cronbachs_a==True:
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
        for tactic, labels in labels2tac.items():
            alpha, _ = cronbach_alpha(data[labels])
            print(f"- {tactic}: {alpha}")                
        
        
    # -----------------------------comm&uniq-------------------------------
    if get_barplot==True:        
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
    
    # -----------------------------3D PCA-------------------------------
    if get_pca3d==True:
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
        # plt.savefig("../figs/CC3D.jpg",dpi=300)
        plt.show()


    # -----------------------------tab-------------------------------
    if get_table==True:
        loadings = fa.loadings_
        communalities = fa.get_communalities()
        uniqueness = fa.get_uniquenesses()
        print(f"[INFO] mean COMM (> 0.6):{communalities.mean()}\n"
            f"[INFO] mean UNIQ : {uniqueness.mean()}\n")

        # gamma 和 sigma
        abs_loadings = np.abs(loadings)
        gamma = abs_loadings.max(axis=1)                   # 最大因子载荷
        sigma = np.sort(abs_loadings, axis=1)[:, -2]       # 第二大因子载荷

        # get DataFrame
        columns = ['gamma', 'sigma'] + \
                [f'Factor{i+1}' for i in range(loadings.shape[1])] + \
                ['communality', 'uniqueness']

        df = pd.DataFrame(
            np.column_stack([gamma, sigma, loadings, communalities, uniqueness]),
            index=data.columns,
            columns=columns
        )

        # df.rename(columns={'Factor1':"exemplarité",
        #                 'Factor2':"sociabilité",
        #                 'Factor3':"hospitalité",
        #                 'Factor4':"auto-promotion",
        #                 'Factor5':"ouverture"},
        #         inplace=True)
        
        # desired_order=['gamma','sigma','ouverture','sociabilité',"hospitalité","auto-promotion","exemplarité"]
        # df=df[desired_order]

        df.round(3)
        outpath_tab=os.path.join(output_folder, 'table_fa.csv')
        df.to_csv(outpath_tab, index=False)
        print(f"[SAVE] table saved to {outpath_tab}!\n")
    
   
        

    return 