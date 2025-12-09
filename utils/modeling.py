        
        
        
        
        
        
        
        
        
        
        # # -----------------------------tab2latex-------------------------------                
        # latex_code = df_fa.round(3).to_latex(
        #     caption="Tableau de l'analyse factorielle",
        #     label="tab:fa_table",
        #     index=True,        # 是否保留行索引（item 名称）
        #     escape=False       # False 可以保留 LaTeX 特殊字符，比如 _ 等
        # )
        # outpath_tab_latex=os.path.join(output_folder, 'table_fa_latex.tex')
        # with open(outpath_tab_latex, 'w') as f:
        #     f.write(latex_code)
        # print(f"[SAVE] table latex saved to {outpath_tab_latex}!\n")   
        # print(f"[WARNING] remember to rename factors by heatmap of zsc data!!!")            
      