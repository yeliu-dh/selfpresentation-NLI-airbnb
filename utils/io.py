import pandas as pd



def save_csv_as_latex(table_csv, output_path,caption, label, round=None):
    if round!=None:
        table_csv=table_csv.round(round)
        
    latex_code =table_csv.to_latex(
            caption=caption,
            label=label,
            index=True,        # 是否保留行索引（item 名称）
            escape=False       # False 可以保留 LaTeX 特殊字符，比如 _ 等
        )
    # print(f"LATEX : \n {latex_code}")
    with open(output_path, 'w') as f:
        f.write(latex_code)
    print(f"[SAVE] table latex saved to {output_path}!\n")   
    
    return 

