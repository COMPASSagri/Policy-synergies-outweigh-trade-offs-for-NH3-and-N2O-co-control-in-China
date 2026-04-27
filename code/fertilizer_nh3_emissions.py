# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 09:12:17 2025

@author: sera
"""
# ===============================================
#   NH3 排放不确定性模拟（新版）
#   - EF：随机森林 bootstrap（逻辑保持不变）
#   - AL：外部 multiplier（取 1000 个乘数）
#   - 保存所有年份全部地市的 1000 次排放 → 一个 Excel 宽表
#   - 按 crop + 年份计算全国 EF 不确定性 → CSV
#   - 不再保存 npy
# ===============================================
import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)   
pd.set_option('display.float_format', lambda x: '%.5f' % x) 
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)
from sklearn.ensemble import RandomForestRegressor
import joblib
from matplotlib.mathtext import _mathtext as mathtext
mathtext.FontConstantsBase.sup1 = 0.3 
import os
import gc
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

#%%载入XY
nh3df = pd.read_excel('./nh3XY.xlsx')
X = nh3df[['STP', 'Prec', 'Tmp', 'soc', 'tn', 'pH', 'bd', 'clay', 'cec', 'Nrate', 'UOA', 'ABC', 'Others', 'Manure', 'Compound', 'SBC', 'DPM', 'Rice', 'Wheat', 'Maize', 'Other_upland', 'Vegetable']]
Y = nh3df['EF']

#%% ====== 1. 载入参数和 multipliers ======
grid     = joblib.load('./RF_grid_V727.pkl')
best_params = grid.best_params_

mult_file = "./CV_FINAL.xlsx"
mult_sheet = "A_multipliers_NH3"
multipliers_df = pd.read_excel(mult_file, sheet_name=mult_sheet)

n_jobs_boot = min(4, joblib.cpu_count())

# multipliers_df 必须包含这些列
crop_multiplier_col = {
    "rice":  "riceA",
    "wheat": "wheatA",
    "maize": "maizeA",
    "vege":  "vegeA",
    "other": "otherA"
}

rename_dict = {"manure": "Manure", "compound": "Compound"}

X_col = ['STP', 'Prec', 'Tmp', 'soc', 'tn', 'pH', 'bd', 'clay', 'cec',
         'Nrate', 'UOA', 'ABC', 'Others', 'Manure', 'Compound', 'SBC',
         'DPM', 'Rice', 'Wheat', 'Maize', 'Other_upland', 'Vegetable']

dfs = ['rice2000', 'wheat2000', 'maize2000', 'rice2001', 'wheat2001', 'maize2001',
       'rice2002', 'wheat2002', 'maize2002', 'rice2003', 'wheat2003', 'maize2003',
       'rice2004', 'wheat2004', 'maize2004', 'rice2005', 'wheat2005', 'maize2005', 
       'rice2006', 'wheat2006', 'maize2006', 'rice2007', 'wheat2007', 'maize2007', 
       'rice2008', 'wheat2008', 'maize2008', 'rice2009', 'wheat2009', 'maize2009', 
       'rice2010', 'wheat2010', 'maize2010', 'rice2011', 'wheat2011', 'maize2011',
       'rice2012', 'wheat2012', 'maize2012', 'rice2013', 'wheat2013', 'maize2013',
       'rice2014', 'wheat2014', 'maize2014', 'rice2015', 'wheat2015', 'maize2015', 
       'rice2016', 'wheat2016', 'maize2016', 'rice2017', 'wheat2017', 'maize2017', 
       'rice2018', 'wheat2018', 'maize2018', 'rice2019', 'wheat2019', 'maize2019',
       'rice2020', 'wheat2020', 'maize2020', 'rice2021', 'wheat2021', 'maize2021', 
       'rice2022', 'wheat2022', 'maize2022', 
       'vege2000', 'vege2001', 'vege2002', 'vege2003', 'vege2004', 'vege2005', 'vege2006', 'vege2007', 'vege2008', 
       'vege2009', 'vege2010', 'vege2011', 'vege2012', 'vege2013', 'vege2014', 'vege2015', 'vege2016', 'vege2017', 
       'vege2018', 'vege2019', 'vege2020', 'vege2021', 'vege2022', 
       'other2000','other2001', 'other2002', 'other2003', 'other2004', 'other2005', 'other2006', 'other2007', 'other2008', 
       'other2009', 'other2010', 'other2011', 'other2012', 'other2013','other2014', 'other2015', 'other2016', 'other2017',
       'other2018', 'other2019', 'other2020', 'other2021', 'other2022']

n_boot = 100
n_mc   = 1000   # Monte Carlo = multiplier 1000 次

# 输出目录（可根据实际情况修改）
os.makedirs("./results_nh3/", exist_ok=True)
#%% ====== 3. 用于全国 EF CV 统计的数据结构 ======
ef_records = []   # 用于最终 EF_uncertainty_national.csv
all_city_records = []   # 保存到 Excel 的宽表
#%%并行运算
#%% 3. 预生成 bootstrap 索引（避免每次重复 np.random.choice）
# ==========================================================
boot_index_map = np.random.randint(0, len(X), size=(n_boot, len(X)))
# ==========================================================
#%% 4. 定义并行训练 RF + 预测单次 bootstrap 的函数
# ==========================================================

def train_and_predict_boot(i, X, Y, ds, X_col, best_params):
    """并行执行：训练一次 bootstrap RF + 预测当前 df 的 EF"""
    idx = boot_index_map[i]
    X_boot = X.iloc[idx]
    y_boot = Y.iloc[idx]

    rf = RandomForestRegressor(**best_params, random_state=i, n_jobs=-1)
    rf.fit(X_boot, y_boot)

    return rf.predict(ds[X_col])


# ==========================================================
#%% 5. 全局存储
# ==========================================================

all_city_records = []   # 保存最终宽表的行
ef_records = []         # 保存全国 EF CV 的基础数据

# ==========================================================
#%% 6. 主循环：处理所有 df
# ==========================================================

for df_name in dfs:

    print(f"Processing {df_name} ...")

    # ---- crop + year ----
    crop = ''.join([c for c in df_name if c.isalpha()])
    year = int(''.join([c for c in df_name if c.isdigit()]))

    # ---- 读取数据 ----
    ds = pd.read_excel(f'./prepare_nh3/{df_name}.xlsx')
    ds.rename(columns=rename_dict, inplace=True)
    n_rows = len(ds)

    # ==========================================================
    #   (A) Bootstrap EF  —— 使用 joblib 并行加速
    # ==========================================================

    EF_preds = np.array(
        Parallel(n_jobs=n_jobs_boot)(
            delayed(train_and_predict_boot)(i, X, Y, ds, X_col, best_params)
            for i in range(n_boot)
        )
    ).T    # 转为 n_rows × n_boot

    # ==========================================================
    #   (B) 构建 AL_sim（乘数因子）
    # ==========================================================

    AL_raw = ds['土耕施肥量'].values.reshape(-1, 1)
    multipliers = multipliers_df[crop_multiplier_col[crop]].values[:n_mc]
    AL_sim = AL_raw * multipliers.reshape(1, -1)   # n_rows × 1000

    # ==========================================================
    #   (C) EF Monte Carlo — 随机抽取 bootstrap
    # ==========================================================

    boot_idx = np.random.randint(0, n_boot, size=n_mc)
    EF_sim = EF_preds[:, boot_idx]   # n_rows × 1000

    # ==========================================================
    #   (D) 排放模拟（向量化）
    # ==========================================================

    EM_sim = EF_sim * AL_sim * 1.214 / 1000    # n_rows × 1000
    EM_city_total = EM_sim.sum(axis=0)         # 1000

    # ==========================================================
    #   (E) 以地市聚合（不再判断 city 数量）
    # ==========================================================

    cities = ds["地市名"].values
    unique_cities = np.unique(cities)

    # 分组计算
    for c in unique_cities:
        mask = (cities == c).reshape(-1, 1)
        city_em = (EM_sim * mask).sum(axis=0)  # 1000 次排放

        all_city_records.append({
            "year": year,
            "city": c,
            **{f"MC_{i+1}": city_em[i] for i in range(n_mc)}
        })

    # ==========================================================
    #   (F) 为全国 EF CV 保存数据
    # ==========================================================

    ef_records.append({
        "year": year,
        "crop": crop,
        "AL": AL_sim.sum(axis=0),      # 各 MC 的施肥量
        "EM": EM_city_total            # 各 MC 的排放量
    })

    # 释放大数组
    del ds, AL_sim, EF_sim, EM_sim
    gc.collect()

    print(f"{df_name} finished\n")


# ==========================================================
# 7. 汇总全国 EF 不确定性（按 crop + year）
# ==========================================================

ef_summary = []

df_ef = pd.DataFrame([
    {"year": r["year"], "crop": r["crop"], "AL": r["AL"], "EM": r["EM"]}
    for r in ef_records
])

for year in sorted(df_ef["year"].unique()):
    for crop in ["rice", "wheat", "maize", "vege", "other"]:
        
        sub = df_ef[(df_ef["year"] == year) & (df_ef["crop"] == crop)]
        if len(sub) == 0:
            continue

        total_AL = np.sum([np.array(x) for x in sub["AL"]], axis=0)
        total_EM = np.sum([np.array(x) for x in sub["EM"]], axis=0)

        EF = total_EM / total_AL

        ef_summary.append({
            "year": year,
            "crop": crop,
            "EF_mean": EF.mean(),
            "EF_sd": EF.std(),
            "EF_cv": EF.std() / EF.mean(),
            "low95": np.percentile(EF, 2.5),
            "up95": np.percentile(EF, 97.5),
        })

ef_summary_df = pd.DataFrame(ef_summary)
ef_summary_df.to_csv("./results_nh3/EF_uncertainty_national.csv", index=False)


# ==========================================================
# 8. 输出最终宽表 Excel（所有年份所有城市）
# ==========================================================

df_city_wide = pd.DataFrame(all_city_records)
df_city_wide.to_excel("./results_nh3/all_city_MC.xlsx", index=False)

print("\nAll Finished Successfully!")
#%%检查
# 识别 MC 列
import numpy as np
import pandas as pd

# 假设 df_city_wide 已经在内存里
# 列名：["year", "city", "MC_1", "MC_2", ..., "MC_1000"]

# 找到所有 MC 列
mc_cols = [c for c in df_city_wide.columns if c.startswith("MC_")]

yearly_stats = []

for year in sorted(df_city_wide["year"].unique()):
    sub = df_city_wide[df_city_wide["year"] == year]

    # 每次 MC 对应全国总排放
    total_per_MC = sub[mc_cols].sum(axis=0).values  # shape = (1000,)

    # 统计指标
    mean_total = total_per_MC.mean()
    sd_total   = total_per_MC.std()
    cv_total   = sd_total / mean_total if mean_total != 0 else np.nan
    low95      = np.percentile(total_per_MC, 2.5)
    up95       = np.percentile(total_per_MC, 97.5)

    yearly_stats.append({
        "year": year,
        "mean_total_emission": mean_total,
        "sd_total_emission": sd_total,
        "cv_total_emission": cv_total,
        "low95_total_emission": low95,
        "up95_total_emission": up95
    })

df_yearly_stats = pd.DataFrame(yearly_stats)
print(df_yearly_stats)



