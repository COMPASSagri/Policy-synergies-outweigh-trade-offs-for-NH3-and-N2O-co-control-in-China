# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 11:08:20 2025

@author: sera
"""

import pandas as pd
import numpy as np

def generate_multiplier(cv, sample_size):
    """生成服从对数正态分布的乘数因子 X，满足 E[X]=1, CV[X]=cv"""
    sigma_sq = np.log(1 + cv**2)
    sigma = np.sqrt(sigma_sq)
    mu = -sigma_sq / 2
    return np.random.lognormal(mean=mu, sigma=sigma, size=sample_size)


# ==================== 配置参数 ====================
file_path = "D:/BaiduSyncdisk/Basic Data/AGR_STAT/0文章分析内容汇总/CV_factors.xlsx"


sample_size = 1000
random_seed = 42
np.random.seed(random_seed)

# ==================== 读取原始 A 和 EF 表的数据列名 ====================
df_a = pd.read_excel(file_path, sheet_name="A")
df_ef = pd.read_excel(file_path, sheet_name="EF")

# 原始列名
a_columns_raw = df_a.columns.tolist()
ef_columns_raw = df_ef.columns.tolist()

# 加上后缀，与 final_df 对应
a_columns = [f"{col}A" for col in a_columns_raw]
ef_columns = [f"{col}EF" for col in ef_columns_raw]

# ==================== 读取 CV 配置表 ====================
cv_df = pd.read_excel(file_path, header=None)
cv_df.columns = cv_df.iloc[0]    # 第一行为列名
cv_df = cv_df.drop(0)            # 删除第一行

cv_dict = cv_df.iloc[0].to_dict()

print("读取到的CV设定：")
for k, v in cv_dict.items():
    print(f"  {k}: CV={v}")

# ==================== 两类排放：NH3 & N2O ====================
emission_types = ["NH3", "N2O"]

# ==================== 输出文件 ====================
output_filename = "CV_FINAL.xlsx"

with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:

    for em_type in emission_types:
        print(f"\n==========================")
        print(f"开始生成：{em_type} 排放的乘数因子")
        print("==========================")

        # -------- A sheet 的乘数 --------
        multipliers_a = {}
        for raw_col, col in zip(a_columns_raw, a_columns):
            cv = float(cv_dict.get(raw_col, 0.05))  # CV 表中的列名是无后缀版本
            values = generate_multiplier(cv, sample_size)
            multipliers_a[col] = values

            print(f"[A - {col}] CV={cv:.3f}, mean={np.mean(values):.4f}, "
                  f"sampleCV={np.std(values)/np.mean(values):.4f}")

        pd.DataFrame(multipliers_a).to_excel(
            writer,
            sheet_name=f"A_multipliers_{em_type}",
            index=False
        )

        # -------- EF sheet 的乘数 --------
        multipliers_ef = {}
        for raw_col, col in zip(ef_columns_raw, ef_columns):
            cv = float(cv_dict.get(raw_col, 0.50))
            values = generate_multiplier(cv, sample_size)
            multipliers_ef[col] = values

            print(f"[EF - {col}] CV={cv:.3f}, mean={np.mean(values):.4f}, "
                  f"sampleCV={np.std(values)/np.mean(values):.4f}")

        pd.DataFrame(multipliers_ef).to_excel(
            writer,
            sheet_name=f"EF_multipliers_{em_type}",
            index=False
        )

print("\n==============================================")
print(f"完成！已生成并保存至：{output_filename}")
print("==============================================")

