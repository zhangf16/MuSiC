import re
import numpy as np

# 原始数据字符串
data_str = """
{'T': 5, 't': 0, 'dropout': 0.2, 'mask_rate': 0.2, 'non_overlap_weight': 0.2, 'lamda': 0.4, 'best_mae_new': 0.6401148073935392, 'best_rmse_new': 0.9013195509333884, 'best_epoch': 25, 'mae': 0.6658163192807713, 'rmse': 0.9013195509333884, 'mae_none': 0.6766536316079983, 'rmse_none': 0.896345408282696, 'ndcg': 0.8729669145133331, 'novel': 0.2988553297738886, 'tail_rmse': 0.886205414580407, 'tail_mae': 0.6572381015656675}
{'T': 5, 't': 0, 'dropout': 0.2, 'mask_rate': 0.2, 'non_overlap_weight': 0.4, 'lamda': 0.4, 'best_mae_new': 0.6370098261946378, 'best_rmse_new': 0.9012000868083611, 'best_epoch': 20, 'mae': 0.6600448425364339, 'rmse': 0.9012000868083611, 'mae_none': 0.6762127157251807, 'rmse_none': 0.8952264067095086, 'ndcg': 0.8730682426727955, 'novel': 0.3014808592905201, 'tail_rmse': 0.8858156689829854, 'tail_mae': 0.6494172057193994}
{'T': 5, 't': 0, 'dropout': 0.2, 'mask_rate': 0.2, 'non_overlap_weight': 0.6, 'lamda': 0.4, 'best_mae_new': 0.6481637838986106, 'best_rmse_new': 0.900704845629003, 'best_epoch': 14, 'mae': 0.6688283055788384, 'rmse': 0.900704845629003, 'mae_none': 0.6642011524075246, 'rmse_none': 0.8995674220493541, 'ndcg': 0.872814982666869, 'novel': 0.3002298109565785, 'tail_rmse': 0.8832707673386236, 'tail_mae': 0.6569073289587323}
{'T': 5, 't': 0, 'dropout': 0.2, 'mask_rate': 0.2, 'non_overlap_weight': 0.8, 'lamda': 0.4, 'best_mae_new': 0.6347352749757332, 'best_rmse_new': 0.8987729020180429, 'best_epoch': 19, 'mae': 0.682299704051257, 'rmse': 0.8987729020180429, 'mae_none': 0.6647117581422725, 'rmse_none': 0.8976467535397891, 'ndcg': 0.8733165538629246, 'novel': 0.30283531321519525, 'tail_rmse': 0.8818719762016933, 'tail_mae': 0.6693845858172415}
{'T': 5, 't': 0, 'dropout': 0.2, 'mask_rate': 0.2, 'non_overlap_weight': 1.0, 'lamda': 0.4, 'best_mae_new': 0.6443654513410944, 'best_rmse_new': 0.9028069004519653, 'best_epoch': 18, 'mae': 0.6698808168791344, 'rmse': 0.9028069004519653, 'mae_none': 0.6705770165211445, 'rmse_none': 0.8981103869207361, 'ndcg': 0.87197142901898, 'novel': 0.30642106123383533, 'tail_rmse': 0.8844025854376404, 'tail_mae': 0.6556685047318989}

"""

# 添加逗号
data_str_with_commas = re.sub(r"\}\s*\{", r"}, {", data_str.strip())

# 将字符串转换为字典列表
data = eval(f"[{data_str_with_commas}]")

# 参数设置
dropouts = [0.2]
mask_rates = [0.1, 0.2, 0.3]
non_overlap_weights = [0.2,0.4,0.6, 0.8, 1.0]

# 创建空的矩阵字典
matrices = {dropout: np.zeros((len(non_overlap_weights), len(mask_rates))) for dropout in dropouts}

# 填充矩阵数据
for entry in data:
    dropout = entry['dropout']
    mask_rate = entry['mask_rate']
    non_overlap_weight = entry['non_overlap_weight']
    best_rmse_new = entry['best_rmse_new']
    i = non_overlap_weights.index(non_overlap_weight)
    j = mask_rates.index(mask_rate)
    matrices[dropout][i, j] = best_rmse_new

# 打印矩阵
for dropout in dropouts:
    print(f"Dropout: {dropout}")
    print("Mask Rate ->  ", mask_rates)
    for i, row in enumerate(matrices[dropout]):
        print(f"{non_overlap_weights[i]:<12} {row}")
    print()
