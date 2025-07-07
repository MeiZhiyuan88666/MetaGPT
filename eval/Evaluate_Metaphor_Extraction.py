import pandas as pd
import os
from META_REC import evaluate_meta_rec

model_path_dict = {
    'memegpt': r'data/minigpt'
}

model_list = ['memegpt']

file_list = 'Process_Meta_Rec2.xlsx'    # 要处理的文件
meta_rec_data = []  # 保存结果

# 最终结果保存路径
result_meta_rec_path = r'memegpt/result_meta_extract.xlsx'
support = 0.7   # 自己设定的阈值, 方便调试

# 可选阈值选项
thresholds_dict = {0.5: 0.5,
                   0.6: 0.6,
                   0.7: 0.7,
                   0.8: 0.8}

# 读取之前处理过的模型
before_data = []
before_output = pd.read_excel(result_meta_rec_path)
for i in range(len(before_output)):
    model_name = before_output.loc[i, 'model_name']
    before_data.append(model_name)


for model in model_list:
    print(f"当前处理模型:{model} - {file_list}")

    if model in before_data:
        print(f"{model}已处理过")
        continue

    pre_path = os.path.join(model_path_dict[model], file_list)

    if not os.path.exists(pre_path):
        print("测试文件不存在, 该模型未测试")

    P, R, F1 = evaluate_meta_rec(pre_path, threshold=thresholds_dict[support])
    meta_rec_data.append({
        'model_name': model,
        'precision': P,
        'recall': R,
        'f1-score': F1,
        'threshold': support
    })
    output = pd.read_excel(result_meta_rec_path)
    df = pd.DataFrame(meta_rec_data)
    output = output.append(df)
    output.to_excel(result_meta_rec_path, index=False)
    meta_rec_data = []
    print(f"{model}处理完成, 保存至{result_meta_rec_path}")
