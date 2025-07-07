import pandas as pd
import torch
from bert_score import BERTScorer

device = "cuda:0" if torch.cuda.is_available() else "cpu"
mode_path = r"Hugging_Model/deberta-xlarge-mnli"
scorer = BERTScorer(lang="en", model_type=mode_path, num_layers=40, batch_size=32, device=device)


def round_result(result):
    return round(result*100, 2)


def evaluate_debert_score(references, predictions, return_F1=None, mode_path=None):
    """计算BERT-Scroe, 使用模型: deberta-xlarge-mnli """
    assert len(references) == len(predictions), 'candidates and references must have the same length'

    # 计算BERTScore-P/R/F1, Shape:[len(references)]
    # num_layers：在源码中查表发现deberta-xlarge-mnli对应于40
    # lang=en：英文默认的BERT就为roberta-large
    # model_type=mode_path：要传入自己的预训练模型路径
    P, R, F1 = scorer.score(predictions, references, verbose=False)
    if return_F1:
        return F1
    return P, R, F1


def evaluate_meta_rec(file_path, threshold=0.7):
    """
    计算隐喻标签匹配值, P R F1(宏平均)
    threshold: 判断BERT-Score的阈值, 默认为0.7, 也就是说当源域匹配值和目标域匹配值均>0.7时，表示预测源域目标域正确
    """

    assert file_path.endswith('.xlsx'), '请传入excel文件'
    train_file = pd.read_excel(file_path)
    precision_list = []  # 记录每张 meme 的 P
    recall_list = []  # 记录每张 meme 的 R
    f1_list = []  # 记录每张 meme 的 F1

    # 需要计算每个模因下的混淆矩阵，然后计算P R F1
    # TP：模型预测源域-目标域正确的数量
    # FP：模型预测不对的数量，也就是模型多报/捏造的对。
    # FN：模型漏掉真实的源域-目标域对的数量。
    for i in range(len(train_file)):
        image_name = train_file.loc[i, 'image_name']    # image_name
        Meta_gt = train_file.loc[i, 'Meta_gt']      # Meta_gt
        Meta_test = train_file.loc[i, 'Meta_test']   # Meta_test
        print(f'{i}-图片名称: ', image_name, '测试阈值: ', threshold)

        # 处理test, gt, 将其转成列表以及索引值
        Meta_gt_list = eval(Meta_gt)
        Meta_test_list = eval(Meta_test)
        matched_gold = [False] * len(Meta_gt_list)  # 记录匹配的gt
        matched_test = [False] * len(Meta_test_list)    # 记录匹配的预测

        # 找出每个预测中最为匹配的gt,然后保存
        for j, meta_test in enumerate(Meta_test_list):
            source_test, target_test = meta_test
            source_max = 0  # 记录源域匹配的最大分数
            target_max = 0  # 记录目标域匹配的最大分数
            max_gt = ''  # 记录最大分数的真实值，也就是真实标签
            gt_index = -1  # 记录匹配的gt的索引
            for k, meta in enumerate(Meta_gt_list):   # 遍历gt,找出和test最匹配的gt
                source_gt, target_gt = meta    # 提取目标标签
                souce_score = evaluate_debert_score([source_test], [source_gt], return_F1=True)
                target_score = evaluate_debert_score([target_test], [target_gt], return_F1=True)
                dif_source = souce_score - source_max
                dif_target = target_score - target_max
                # print('源域得分: ', source_score, '目标域得分: ', target_score)
                if (dif_source >= 0 and dif_target >= 0) or (dif_source >= 0 and target_score > threshold) or (souce_score > threshold and dif_target >= 0):
                    gt_index = k
                    max_gt = meta
                    source_max = souce_score
                    target_max = target_score
            # 如果源域和目标域都匹配成功，则记录预测和真实位置都为true
            if source_max > threshold and target_max > threshold:
                matched_test[j] = True
                matched_gold[gt_index] = True

        # 统计源域和目标域匹配的个数, TP/FP/FN
        TP = matched_test.count(True)   # 模型预测源域-目标域正确的数量
        FP = matched_test.count(False)  # 模型多报/捏造的对/预测错的数量
        FN = matched_gold.count(False)  # 模型漏掉真实的源域-目标域对的数量

        # 计算P/R/F1
        precision = TP / (TP + FP) if TP + FP else 0.0  # 如果TP + FP均为0，则返回0
        recall = TP / (TP + FN) if TP + FN else 0.0     # 如果TP + FP均为0，则返回0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0   # 如果precision + recall均为0，则返回0

        # 存储结果
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    # 输出宏平均值 P R F1
    print('P: ', sum(precision_list) / len(precision_list))
    print('R: ', sum(recall_list) / len(recall_list))
    print('F1: ', sum(f1_list) / len(f1_list))

    P = round_result(sum(precision_list) / len(precision_list))
    R = round_result(sum(recall_list) / len(recall_list))
    F1 = round_result(sum(f1_list) / len(f1_list))

    return P, R, F1

if __name__ == '__main__':
    file_path = r'Process_Meta_Rec2.xlsx'
    evaluate_meta_rec(file_path)
