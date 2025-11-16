import pandas as pd
import torch
from bert_score import BERTScorer


device = "cuda" if torch.cuda.is_available() else "cpu"
mode_path = r"deberta-xlarge-mnli"
scorer = BERTScorer(lang="en", model_type=mode_path, num_layers=40, batch_size=32, device=device)


def round_result(result):
    return round(result*100, 2)


def evaluate_debert_score(references, predictions, return_F1=None):
    """BERT-Score, model: deberta-xlarge-mnli """
    assert len(references) == len(predictions), 'candidates and references must have the same length'

    P, R, F1 = scorer.score(predictions, references, verbose=False)
    if return_F1:
        return F1
    return P, R, F1


def evaluate_meta_rec(file_path, threshold=0.7):
    """
    threshold: τ, default 0.7
    """
    assert file_path.endswith('.xlsx'), 'Please pass in the excel file.'
    train_file = pd.read_excel(file_path)
    precision_list = []
    recall_list = []
    f1_list = []

    for i in range(len(train_file)):
        image_name = train_file.loc[i, 'image_name']    # image_name
        Meta_gt = train_file.loc[i, 'Meta_gt']      # Meta_gt
        Meta_test = train_file.loc[i, 'Meta_test']   # Meta_test
        print(f'{i}-image name: ', image_name, 'threshold: ', threshold)

        # Process test and gt, converting them into lists and index values
        Meta_gt_list = eval(Meta_gt)
        Meta_test_list = eval(Meta_test)
        matched_gold = [False] * len(Meta_gt_list)
        matched_test = [False] * len(Meta_test_list)

        # Find the gt that best matches each prediction and then save it
        for j, meta_test in enumerate(Meta_test_list):
            source_test, target_test = meta_test
            source_max = 0
            target_max = 0
            max_gt = ''
            gt_index = -1
            for k, meta in enumerate(Meta_gt_list):
                source_gt, target_gt = meta
                souce_score = evaluate_debert_score([source_test], [source_gt], return_F1=True)
                target_score = evaluate_debert_score([target_test], [target_gt], return_F1=True)
                dif_source = souce_score - source_max
                dif_target = target_score - target_max
                # print('source_score: ', source_score, 'target_score: ', target_score)
                if (dif_source >= 0 and dif_target >= 0) or (dif_source >= 0 and target_score > threshold) or (souce_score > threshold and dif_target >= 0):
                    gt_index = k
                    max_gt = meta
                    source_max = souce_score
                    target_max = target_score
            if source_max > threshold and target_max > threshold:
                matched_test[j] = True
                matched_gold[gt_index] = True

        # TP/FP/FN
        TP = matched_test.count(True)
        FP = matched_test.count(False)
        FN = matched_gold.count(False)

        # P/R/F1
        precision = TP / (TP + FP) if TP + FP else 0.0
        recall = TP / (TP + FN) if TP + FN else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        # result
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

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
