#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import csv
import argparse
import yaml
import os
from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def load_model_output(file_path):
    """加载模型输出的JSON文件（整个数组），返回 {dialogue_id: [demands]}"""
    data = defaultdict(list)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = json.load(f)  # 解析整个JSON
    # content 应该是一个列表，每个元素是一个时间窗口对象
    for time_window_item in content:
        # 每个时间窗口对象包含 demands 列表
        demands = time_window_item.get('demands', [])
        for demand in demands:
            dialogue_id = demand.get('source_dialogue_id')
            if not dialogue_id:
                # 如果没有source_dialogue_id，可以用时间窗口索引+需求索引构造，但这里暂不处理
                continue
            # 将demand作为一条提取结果
            data[dialogue_id].append(demand)
    return dict(data)


def load_ground_truth(file_path, dialogue_id_field='dialogue_id'):
    """加载 Ground Truth CSV 文件，返回 {dialogue_id: [demands]}"""
    data = defaultdict(list)
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dialogue_id = row.get(dialogue_id_field)
            if not dialogue_id:
                continue
            # 将 CSV 行转换为与模型输出一致的需求字典
            demand = {
                'location': row.get('location'),
                'item': row.get('item'),
                'quantity': float(row.get('quantity', 0)),
                'urgency': row.get('urgency'),
                # 如果有其他字段（如时间戳）也可以加入
            }
            data[dialogue_id].append(demand)
    return dict(data)


def normalize_demand(demand):
    """对需求进行标准化，便于比较（例如去除空格、统一大小写）"""
    normalized = {}
    for k, v in demand.items():
        if isinstance(v, str):
            normalized[k] = v.strip().lower()
        elif isinstance(v, float):
            normalized[k] = v
        else:
            normalized[k] = v
    return normalized


def demand_matches(pred_demand, true_demand, config):
    """
    判断预测需求是否与真实需求匹配（基于配置的容差）
    返回布尔值和匹配类型（exact 或 partial）
    """
    # 简单实现：所有字段必须完全一致（除数量允许误差）
    match = True
    match_type = 'exact'
    # 检查非数量字段
    for field in ['location', 'item', 'urgency']:
        if pred_demand.get(field) != true_demand.get(field):
            # 可加入模糊匹配，如地点名称相似度，此处简化
            match = False
            break
    if match:
        # 检查数量，允许误差
        pred_q = pred_demand.get('quantity', 0)
        true_q = true_demand.get('quantity', 0)
        tolerance = config.get('matching', {}).get('quantity_tolerance', 0)
        if abs(pred_q - true_q) > tolerance * max(true_q, 1):
            match = False
            match_type = 'quantity_mismatch'
    else:
        match_type = 'field_mismatch'
    return match, match_type


def evaluate_dialogue(pred_demands, true_demands, config):
    """
    评估单个对话的提取结果
    返回：tp, fp, fn, 以及每个预测的匹配详情
    """
    tp = 0
    matched_true_indices = set()
    details = []  # 记录每个预测的匹配情况

    # 为简化，采用贪婪匹配：对每个预测，尝试匹配第一个未匹配的真实需求
    for pred in pred_demands:
        pred_norm = normalize_demand(pred)
        matched = False
        for i, true in enumerate(true_demands):
            if i in matched_true_indices:
                continue
            true_norm = normalize_demand(true)
            match, match_type = demand_matches(pred_norm, true_norm, config)
            if match:
                tp += 1
                matched_true_indices.add(i)
                matched = True
                details.append({'pred': pred, 'true': true, 'match': 'tp', 'type': match_type})
                break
        if not matched:
            details.append({'pred': pred, 'true': None, 'match': 'fp'})

    # 未匹配的真实需求为假阴性
    fn = len(true_demands) - len(matched_true_indices)
    for i, true in enumerate(true_demands):
        if i not in matched_true_indices:
            details.append({'pred': None, 'true': true, 'match': 'fn'})

    fp = len(pred_demands) - tp
    return tp, fp, fn, details


def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM demand extraction.')
    parser.add_argument('--model_output', required=True, help='Path to model output JSONL file')
    parser.add_argument('--ground_truth', required=True, help='Path to ground truth CSV file')
    parser.add_argument('--config', default='src/config/eval_config.yaml', help='Path to config YAML')
    parser.add_argument('--output_dir', default='evals/results', help='Directory to save results')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 加载数据
    model_data = load_model_output(args.model_output)
    gt_data = load_ground_truth(args.ground_truth)

    # 确保只评估共同存在的对话
    common_ids = set(model_data.keys()) & set(gt_data.keys())
    if not common_ids:
        print("Warning: No common dialogue IDs found between model output and ground truth.")
        return

    # 存储总体统计
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_details = []

    for dialogue_id in common_ids:
        preds = model_data[dialogue_id]
        truths = gt_data[dialogue_id]
        tp, fp, fn, details = evaluate_dialogue(preds, truths, config)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_details.extend(details)

    # 计算宏观指标
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 如果有紧急程度字段，可以单独计算分类准确率
    urgency_true = []
    urgency_pred = []
    for detail in all_details:
        if detail['match'] == 'tp' and 'urgency' in detail['pred'] and 'urgency' in detail['true']:
            urgency_true.append(detail['true']['urgency'])
            urgency_pred.append(detail['pred']['urgency'])
    urgency_acc = accuracy_score(urgency_true, urgency_pred) if urgency_true else None

    # 组装结果
    results = {
        'total_dialogues': len(common_ids),
        'total_true_demands': total_tp + total_fn,
        'total_pred_demands': total_tp + total_fp,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'urgency_accuracy': urgency_acc,
    }

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # 可选：保存详细匹配报告
    details_path = os.path.join(args.output_dir, 'matching_details.json')
    with open(details_path, 'w') as f:
        json.dump(all_details, f, indent=2)

    print(f"Evaluation completed. Results saved to {out_path}")


if __name__ == '__main__':
    import sys

    sys.argv = [
        'evals.py',
        '--model_output', 'data/test/extracted_demands.json',
        '--ground_truth', 'data/seed/daily_demand_events_manifest.jsonl',
        '--config', 'evals/eval_config.yaml'
    ]
    main()
