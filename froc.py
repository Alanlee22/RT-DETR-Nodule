import os
import glob
import math
import matplotlib.pyplot as plt
import bisect  # 导入bisect库用于二分查找

############################
# 1. 解析 YOLO 格式（保持不变）
############################
def parse_yolo_label(txt_file, image_width, image_height):
    # 原有代码不变
    boxes = []
    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx = float(parts[1])
            cy = float(parts[2])
            w  = float(parts[3])
            h  = float(parts[4])
            conf = 1.0
            if len(parts) >= 6:
                conf = float(parts[5])
            box_w = w * image_width
            box_h = h * image_height
            box_cx = cx * image_width
            box_cy = cy * image_height
            x1 = box_cx - box_w / 2.0
            y1 = box_cy - box_h / 2.0
            x2 = box_cx + box_w / 2.0
            y2 = box_cy + box_h / 2.0
            boxes.append([cls_id, x1, y1, x2, y2, conf])
    return boxes

############################
# 2. 计算 IoU（保持不变）
############################
def compute_iou(boxA, boxB):
    # 原有代码不变
    x1A, y1A, x2A, y2A = boxA[1], boxA[2], boxA[3], boxA[4]
    x1B, y1B, x2B, y2B = boxB[1], boxB[2], boxB[3], boxB[4]
    inter_x1 = max(x1A, x1B)
    inter_y1 = max(y1A, y1B)
    inter_x2 = min(x2A, x2B)
    inter_y2 = min(y2A, y2B)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    areaA = (x2A - x1A) * (y2A - y1A)
    areaB = (x2B - x1B) * (y2B - y1B)
    union_area = areaA + areaB - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

############################
# 3. 读取检测 & GT（保持不变）
############################
def load_detections_and_gts(det_dir, gt_dir, image_width, image_height):
    # 原有代码不变
    all_detections = {}
    all_gts = {}
    det_files = sorted(glob.glob(os.path.join(det_dir, '*.txt')))
    for detf in det_files:
        img_id = os.path.splitext(os.path.basename(detf))[0]
        det_boxes = parse_yolo_label(detf, image_width, image_height)
        all_detections[img_id] = det_boxes
    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.txt')))
    for gtf in gt_files:
        img_id = os.path.splitext(os.path.basename(gtf))[0]
        gt_boxes = parse_yolo_label(gtf, image_width, image_height)
        for b in gt_boxes:
            b[5] = 1.0
        all_gts[img_id] = gt_boxes
    return all_detections, all_gts

############################
# 4. 计算 FROC（修改部分）
############################
def compute_froc(detections, gts, iou_thr=0.5):
    # 合并检测框并按置信度排序
    det_list = []
    for img_id, dets in detections.items():
        for d in dets:
            det_list.append([img_id] + d)  # 格式：[img_id, cls, x1, y1, x2, y2, conf]
    det_list.sort(key=lambda x: x[-1], reverse=True)  # 降序排列

    # 初始化变量
    all_img_ids = list(gts.keys())
    n_images = len(all_img_ids)
    total_gt = sum(len(v) for v in gts.values())
    used_gt_flags = {img: [False] * len(gts[img]) for img in all_img_ids}
    TP, FP = 0, 0
    avg_fps = []  # 记录每个步骤的平均FP
    recalls = []   # 记录每个步骤的召回率

    # 处理每个检测框，记录数据点
    for det in det_list:
        img_id, cls, x1, y1, x2, y2, conf = det
        best_iou = 0.0
        best_gt_idx = -1
        # 查找匹配的GT
        for i, gt in enumerate(gts[img_id]):
            if used_gt_flags[img_id][i]:
                continue
            iou = compute_iou([cls, x1, y1, x2, y2, conf], gt)
            if iou >= iou_thr and iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        # 更新TP/FP
        if best_gt_idx != -1:
            TP += 1
            used_gt_flags[img_id][best_gt_idx] = True
        else:
            FP += 1
        # 计算当前指标并记录
        current_avg_fp = FP / n_images if n_images > 0 else 0.0
        current_recall = TP / total_gt if total_gt > 0 else 0.0
        avg_fps.append(current_avg_fp)
        recalls.append(current_recall)

    # 定义目标FROC横坐标
    froc_x = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    froc_y = []

    # 为每个x_target找到对应的召回率
    for x in froc_x:
        # 使用bisect找到第一个avg_fp >=x的位置
        idx = bisect.bisect_left(avg_fps, x)
        if idx < len(recalls):
            froc_y.append(recalls[idx])
        else:
            # 若所有avg_fp都小于x，使用最大召回率
            froc_y.append(recalls[-1] if recalls else 0.0)

    return froc_x, froc_y

############################
# 5. 绘制 FROC（保持不变）
############################
def plot_froc(froc_x, froc_y):
    plt.figure()
    plt.plot(froc_x, froc_y, marker='o', linestyle='-')
    plt.xscale('log')  # 横坐标使用对数刻度以更好展示
    plt.xlabel('Average FP per Image (log scale)')
    plt.ylabel('Recall (Sensitivity)')
    plt.title('FROC Curve')
    plt.grid(True, which="both", ls="--")
    plt.xticks(froc_x, labels=[str(x) for x in froc_x])  # 确保横坐标显示指定值
    plt.show()

############################
# 6. main 函数示例（保持不变）
############################
if __name__ == '__main__':
    image_w, image_h = 640, 640
    det_dir = 'E:\\360MoveData\\Users\\Alan\\Desktop\\RTDETR-main\\runs\\detect\\exp\\labels'
    gt_dir = 'E:\\360MoveData\\Users\\Alan\\Desktop\\RTDETR-main\\dataset2\\val\\labels'
    all_dets, all_gts = load_detections_and_gts(det_dir, gt_dir, image_w, image_h)
    froc_x, froc_y = compute_froc(all_dets, all_gts, iou_thr=0.5)
    plot_froc(froc_x, froc_y)