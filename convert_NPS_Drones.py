import os
import argparse
import re
import cv2
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to dataset root containing Videos/ and Video_Annotation/')
    parser.add_argument('--save_root', type=str, required=True,
                        help='Path to the datasets directory')
    parser.add_argument('--save_vis', action='store_true',
                        help='Enable visualization using OpenCV')
    return parser.parse_args()

def convert_bbox_to_yolo(size, box):
    """Convert bbox from (xmin, ymin, xmax, ymax) to YOLO format."""
    width, height = size
    xmin, ymin, xmax, ymax = box
    x_center = (xmin + xmax) / 2.0 / width
    y_center = (ymin + ymax) / 2.0 / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    return [x_center, y_center, w, h]

def parse_txt_line(line):
    """Parse a line with one or more bounding boxes."""
    match = re.match(r'time_layer: (\d+)\s+detections:', line.strip())
    if not match:
        return None, []
    frame_id = int(match.group(1))
    bboxes = re.findall(r'\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)', line)
    boxes = [tuple(map(int, bbox)) for bbox in bboxes]
    return frame_id, boxes

def convert_clip(txt_path, video_path, image_save_path, labels_save_path, save_vis=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Unable to open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 读取标注文件
    annotations = {}
    with open(txt_path, 'r') as f:
        for line in f:
            frame_id, boxes = parse_txt_line(line)
            if frame_id is not None:
                annotations.setdefault(frame_id, []).extend(boxes)

    frame_idx = 0
    video_name = os.path.splitext(os.path.basename(txt_path))[0].replace('_gt', '')

    pbar = tqdm(total=total_frames, desc=os.path.basename(txt_path))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id = frame_idx + 1
        yolo_lines = []

        if frame_id in annotations:
            for bbox in annotations[frame_id]:
                yolo_box = convert_bbox_to_yolo((frame_width, frame_height), bbox)
                yolo_line = f"0 {' '.join([f'{x:.6f}' for x in yolo_box])}"
                yolo_lines.append(yolo_line)

                if save_vis:
                    ymin, xmin , ymax, xmax = bbox
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # 写入 YOLO 标签
        txt_name = f"{video_name}_frame{frame_id:04d}.txt"
        txt_output_path = os.path.join(labels_save_path, txt_name)
        with open(txt_output_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        # 保存图片
        image_name = os.path.join(image_save_path, txt_name.replace('.txt', '.jpg'))
        cv2.imwrite(image_name, frame)

        if save_vis and yolo_lines:
            cv2.imshow("YOLO Visualization", frame)
            cv2.waitKey(0)

        frame_idx += 1
        pbar.update(1)

    cap.release()
    if save_vis:
        cv2.destroyAllWindows()
    pbar.close()

def convert_all(data_root, save_root, dataset_name, save_vis=False):
    video_dir = os.path.join(data_root, 'Videos')
    anno_dir = os.path.join(data_root, 'Video_Annotation')
    # label_dir = os.path.join(data_root, 'yolo_labels')
    images_save_path = os.path.join(save_root, dataset_name, 'images')
    labels_save_path = os.path.join(save_root, dataset_name, 'labels')
    os.makedirs(images_save_path, exist_ok=True)
    os.makedirs(labels_save_path, exist_ok=True)

    txt_files = [f for f in os.listdir(anno_dir) if f.endswith('_gt.txt')]

    for txt_file in txt_files:
        txt_path = os.path.join(anno_dir, txt_file)
        video_name = txt_file.replace('_gt.txt', '.mov')
        video_path = os.path.join(video_dir, video_name)

        if not os.path.exists(video_path):
            print(f"⚠️ Missing video: {video_path}")
            continue

        convert_clip(txt_path, video_path, images_save_path, labels_save_path, save_vis=save_vis)

def main():
    args = parse_args()
    convert_all(args.data_root, args.save_root, dataset_name="NPS", save_vis=args.save_vis)

if __name__ == '__main__':
    main()
