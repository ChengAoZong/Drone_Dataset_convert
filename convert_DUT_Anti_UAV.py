import os
import argparse
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to the "train" directory')
    parser.add_argument('--save_vis', action='store_true',
                        help='Enable visualization using OpenCV')
    return parser.parse_args()

def convert_bbox_to_yolo(size, box):
    """将边界框转换为YOLO格式"""
    width, height = size
    xmin, ymin, xmax, ymax = box
    x_center = (xmin + xmax) / 2.0 / width
    y_center = (ymin + ymax) / 2.0 / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    return [x_center, y_center, w, h]

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text.strip()
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        objects.append((name, (xmin, ymin, xmax, ymax)))
    return width, height, objects

def convert_all(data_root, split='train', save_vis=False):
    xml_dir = os.path.join(data_root, split, 'xml')
    img_dir = os.path.join(data_root, split, 'img')
    label_dir = os.path.join(data_root, split, 'yolo_labels')
    os.makedirs(label_dir, exist_ok=True)

    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]

    for xml_file in tqdm(xml_files, desc="Converting"):
        xml_path = os.path.join(xml_dir, xml_file)
        width, height, objects = parse_xml(xml_path)

        yolo_lines = []
        for name, bbox in objects:
            yolo_box = convert_bbox_to_yolo((width, height), bbox)
            # 假设只有一个类 “UAV”，类别ID为0
            yolo_line = f"0 {' '.join([f'{x:.6f}' for x in yolo_box])}"
            yolo_lines.append(yolo_line)

        # 保存 YOLO 标签
        txt_filename = os.path.splitext(xml_file)[0] + '.txt'
        txt_path = os.path.join(label_dir, txt_filename)
        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        # 可视化
        if save_vis:
            img_path = os.path.join(img_dir, os.path.splitext(xml_file)[0] + '.jpg')
            img = cv2.imread(img_path)
            H, W, C = img.shape
            if img is not None:
                with open(txt_path, 'r') as f:
                    anno = f.readlines()[0].split()
                    anno = [float(item) for item in anno]
                _, x_center, y_center, w, h = anno

                x_min = int((x_center - w/2)*W)
                x_max = int((x_center + w/2)*W)
                y_min = int((y_center - h/2)*H)
                y_max = int((y_center + h/2)*H)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.imshow("Annotation", img)
                cv2.waitKey(0)

    if save_vis:
        cv2.destroyAllWindows()

def main():
    args = parse_args()
    convert_all(args.data_root, split='train', save_vis=args.save_vis)
    convert_all(args.data_root, split='val', save_vis=args.save_vis)
    convert_all(args.data_root, split='test', save_vis=args.save_vis)

if __name__ == '__main__':
    main()
