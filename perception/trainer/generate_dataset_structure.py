import os
import argparse

def create_dirs(root):
    paths = [
        os.path.join(root, 'images', 'train'),
        os.path.join(root, 'images', 'val'),
        os.path.join(root, 'labels', 'train'),
        os.path.join(root, 'labels', 'val'),
    ]
    for p in paths:
        os.makedirs(p, exist_ok=True)
        # place .gitkeep for empty dirs
        open(os.path.join(p, '.gitkeep'), 'a').close()

def create_classifier_dirs(root):
    # classifier_data/train/<class>, classifier_data/val/<class>
    for split in ('train','val'):
        os.makedirs(os.path.join(root, split), exist_ok=True)

def write_sample_names(path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('示例英雄A\n示例英雄B\n')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='../dataset_detector', help='root path for detector data')
    p.add_argument('--classifier_root', default='../classifier_data', help='root path for classifier data')
    p.add_argument('--names', default='names.txt', help='output names file for classes')
    args = p.parse_args()

    create_dirs(args.root)
    create_classifier_dirs(args.classifier_root)
    write_sample_names(args.names)

    print('Created dataset skeleton:')
    print(' - detector images:', os.path.abspath(os.path.join(args.root, 'images')))
    print(' - detector labels:', os.path.abspath(os.path.join(args.root, 'labels')))
    print(' - classifier dirs:', os.path.abspath(args.classifier_root))
    print(' - names file:', os.path.abspath(args.names))

if __name__ == '__main__':
    main()
