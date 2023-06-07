# -*- coding: utf-8 -*-
# @Author  : ouyang
# @Time    : 2023/6/6 8:53
import argparse
import os
import os.path as osp

from config.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose model')
    parser.add_argument('config', help='train config file path')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.file2dict(args.config)
    print(cfg)


if __name__ == '__main__':
    main()