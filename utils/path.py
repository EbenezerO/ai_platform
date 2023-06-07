# -*- coding: utf-8 -*-
# @Author  : ouyang
# @Time    : 2023/6/7 11:38
import os
import os.path as osp


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))