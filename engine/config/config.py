# -*- coding: utf-8 -*-
# @Author  : ouyang
# @Time    : 2023/6/6 9:18
import os.path as osp
from pathlib import Path
from typing import Optional, Union

from engine.utils.path import check_file_exist


class Config(dict):
    def __init__(self,
                 filename: Optional[Union[str, Path]] = None,
                 cfg_dict: dict = None,):
        if '__builtins__' in cfg_dict:
            del cfg_dict['__builtins__']
        self.filename = filename
        super().__init__(cfg_dict)

    @staticmethod
    def file2dict(filename):
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in ['.py', ]:
            raise OSError('Only py are supported now!')
        filename = str(filename) if isinstance(filename, Path) else filename

        # 定义一个空字典，用于存储变量及对应值
        variables = {}

        # 读取 py 文件中的内容
        with open(filename) as f:
            code = compile(f.read(), filename, 'exec')

        # 执行 py 文件中的代码，并将变量及对应值存储到字典中
        exec(code, variables)

        # 打印存储的变量及对应值
        print(variables)
        return Config(filename, variables)