#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

class HabitatLogger(logging.Logger):
    def __init__(
        self,
        name,
        level,
        filename=None,
        filemode="a",
        stream=None,
        format=None,
        dateformat=None,
        style="%",
    ):
        super().__init__(name, level)
        from habitat_baselines.common.param import args
        from polyaxon_client.tracking import get_outputs_path
        if args.atp:
            filename = os.path.join(get_outputs_path()['host-path'], filename)
        file_dir = os.path.dirname(filename)
        print(file_dir, filename)
        os.makedirs(file_dir, exist_ok=True)
        if filename is not None:
            handler = logging.FileHandler(filename, filemode)
        else:
            handler = logging.StreamHandler(stream)
        self._formatter = logging.Formatter(format, dateformat, style)
        handler.setFormatter(self._formatter)
        super().addHandler(handler)

    def add_filehandler(self, log_filename):
        filehandler = logging.FileHandler(log_filename)
        filehandler.setFormatter(self._formatter)
        self.addHandler(filehandler)

import time
timestr = time.strftime("%Y%m%d-%H%M%S")

logger = HabitatLogger(
    filename=f"logs.2/{timestr}.log",
    name="habitat", 
    level=logging.INFO, 
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
)
