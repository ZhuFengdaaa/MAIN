import argparse
from polyaxon_client.tracking import get_data_paths
import os


class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")
        self.parser.add_argument(
            "--run-type",
            choices=["train", "eval"],
            required=True,
            help="run type of the experiment (train or eval)",
        )
        self.parser.add_argument(
            "--exp-config",
            type=str,
            required=True,
            help="path to config yaml containing info about experiment",
        )

        self.parser.add_argument(
            "--agent-type",
            choices=["no-map", "oracle", "oracle-ego", "proj-neural", "obj-recog"],
            required=True,
            help="agent type: oracle, oracleego, projneural, objrecog",
        )

        self.parser.add_argument(
            "opts",
            default=None,
            nargs=argparse.REMAINDER,
            help="Modify config options from command line",
        )

        self.parser.add_argument(
            "--atp-data",
            default="habitat-data"
        )

        self.parser.add_argument(
            "--atp",
            action="store_const", 
            default=False, 
            const=True
        )

        self.args = self.parser.parse_args()

        if self.args.atp:
            self.args.atp_path = os.path.join(get_data_paths()['ceph'], self.args.atp_data)
            self.args.host_path = os.path.join(get_data_paths()['host-path'], self.args.atp_data)

param = Param()
args = param.args