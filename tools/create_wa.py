import argparse
import os
from os import mkdir
import sys

import torch
import yacs
from yacs.config import CfgNode

# Ugly but whatever
sys.path.append(".")
from net_default_cfg.default_config import get_default_config
from engine.inference_to_wa import format_to_wa
from roberta_model.RobertaISTS import RobertaISTS
from logger.logger import create_logger


def main():
    parser = argparse.ArgumentParser(description="Roberta iSTS - creating .wa files")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        with open(args.config_file, "r") as f:
            raw_yaml = f.read()
            config = yacs.config.CfgNode.load_cfg(raw_yaml)
    else:
        config = get_default_config()
    config.merge_from_list(args.opts)
    config.freeze()

    output_dir = config.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = create_logger("Model", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)
    logger.propagate = False

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(config))

    model = RobertaISTS.build_model_from_cfg(config)
    model.load_state_dict(torch.load(config.TEST.WEIGHT))

    format_to_wa(config, model)


if __name__ == '__main__':
    main()
