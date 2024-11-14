from yacs.config import CfgNode
import torch

# Values to fall back on in case no config file is passed
# Basic parameters
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_NUM_CLASSES = 6
DEFAULT_DROPOUT = 0.25
DEFAULT_HIDDEN_NEURONS = 1024

# List of the dataset names for training
DEFAULT_DATASETS_TRAIN = "data/datasets/train.csv"
# List of the dataset names for testing
DEFAULT_DATASETS_TEST = "data/datasets/test.csv"
DEFAULT_DATASETS_TEST_WA = "data/datasets/STSint.testinput.answers-students.wa"

# Number of data loading threads
DEFAULT_DATALOADER_NUM_WORKERS = 8

# Solver parameters
DEFAULT_SOLVER_OPTIMIZER_NAME = "ADAM"
DEFAULT_SOLVER_MAX_EPOCHS = 100
DEFAULT_SOLVER_BASE_LR = 0.0001
DEFAULT_SOLVER_LOG_PERIOD = 100
DEFAULT_SOLVER_BATCH_SIZE = 1

# Testing parameters
DEFAULT_TEST_BATCH_SIZE = 1
DEFAULT_TEST_WEIGHT = ""

# Output
DEFAULT_OUTPUT_DIR = "output"


def get_default_config() -> CfgNode:
    main_cfg = CfgNode()

    # Basic model attributes
    main_cfg.MODEL = CfgNode()
    main_cfg.MODEL.DEVICE = DEFAULT_DEVICE

    main_cfg.MODEL.NUM_CLASSES = DEFAULT_NUM_CLASSES
    main_cfg.MODEL.DROPOUT = DEFAULT_DROPOUT
    main_cfg.MODEL.HIDDEN_NEURONS = DEFAULT_HIDDEN_NEURONS

    # Datasets locations
    main_cfg.DATASETS = CfgNode()
    main_cfg.DATASETS.TRAIN = DEFAULT_DATASETS_TRAIN
    main_cfg.DATASETS.TEST = DEFAULT_DATASETS_TEST
    main_cfg.DATASETS.TEST_WA = DEFAULT_DATASETS_TEST_WA

    # Data loader workers
    main_cfg.DATALOADER = CfgNode()
    main_cfg.DATALOADER.NUM_WORKERS = DEFAULT_DATALOADER_NUM_WORKERS

    # Solver parameters set up
    main_cfg.SOLVER = CfgNode()
    main_cfg.SOLVER.OPTIMIZER_NAME = DEFAULT_SOLVER_OPTIMIZER_NAME
    main_cfg.SOLVER.MAX_EPOCHS = DEFAULT_SOLVER_MAX_EPOCHS
    main_cfg.SOLVER.BASE_LR = DEFAULT_SOLVER_BASE_LR
    main_cfg.SOLVER.LOG_PERIOD = DEFAULT_SOLVER_LOG_PERIOD
    main_cfg.SOLVER.BATCH_SIZE = DEFAULT_SOLVER_BATCH_SIZE

    # Testing parameters
    main_cfg.TEST = CfgNode()
    main_cfg.TEST.BATCH_SIZE = DEFAULT_TEST_BATCH_SIZE
    main_cfg.TEST.WEIGHT = DEFAULT_TEST_WEIGHT

    # Output
    main_cfg.OUTPUT_DIR = DEFAULT_OUTPUT_DIR

    return main_cfg
