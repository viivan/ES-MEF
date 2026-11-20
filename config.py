

import os

MODEL_NAME = "ES-MEF"

CSV_PATH = r""
SAVE_DIR = r""


WINDOW_SEC = 8.0            
STRIDE_SEC = 4.0
FS_ACO = 16000            
NODE_WINDOW = 8             


D_MODEL = 32            
LSTM_HIDDEN = 32
LSTM_LAYERS = 1
AMP_AUG = 0.3
CNN_CHANNELS = [8, 16, 32]
CNN_KERNELS = [3, 5, 7]
CNN_KERNEL_BASE = 3
DROPOUT = 0.5


W_LOCAL = 3                 
GCN_LAYERS = 2
GCN_DROPOUT = 0.5


GAMMA = 0.5                 
RGCN_LAYERS = 2


ACF_HEADS = 4
ACF_DROPOUT = 0.5


HIDDEN_DIM = 32
NUM_CLASSES = 3            


ETA = 1.0                 


BATCH_SIZE = 32
EPOCHS = 100
LR = 5e-4
WEIGHT_DECAY = 5e-3
LAMBDA_1 = 0.3              
LAMBDA_2 = 0.3             
DEVICE = "cuda"


ACO_PRETRAIN = True
ACO_PRETRAIN_PATH = r""
ACO_PRETRAIN_FREEZE = False


TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
RANDOM_SEED = 42
SPLIT_MODE = "date_time"


FAULT_NAMES = ["Normal", "Minor_Fault", "Critical_Fault"]
FAULT_TO_IDX = {name: i for i, name in enumerate(FAULT_NAMES)}
