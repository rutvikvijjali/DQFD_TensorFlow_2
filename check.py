from exp_replay import PartitionedMemory
from utils import load_demo_data_from_file, reward_threshold_subset, RocketProcessor
import numpy as np
processor = RocketProcessor()

expert_demo_data = np.load('demos.npy', allow_pickle=True)
expert_demo_data = reward_threshold_subset(expert_demo_data,0)
# print(expert_demo_data.shape)
expert_demo_data = processor.process_demo_data(expert_demo_data)
WINDOW_LENGTH = 2
memory = PartitionedMemory(1000000, pre_load_data=expert_demo_data, alpha=.6, start_beta=.4, end_beta=.4, window_length=WINDOW_LENGTH)

memory.sample(120000000, 64, 10, 0.99)
