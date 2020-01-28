import numpy as np
import operator

import time

class SegmentTree(object):
    """
    Abstract SegmentTree data structure used to create PrioritizedMemory.
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
    """
    def __init__(self, capacity, operation, neutral_element):

        #powers of two have no bits in common with the previous integer
        assert capacity > 0 and capacity & (capacity - 1) == 0, "Capacity must be positive and a power of 2"
        self._capacity = capacity

        #a segment tree has (2*n)-1 total nodes
        self._value = [neutral_element for _ in range(2 * capacity)]

        self._operation = operation

        self.next_index = 0

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]

class SumSegmentTree(SegmentTree):
    """
    SumTree allows us to sum priorities of transitions in order to assign each a probability of being sampled.
    """
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity

class MinSegmentTree(SegmentTree):
    """
    In PrioritizedMemory, we normalize importance weights according to the maximum weight in the buffer.
    This is determined by the minimum transition priority. This MinTree provides an efficient way to
    calculate that.
    """
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)

def record_demo_data(env_name, steps, frame_delay=0.03, env_seed=123, data_filepath='expert_demo_data.npy'):
    """
    Basic script for recording your own demonstration gameplay in a gym environment. Modified
    from gym keyboard agent.
    """
    import gym
    env = gym.make(env_name)
    np.random.seed(env_seed)
    env.seed(env_seed)
    nb_actions = env.action_space.n

    action = 0
    human_wants_restart = False
    human_sets_pause = False

    def key_press(key, mod):
        nonlocal action, human_sets_pause, human_wants_restart
        if key==0xff0d: human_wants_restart = True
        if key==32: human_sets_pause = not human_sets_pause
        a = int( key - ord('0') )
        if a <= 0 or a >= nb_actions: return
        action = a

    def key_release(key, mod):
        nonlocal action
        a = int( key - ord('0') )
        if a <= 0 or a >= nb_actions: return
        if action == a:
            action = 0

    env.render(mode='human')
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release

    print("ACTIONS={}".format(nb_actions))
    print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
    print("No keys pressed is taking action 0")

    transitions = []
    obser = env.reset()
    total_reward = 0
    total_timesteps = 0

    while total_timesteps < steps:
        if total_timesteps % 1000 == 0:
            print("Steps Elapsed: " + str(total_timesteps))
        act = action
        obs, r, done, info = env.step(act)
        transitions.append([obs, act, r, done])
        total_timesteps += 1
        env.render(mode='human')
        if done:
            env.reset()
        if human_wants_restart:
                transitions = []
                total_timesteps = 0
        while human_sets_pause:
            env.render(mode='human')
            time.sleep(0.1)
        #Gym runs the environments fast by default. Tweak the frame_delay parameter to adjust play speed.
        time.sleep(frame_delay)

    data_matrix = np.array(transitions)
    np.save(data_filepath, data_matrix)
class Processor(object):
    def process_step(self, observation, reward, done, info):
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        return observation, reward, done, info

    def process_observation(self, observation):
        return observation

    def process_reward(self, reward):
        return action

    def process_state_batch(self, batch):
        return batch

    @property
    def metrics(self):
        return []

    @property
    def metrics_names(self):
        return []

class RocketProcessor(Processor):
    def process_observation(self, observation):
        return np.array(observation, dtype='float32')

    def process_state_batch(self, batch):
        return np.array(batch).astype('float32')

    def process_reward(self, reward):
        return np.sign(reward) * np.log(1 + abs(reward))

    def process_demo_data(self, demo_data):
        for step in demo_data:
            step[0] = self.process_observation(step[0])
            step[2] = self.process_reward(step[2])
        return demo_data

def load_demo_data_from_file(data_filepath='expert_demo_data.npy'):
    return np.load(data_filepath)

def calc_ep_rs(demo_array):
    episode_rs = np.zeros(demo_array.shape[0])
    episode_total = 0
    episode_start = 0
    for i, transition in enumerate(demo_array):
        reward = transition[-2]
        reward = np.sign(reward) * np.log(1 + abs(reward))
        episode_total += reward
        if transition[-1]: #terminal
            episode_rs[episode_start : i + 1] = episode_total
            episode_total = 0
            episode_start = i + 1

    return episode_rs

def reward_threshold_subset(demo_array, reward_min):
    rs = calc_ep_rs(demo_array)
    cropped_demos = []
    for i,transition in enumerate(demo_array):
        if rs[i] > reward_min:
            cropped_demos.append(transition)
    return np.array(cropped_demos)