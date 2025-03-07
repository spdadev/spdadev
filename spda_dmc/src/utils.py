import torch
import numpy as np
import os
import glob
import json
import random
import augmentations
import subprocess
import time
from datetime import datetime
from arguments import parse_args
args = parse_args()

global de_num
de_num = args.de_num

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def cat(x, y, axis=0):
    return torch.cat([x, y], axis=0)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def write_info(args, fp):
    data = {
        'timestamp': str(datetime.now()),
        # 'git': subprocess.check_output(["git", "describe", "--always"]).strip().decode(),
        'args': vars(args)
    }
    with open(fp, 'w') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))


def load_config(key=None):
    path = os.path.join('setup', 'config.cfg')
    with open(path) as f:
        data = json.load(f)
    if key is not None:
        return data[key]
    return data


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def listdir(dir_path, filetype='jpg', sort=True):
    fpath = os.path.join(dir_path, f'*.{filetype}')
    fpaths = glob.glob(fpath, recursive=True)
    if sort:
        return sorted(fpaths)
    return fpaths


def prefill_memory(obses, capacity, obs_shape):
    """Reserves memory for replay buffer"""
    c, h, w = obs_shape
    obses = np.ones((capacity,2,c,h,w), dtype=np.uint8)
    return obses


class ReplayBuffer(object):
    """Buffer to store environment transitions"""

    def __init__(self, obs_shape, action_shape, capacity, batch_size, encoder_multi=3, prefill=True):

        self.capacity = capacity
        self.batch_size = batch_size
        # ------------------------------------------------------------------
        self.encoder_multi = encoder_multi
        # ------------------------------------------------------------------
        self._obses = []
        if prefill:    # True
            self._obses = np.array(prefill_memory(self._obses, capacity, obs_shape))
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        obses = (obs, next_obs)
        if self.idx >= len(self._obses):
            self._obses.append(obses)
        else:
            self._obses[self.idx] = (obses)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def _get_idxs(self, n=None):
        if n is None:
            n = self.batch_size
        return np.random.randint(
            0, self.capacity if self.full else self.idx, size=n
        )

    def _encode_obses(self, idxs):
        obses = self._obses[idxs]
        obs = obses[:,0:1,:,:,:].squeeze()
        next_obs = obses[:,1:2,:,:,:].squeeze()
        return obs, next_obs

    def sample_soda(self, n=None):
        idxs = self._get_idxs(n)
        obs, _ = self._encode_obses(idxs)
        return torch.as_tensor(obs).cuda(de_num).float()

    def __sample__(self, n=None, for_encoder=False):
        idxs = self._get_idxs(n)
        # --------------------------------------------------------
        if for_encoder:
            obs, _ = self._encode_obses(idxs)
            obs = torch.as_tensor(obs).cuda(de_num).float()
            actions = torch.as_tensor(self.actions[idxs]).cuda(de_num)
            return obs, actions
        # --------------------------------------------------------
        else:
            obs, next_obs = self._encode_obses(idxs)
            obs = torch.as_tensor(obs).cuda(de_num).float()
            next_obs = torch.as_tensor(next_obs).cuda(de_num).float()
            actions = torch.as_tensor(self.actions[idxs]).cuda(de_num)
            rewards = torch.as_tensor(self.rewards[idxs]).cuda(de_num)
            not_dones = torch.as_tensor(self.not_dones[idxs]).cuda(de_num)

            return obs, actions, rewards, next_obs, not_dones

    def sample_curl(self, n=None):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        pos = augmentations.random_crop(obs.clone())
        obs = augmentations.random_crop(obs)
        next_obs = augmentations.random_crop(next_obs)

        return obs, actions, rewards, next_obs, not_dones, pos

    def sample_drq(self, n=None, pad=4):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        obs = augmentations.random_shift(obs, pad)
        next_obs = augmentations.random_shift(next_obs, pad)

        return obs, actions, rewards, next_obs, not_dones

    def sample_svea(self, n=None, pad=4):
        # obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        # obs = augmentations.random_shift(obs, pad)

        # return obs, actions, rewards, next_obs, not_dones
        return self.sample_drq(n=n, pad=pad)
    
    def sample_encoder(self, n=None, pad=4):
        if self.encoder_multi != 0:
            obs, actions = self.__sample__(n=self.batch_size*self.encoder_multi, for_encoder=True)
            obs = augmentations.random_shift(obs, pad)
            return obs, actions
        else:
            return torch.as_tensor([]).cuda(de_num).float(), torch.as_tensor([]).cuda(de_num).float()
    def sample(self, n=None):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        obs = augmentations.random_crop(obs)
        next_obs = augmentations.random_crop(next_obs)

        return obs, actions, rewards, next_obs, not_dones
    
    def sample_encoder_sac(self, n=None, pad=4):
        if self.encoder_multi != 0:
            obs, actions = self.__sample__(n=self.batch_size*self.encoder_multi, for_encoder=True)
            obs = augmentations.random_crop(obs)
            return obs, actions
        else:
            return torch.as_tensor([]).cuda(de_num).float(), torch.as_tensor([]).cuda(de_num).float()    


class Test_Buffer(object):
    """Buffer to store environment transitions"""

    def __init__(self, obs_shape, action_shape, capacity, prefill=True):
        self.capacity = capacity

        self._obses = []
        if prefill:
            self._obses = prefill_memory(self._obses, capacity, obs_shape)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs):
        obses = (obs, next_obs)
        if self.idx >= len(self._obses):
            self._obses.append(obses)
        else:
            self._obses[self.idx] = (obses)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0


class LazyFrames(object):
    def __init__(self, frames, extremely_lazy=True):
        self._frames = frames
        self._extremely_lazy = extremely_lazy
        self._out = None

    @property
    def frames(self):
        return self._frames

    def _force(self):
        if self._extremely_lazy:
            return np.concatenate(self._frames, axis=0)
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        if self._extremely_lazy:
            return len(self._frames)
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        if self.extremely_lazy:
            return len(self._frames)
        frames = self._force()
        return frames.shape[0]//3

    def frame(self, i):
        return self._force()[i*3:(i+1)*3]


def count_parameters(net, as_int=False):
    """Returns total number of params in a network"""
    count = sum(p.numel() for p in net.parameters())
    if as_int:
        return count
    return f'{count:,}'
