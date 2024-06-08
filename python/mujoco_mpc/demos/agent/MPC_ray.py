# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import numpy as np
import os
import pathlib
from mujoco_viewer import MujocoViewer
import time as time_
# set current directory: mujoco_mpc/python/mujoco_mpc
from mujoco_mpc import agent as agent_lib
import ray

@ray.remote
class RayMPC(object):
    def __init__(self, model_path, worker_id):
        self.worker_id = worker_id
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.viewer = MujocoViewer(
          self.model,self.data,mode='window',title="SMPL",
          width=1200,height=800,hide_menus=True
        )
        self.agent = agent_lib.Agent(task_id="SMPL Track", model=self.model)
        self.init_value()
        pass
    
    def init_value(self):
      # rollout horizon
      self.T = 315

      # trajectories
      self.qpos = np.zeros((self.model.nq, self.T))
      self.qvel = np.zeros((self.model.nv, self.T))
      self.ctrl = np.zeros((self.model.nu, self.T - 1))
      self.time = np.zeros(self.T)

      # costs
      self.cost_total = np.zeros(self.T - 1)
      self.cost_terms = np.zeros((len(self.agent.get_cost_term_values()), self.T - 1))

    def agent_step(self, t):
        
      self.agent.set_state(
        time=self.data.time,
        qpos=self.data.qpos,
        qvel=self.data.qvel,
        act=self.data.act,
        mocap_pos=self.data.mocap_pos,
        mocap_quat=self.data.mocap_quat,
        userdata=self.data.userdata,
      )

      # run planner for num_steps
      num_steps = 1
      for _ in range(num_steps):
        self.agent.planner_step()

      # get costs
      self.cost_total[t] = self.agent.get_total_cost()
      for i, c in enumerate(self.agent.get_cost_term_values().items()):
        self.cost_terms[i, t] = c[1]

      # set ctrl from agent policy
      self.data.ctrl = self.agent.get_action()
      self.ctrl[:, t] = self.data.ctrl

    def loop(self):
      mujoco.mj_resetDataKeyframe(self.model, self.data, self.worker_id*40)

      for t in range(self.T - 1):
        if t % 100 == 0:
          print("t = ", t)

        self.agent_step(t)

        # step
        mujoco.mj_step(self.model, self.data)

        # cache
        self.qpos[:, t + 1] = self.data.qpos
        self.qvel[:, t + 1] = self.data.qvel
        self.time[t + 1] = self.data.time

        self.viewer.render()