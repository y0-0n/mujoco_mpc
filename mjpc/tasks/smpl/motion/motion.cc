// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/tasks/smpl/motion/motion.h"

#include <iostream>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

#include <iostream>

using namespace std;

namespace mjpc::smpl {
std::string Motion::XmlPath() const {
  return GetModelPath("smpl/motion/task.xml");
}
std::string Motion::Name() const { return "SMPL Motion"; }

// ------------------ Residuals for SMPL motion task ------------
//   Number of residuals:
//     Residual (0): torso height
//     Residual (1): pelvis-feet aligment
//     Residual (2): balance
//     Residual (3): upright
//     Residual (4): posture
//     Residual (5): walk
//     Residual (6): move feet
//     Residual (7): control
//   Number of parameters:
//     Parameter (0): torso height goal
//     Parameter (1): speed goal
// ----------------------------------------------------------------
void Motion::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                double* residual) const {
  int counter = 0;
  int tick = min((this->task_->first_frame + max(int((data->time - this->task_->reference_time) / 0.002), 0)), 1954); // this->task_->batch_horizon % 299; // int(data->time / 0.0083333);
  // cout << tick << endl;
  // this->task_->batch_horizon = 1;

  double qpos_loss[42];
  for (int i = 2; i < model->nq; i++) {
    // cout << i << ":" << (data->qpos+7)[i] << this->task_->motion_vector[0][i] << endl;
    qpos_loss[i-2] = abs((data->qpos)[i] - this->task_->motion_vector_qpos[tick][i]);
  }
  mju_copy(&residual[counter], qpos_loss, model->nq-2);
  counter += model->nq-2;

  double xpos_loss[21] = {0};
  int body_idx[21] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 31, 32, 33, 34, 55, 56, 57, 58, 59, 60}; // Exclude hand
  // for (int i = 0; i < model->nbody-1; i++) {
  for (int i = 0; i < 21; i++) {
    int idx = body_idx[i];
    if (idx == 57 || idx == 60) {
      if (data->xpos[3*idx+5] <= 0.14) {
          xpos_loss[idx] = 0;
          continue;
      }
    }
    // xpos_loss[idx] += abs((data->xpos[3*idx+3]) - this->task_->motion_vector_xpos[tick][3*idx]);
    // xpos_loss[idx] += abs((data->xpos[3*idx+4]) - this->task_->motion_vector_xpos[tick][3*idx+1]);
    xpos_loss[i] += abs((data->xpos[3*idx+5]) - this->task_->motion_vector_xpos[tick][3*idx+2]);
  }
  // TODO: fix hard coding (n_body=61)
  mju_scl(xpos_loss, xpos_loss, 1./3., 21);
  mju_copy(&residual[counter], xpos_loss, 21);
  counter += 21;
  

  // ----- joint velocity ----- //
  double qvel_loss[43];
  for (int i = 0; i < model->nq; i++) {
    // cout << i << ":" << (data->qpos+7)[i] << this->task_->motion_vector[0][i] << endl;
    qvel_loss[i] = abs((data->qvel)[i] - this->task_->motion_vector_qvel[tick][i]);
  }
  mju_copy(&residual[counter], qvel_loss, model->nv);
  counter += model->nv;

  // sensor dim sanity check
  // TODO: use this pattern everywhere and make this a utility function
  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error_i(
        "mismatch between total user-sensor dimension "
        "and actual length of residual %d",
        counter);
  }
}

}  // namespace mjpc::smpl
