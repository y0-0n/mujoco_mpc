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
<<<<<<< HEAD
  int tick = min((this->task_->first_frame + int((data->time - this->task_->reference_time) / 0.002)), 1954); // this->task_->batch_horizon % 299; // int(data->time / 0.0083333);
=======
  int tick = (this->task_->first_frame + int((data->time - this->task_->reference_time) / 0.0083)) % 470; // this->task_->batch_horizon % 299; // int(data->time / 0.0083333);
  tick = max(0, tick);
>>>>>>> e1669ed715cebdc93fd3960e7a8ee78f6b657710
  // cout << tick << endl;
  // this->task_->batch_horizon = 1;

  // // ----- torso height ----- //
  // double torso_height = SensorByName(model, data, "torso_position")[2];
  // // residual[counter++] = torso_height - parameters_[0];

  // // // ----- pelvis / feet ----- //
  // double* foot_right = SensorByName(model, data, "foot_right");
  // double* foot_left = SensorByName(model, data, "foot_left");
  // // double pelvis_height = SensorByName(model, data, "pelvis_position")[2];
  // // residual[counter++] =
  // //     0.5 * (foot_left[2] + foot_right[2]) - pelvis_height - 0.2;

  // // // ----- balance ----- //
  // // capture point
  // double* subcom = SensorByName(model, data, "torso_subcom");
  // double* subcomvel = SensorByName(model, data, "torso_subcomvel");

  // double capture_point[3];
  // mju_addScl(capture_point, subcom, subcomvel, 0.3, 3);
  // capture_point[2] = 1.0e-3;

  // // project onto line segment

  // double axis[3];
  // double center[3];
  // double vec[3];
  // double pcp[3];
  // mju_sub3(axis, foot_right, foot_left);
  // axis[2] = 1.0e-3;
  // double length = 0.5 * mju_normalize3(axis) - 0.05;
  // mju_add3(center, foot_right, foot_left);
  // mju_scl3(center, center, 0.5);
  // mju_sub3(vec, capture_point, center);

  // // project onto axis
  // double t = mju_dot3(vec, axis);

  // // clamp
  // t = mju_max(-length, mju_min(length, t));
  // mju_scl3(vec, axis, t);
  // mju_add3(pcp, vec, center);
  // pcp[2] = 1.0e-3;

  // // is standing
  // double standing =
  //     torso_height / mju_sqrt(torso_height * torso_height + 0.45 * 0.45) - 0.4;

  // mju_sub(&residual[counter], capture_point, pcp, 2);
  // mju_scl(&residual[counter], &residual[counter], standing, 2);

  // counter += 2;

  // // ----- upright ----- //
  // double* torso_up = SensorByName(model, data, "torso_up");
  // double* pelvis_up = SensorByName(model, data, "pelvis_up");
  // double* foot_right_up = SensorByName(model, data, "foot_right_up");
  // double* foot_left_up = SensorByName(model, data, "foot_left_up");
  // double z_ref[3] = {0.0, 0.0, 1.0};

  // // torso
  // residual[counter++] = torso_up[2] - 1.0;

  // // pelvis
  // residual[counter++] = 0.3 * (pelvis_up[2] - 1.0);

  // // right foot
  // mju_sub3(&residual[counter], foot_right_up, z_ref);
  // mju_scl3(&residual[counter], &residual[counter], 0.1 * standing);
  // counter += 3;

  // mju_sub3(&residual[counter], foot_left_up, z_ref);
  // mju_scl3(&residual[counter], &residual[counter], 0.1 * standing);
  // counter += 3;

  // ----- posture ----- //
  // cout << "motion: " << tick << endl;

  double qpos_loss[44];
  for (int i = 0; i < model->nq; i++) {
    // cout << i << ":" << (data->qpos+7)[i] << this->task_->motion_vector[0][i] << endl;
    qpos_loss[i] = abs((data->qpos)[i] - this->task_->motion_vector_qpos[tick][i]);
  }
  mju_copy(&residual[counter], qpos_loss, model->nq);
  counter += model->nq;

  double xpos_loss[21] = {0};
  int body_idx[21] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 31, 32, 33, 34, 55, 56, 57, 58, 59, 60}; // Exclude hand
  // for (int i = 0; i < model->nbody-1; i++) {
  for (int i = 0; i < 21; i++) {
    int idx = body_idx[i];
    xpos_loss[idx] += abs((data->xpos[3*idx+3]) - this->task_->motion_vector_xpos[tick][3*idx]);
    xpos_loss[idx] += abs((data->xpos[3*idx+4]) - this->task_->motion_vector_xpos[tick][3*idx+1]);
    xpos_loss[idx] += abs((data->xpos[3*idx+5]) - this->task_->motion_vector_xpos[tick][3*idx+2]);
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

  // // ----- walk ----- //
  // double* torso_forward = SensorByName(model, data, "torso_forward");
  // double* pelvis_forward = SensorByName(model, data, "pelvis_forward");
  // double* foot_right_forward = SensorByName(model, data, "foot_right_forward");
  // double* foot_left_forward = SensorByName(model, data, "foot_left_forward");

  // double forward[2];
  // mju_copy(forward, torso_forward, 2);
  // mju_addTo(forward, pelvis_forward, 2);
  // mju_addTo(forward, foot_right_forward, 2);
  // mju_addTo(forward, foot_left_forward, 2);
  // mju_normalize(forward, 2);

  // // com vel
  // // double* waist_lower_subcomvel =
  // //     SensorByName(model, data, "waist_lower_subcomvel");
  // // double* torso_velocity = SensorByName(model, data, "torso_velocity");
  // // double com_vel[2];
  // // mju_add(com_vel, waist_lower_subcomvel, torso_velocity, 2);
  // // mju_scl(com_vel, com_vel, 0.5, 2);
  // // double* waist_lower_subcomvel =
  // //     SensorByName(model, data, "waist_lower_subcomvel");
  // double* torso_velocity = SensorByName(model, data, "torso_velocity");
  // double com_vel[2] = {torso_velocity[0], torso_velocity[1]};
  // mju_scl(com_vel, com_vel, 0.5, 2);

  // // walk forward
  // residual[counter++] =
  //     standing * (mju_dot(com_vel, forward, 2) - parameters_[1]);

  // // ----- move feet ----- //
  // double* foot_right_vel = SensorByName(model, data, "foot_right_velocity");
  // double* foot_left_vel = SensorByName(model, data, "foot_left_velocity");
  // double move_feet[2];
  // mju_copy(move_feet, com_vel, 2);
  // mju_addToScl(move_feet, foot_right_vel, -0.5, 2);
  // mju_addToScl(move_feet, foot_left_vel, -0.5, 2);

  // mju_copy(&residual[counter], move_feet, 2);
  // mju_scl(&residual[counter], &residual[counter], standing, 2);
  // counter += 2;

  // // ----- control ----- //
  // mju_copy(&residual[counter], data->ctrl, model->nu);
  // counter += model->nu;

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
