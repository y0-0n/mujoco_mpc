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

#include "mjpc/tasks/smpl/stand/stand.h"

#include <string>
#include <iostream>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"


namespace mjpc::smpl {

std::string Stand::XmlPath() const {
  return GetModelPath("smpl/stand/task.xml");
}
std::string Stand::Name() const { return "SMPL Stand"; }

// ------------------ Residuals for smpl stand task ------------
//   Number of residuals: 6
//     Residual (0): Desired height
//     Residual (1): Balance: COM_xy - average(feet position)_xy
//     Residual (2): Com Vel: should be 0 and equal feet average vel
//     Residual (3): Control: minimise control
//     Residual (4): Joint vel: minimise joint velocity
//   Number of parameters: 1
//     Parameter (0): height_goal
// ----------------------------------------------------------------
void Stand::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                 double* residual) const {
  using namespace std;

  int counter = 0;

  // ----- Height: head feet vertical error ----- //

  // feet sensor positions
  double* f1_position = SensorByName(model, data, "sp0");
  double* f2_position = SensorByName(model, data, "sp1");
  double* f3_position = SensorByName(model, data, "sp2");
  double* f4_position = SensorByName(model, data, "sp3");
  double* head_position = SensorByName(model, data, "head_position");
  double head_feet_error =
      head_position[2] - 0.25 * (f1_position[2] + f2_position[2] +
                                 f3_position[2] + f4_position[2]);
  // cout << head_feet_error << ',' << parameters_[0] << endl;
  residual[counter++] = head_feet_error - parameters_[0];
  

  // ----- Balance: CoM-feet xy error ----- //

  // capture point
  double* com_position = SensorByName(model, data, "torso_subtreecom");
  double* com_velocity = SensorByName(model, data, "torso_subtreelinvel");
  double kFallTime = 0.2;
  double capture_point[3] = {com_position[0], com_position[1], com_position[2]};
  // cout << com_position[2] << endl <<
  //  com_velocity[2] << endl <<
  //  kFallTime << endl <<
  //  capture_point[2] << endl;

  mju_addToScl3(capture_point, com_velocity, kFallTime);
  // cout << capture_point[0] << ", " << capture_point[1]  << ", " <<  capture_point[2] << endl;
  // average feet xy position
  double fxy_avg[2] = {0.0};
  mju_addTo(fxy_avg, f1_position, 2);
  mju_addTo(fxy_avg, f2_position, 2);
  mju_addTo(fxy_avg, f3_position, 2);
  mju_addTo(fxy_avg, f4_position, 2);
  mju_scl(fxy_avg, fxy_avg, 0.25, 2);
  // cout << "fxy_avg:" << fxy_avg[0] << ", " << fxy_avg[1] << ", " <<  endl
  //     << "capture_point xy:" << capture_point[0] << ", " << capture_point[1] << ", " <<  endl;

  mju_subFrom(fxy_avg, capture_point, 2);
  double com_feet_distance = mju_norm(fxy_avg, 2);
  residual[counter++] = com_feet_distance;

  // ----- COM xy velocity should be 0 ----- //
  mju_copy(&residual[counter], com_velocity, 2);
  counter += 2;

  // ----- joint velocity ----- //
  mju_copy(residual + counter, data->qvel + 6, model->nv - 6);
  // for (int i = 0; i < model->nu ; i++) {
  //     cout << "joint vel" << i << ":" << data->qvel[i+6] << endl;
  // }
  counter += model->nv - 6;

  // ----- action ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  // for (int i = 0; i < model->nu ; i++) {
  //     cout << "ctrl" << i << ":" << data->ctrl[i] << endl;
  // }
  counter += model->nu;

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
