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

#include "mjpc/tasks/smpl/tracking/tracking.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <string>
#include <tuple>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace {
// compute interpolation between mocap frames
std::tuple<int, int, double, double> ComputeInterpolationValues(double index,
                                                                int max_index) {
  int index_0 = std::floor(std::clamp(index, 0.0, (double)max_index));
  int index_1 = std::min(index_0 + 1, max_index);

  double weight_1 = std::clamp(index, 0.0, (double)max_index) - index_0;
  double weight_0 = 1.0 - weight_1;

  return {index_0, index_1, weight_0, weight_1};
}

// Hardcoded constant matching keyframes from CMU mocap dataset.
constexpr double kFps = 120.0;

constexpr int kMotionLengths[] = {
    315,  // Jump - CMU-CMU-02-02_04
    // 154,  // Kick Spin - CMU-CMU-87-87_01
    // 115,  // Spin Kick - CMU-CMU-88-88_06
    // 78,   // Cartwheel (1) - CMU-CMU-88-88_07
    // 145,  // Crouch Flip - CMU-CMU-88-88_08
    // 188,  // Cartwheel (2) - CMU-CMU-88-88_09
    // 260,  // Monkey Flip - CMU-CMU-90-90_19
    // 279,  // Dance - CMU-CMU-103-103_08
    // 39,   // Run - CMU-CMU-108-108_13
    // 510,  // Walk - CMU-CMU-137-137_40
};

// return length of motion trajectory
int MotionLength(int id) { return kMotionLengths[id]; }

// return starting keyframe index for motion
int MotionStartIndex(int id) {
  int start = 0;
  for (int i = 0; i < id; i++) {
    start += MotionLength(i);
  }
  return start;
}

// names for smpl bodies
const std::array<std::string, 15> body_names = {
    "root", "spine2", "head", 
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pelvis","right_pelvis",
    "left_knee",  "right_knee",
    "left_ankle", "right_ankle"
};

const std::array<std::string, 37> joint_names = {
    "root1", "root2", "root3",
    "spine1", "spine2", "spine3",
    "head1", "head2", "head3", 
    "l_shoulder1", "l_shoulder2", "l_shoulder3",
    "r_shoulder1", "r_shoulder2", "r_shoulder3",
    "l_elbow","r_elbow",
    "l_wrist1", "l_wrist2", "l_wrist3",
    "r_wrist1", "r_wrist2", "r_wrist3",
    "l_pelvis1", "l_pelvis2", "l_pelvis3",
    "r_pelvis1", "r_pelvis2", "r_pelvis3",
    "l_knee","r_knee",
    "l_ankle1", "l_ankle2", "l_ankle3",
    "r_ankle1", "r_ankle2", "r_ankle3"
};

}  // namespace

namespace mjpc::smpl {

std::string Tracking::XmlPath() const {
  return GetModelPath("smpl/tracking/task.xml");
}
std::string Tracking::Name() const { return "SMPL Track"; }

// ------------- Residuals for smpl tracking task -------------
//   Number of residuals:
//     Residual (0): Joint vel: minimise joint velocity
//     Residual (1): Control: minimise control
//     Residual (2-11): Tracking position: minimise tracking position error
//         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
//     Residual (11-20): Tracking velocity: minimise tracking velocity error
//         for {root, head, toe, heel, knee, hand, elbow, shoulder, hip}.
//   Number of parameters: 0
// ----------------------------------------------------------------
void Tracking::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                  double *residual) const {
  // ----- get mocap frames ----- //
  // get motion start index
  int start = MotionStartIndex(current_mode_);
  // get motion trajectory length
  int length = MotionLength(current_mode_);
  double current_index = (data->time - reference_time_) * kFps + start;
  int last_key_index = start + length - 1;

  // Positions:
  // We interpolate linearly between two consecutive key frames in order to
  // provide smoother signal for tracking.
  int key_index_0, key_index_1;
  double weight_0, weight_1;
  std::tie(key_index_0, key_index_1, weight_0, weight_1) =
      ComputeInterpolationValues(current_index, last_key_index);

  // ----- residual ----- //
  int counter = 0;

  // ----- joint velocity ----- //
  // mju_copy(residual + counter, data->qvel + 6, model->nv - 6);
  // counter += model->nv - 6;

  // ----- action ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;

  // ----- task space position ----- //
  // Compute interpolated frame.
  auto get_body_mpos = [&](const std::string &body_name, double result[3]) {
    std::string mocap_body_name = "mocap[" + body_name + "]";
    int mocap_body_id = mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str());
    assert(0 <= mocap_body_id);
    int body_mocapid = model->body_mocapid[mocap_body_id];
    assert(0 <= body_mocapid);

    // current frame
    mju_scl3(
        result,
        model->key_mpos + model->nmocap * 3 * key_index_0 + 3 * body_mocapid,
        weight_0);

    // next frame
    mju_addToScl3(
        result,
        model->key_mpos + model->nmocap * 3 * key_index_1 + 3 * body_mocapid,
        weight_1);
  };

  auto get_body_sensor_pos = [&](const std::string &body_name,
                                 double result[3]) {
    std::string pos_sensor_name = "tracking_pos[" + body_name + "]";
    double *sensor_pos = SensorByName(model, data, pos_sensor_name.c_str());
    mju_copy3(result, sensor_pos);
  };

  // compute marker and sensor averages
  double avg_mpos[3] = {0};
  double avg_sensor_pos[3] = {0};
  int num_body = 0;
  for (const auto &body_name : body_names) {
    double body_mpos[3];
    double body_sensor_pos[3];
    get_body_mpos(body_name, body_mpos);
    mju_addTo3(avg_mpos, body_mpos);
    get_body_sensor_pos(body_name, body_sensor_pos);
    mju_addTo3(avg_sensor_pos, body_sensor_pos);
    num_body++;
  }
  mju_scl3(avg_mpos, avg_mpos, 1.0/num_body);
  mju_scl3(avg_sensor_pos, avg_sensor_pos, 1.0/num_body);

  // residual for averages
  mju_sub3(&residual[counter], avg_mpos, avg_sensor_pos);
  counter += 3;

  for (const auto &body_name : body_names) {
    double body_mpos[3];
    get_body_mpos(body_name, body_mpos);

    // current position
    double body_sensor_pos[3];
    get_body_sensor_pos(body_name, body_sensor_pos);

    mju_subFrom3(body_mpos, avg_mpos);
    mju_subFrom3(body_sensor_pos, avg_sensor_pos);

    mju_sub3(&residual[counter], body_mpos, body_sensor_pos);

    if (body_name == "right_ankle" || body_name == "left_ankle") {
      if(body_sensor_pos[2] < 0.03 && body_mpos[2] < 0.03) {
        residual[counter+2] = 0.0;
      }
    }

    counter += 3;
  }


  // ----- joint space position ----- //
  // Compute interpolated frame.
  auto get_joint_qpos = [&](const std::string &joint_name, double result[1]) {
    int joint_id = mj_name2id(model, mjOBJ_JOINT, joint_name.data());
    // int body_id = model->jnt_bodyid[joint_id];
    // std::string body_name = mj_id2name(model, mjOBJ_BODY, body_id);
    // std::string mocap_body_name = "mocap[" + body_name + "]";
    // int mocap_body_id = mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str());
    // assert(0 <= mocap_body_id);
    // int body_mocapid = model->body_mocapid[mocap_body_id];
    assert(0 <= joint_id);

    // current frame
    mju_scl(
        result,
        model->key_qpos + model->nq * key_index_0 + (joint_id + 6),
        weight_0,
        1);

    // next frame
    mju_addToScl(
        result,
        model->key_qpos + model->nq * key_index_1 + (joint_id + 6),
        weight_1,
        1);
  };

  auto get_joint_sensor_qpos = [&](const std::string &joint_name,
                                 double result[1]) {
    std::string pos_sensor_name = "joint_pos[" + joint_name + "]";
    double *sensor_pos = SensorByName(model, data, pos_sensor_name.c_str());
    mju_copy(result, sensor_pos, 1);
  };

  // compute marker and sensor averages
  // double avg_qpos[1] = {0};
  // double avg_sensor_qpos[1] = {0};
  // int num_joint = 0;
  // for (const auto &joint_name : joint_names) {
  //   double joint_qpos[1];
  //   double joint_sensor_qpos[1];
  //   get_joint_qpos(joint_name, joint_qpos);
  //   mju_addTo(avg_qpos, joint_qpos, 1);
  //   get_joint_sensor_qpos(joint_name, joint_sensor_qpos);
  //   mju_addTo(avg_sensor_qpos, joint_sensor_qpos, 1);
  //   num_joint++;
  // }
  // mju_scl(avg_qpos, avg_qpos, 1.0/num_joint, 1);
  // mju_scl(avg_sensor_qpos, avg_sensor_qpos, 1.0/num_joint, 1);

  // // residual for averages
  // mju_sub(&residual[counter], avg_qpos, avg_sensor_qpos, 1);
  // counter += 1;

  for (const auto &joint_name : joint_names) {
    double joint_qpos[1];
    get_joint_qpos(joint_name, joint_qpos);

    // current position
    double joint_sensor_qpos[1];
    get_joint_sensor_qpos(joint_name, joint_sensor_qpos);

    // mju_subFrom(joint_qpos, avg_qpos, 1);
    // mju_subFrom(joint_sensor_qpos, avg_sensor_qpos, 1);

    mju_sub(&residual[counter], joint_qpos, joint_sensor_qpos, 1);

    counter += 1;
  }
  
  // ----- velocity ----- //
  // for (const auto &body_name : body_names) {
  //   std::string mocap_body_name = "mocap[" + body_name + "]";
  //   std::string linvel_sensor_name = "tracking_linvel[" + body_name + "]";
  //   int mocap_body_id = mj_name2id(model, mjOBJ_BODY, mocap_body_name.c_str());
  //   assert(0 <= mocap_body_id);
  //   int body_mocapid = model->body_mocapid[mocap_body_id];
  //   assert(0 <= body_mocapid);

  //   // compute finite-difference velocity
  //   mju_copy3(
  //       &residual[counter],
  //       model->key_mpos + model->nmocap * 3 * key_index_1 + 3 * body_mocapid);
  //   mju_subFrom3(
  //       &residual[counter],
  //       model->key_mpos + model->nmocap * 3 * key_index_0 + 3 * body_mocapid);
  //   mju_scl3(&residual[counter], &residual[counter], kFps);

  //   // subtract current velocity
  //   double *sensor_linvel =
  //       SensorByName(model, data, linvel_sensor_name.c_str());
  //   mju_subFrom3(&residual[counter], sensor_linvel);

  //   counter += 3;
  // }


  CheckSensorDim(model, counter);
}

// --------------------- Transition for humanoid task -------------------------
//   Set `data->mocap_pos` based on `data->time` to move the mocap sites.
//   Linearly interpolate between two consecutive key frames in order to
//   smooth the transitions between keyframes.
// ----------------------------------------------------------------------------
void Tracking::TransitionLocked(mjModel *model, mjData *d) {
  // get motion start index
  int start = MotionStartIndex(mode);
  // get motion trajectory length
  int length = MotionLength(mode);

  // check for motion switch
  if (residual_.current_mode_ != mode || d->time == 0.0) {
    residual_.current_mode_ = mode;       // set motion id
    residual_.reference_time_ = d->time;  // set reference time

    // set initial state
    mju_copy(d->qpos, model->key_qpos + model->nq * start, model->nq);
    mju_copy(d->qvel, model->key_qvel + model->nv * start, model->nv);
  }

  // indices
  double current_index = (d->time - residual_.reference_time_) * kFps + start;
  int last_key_index = start + length - 1;

  // Positions:
  // We interpolate linearly between two consecutive key frames in order to
  // provide smoother signal for tracking.
  int key_index_0, key_index_1;
  double weight_0, weight_1;
  std::tie(key_index_0, key_index_1, weight_0, weight_1) =
      ComputeInterpolationValues(current_index, last_key_index);

  mj_markStack(d);

  mjtNum *mocap_pos_0 = mj_stackAllocNum(d, 3 * model->nmocap);
  mjtNum *mocap_pos_1 = mj_stackAllocNum(d, 3 * model->nmocap);

  // Compute interpolated frame.
  mju_scl(mocap_pos_0, model->key_mpos + model->nmocap * 3 * key_index_0,
          weight_0, model->nmocap * 3);

  mju_scl(mocap_pos_1, model->key_mpos + model->nmocap * 3 * key_index_1,
          weight_1, model->nmocap * 3);

  mju_copy(d->mocap_pos, mocap_pos_0, model->nmocap * 3);
  mju_addTo(d->mocap_pos, mocap_pos_1, model->nmocap * 3);

  mj_freeStack(d);
}

}  // namespace mjpc::humanoid
