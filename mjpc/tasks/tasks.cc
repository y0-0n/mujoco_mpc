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

#include "mjpc/tasks/tasks.h"

#include <memory>
#include <vector>

#include "mjpc/tasks/acrobot/acrobot.h"
#include "mjpc/tasks/cartpole/cartpole.h"
#include "mjpc/tasks/fingers/fingers.h"
#include "mjpc/tasks/hand/hand.h"
#include "mjpc/tasks/humanoid/stand/stand.h"
#include "mjpc/tasks/humanoid/tracking/tracking.h"
#include "mjpc/tasks/humanoid/walk/walk.h"
#include "mjpc/tasks/panda/panda.h"
#include "mjpc/tasks/manipulation/manipulation.h"
// DEEPMIND INTERNAL IMPORT
#include "mjpc/tasks/particle/particle.h"
#include "mjpc/tasks/quadrotor/quadrotor.h"
#include "mjpc/tasks/quadruped/quadruped.h"
#include "mjpc/tasks/swimmer/swimmer.h"
#include "mjpc/tasks/walker/walker.h"
// RILAB IMPORT
#include "mjpc/tasks/atlas/stand/stand.h"
#include "mjpc/tasks/atlas/motion/motion.h"
#include "mjpc/tasks/smpl/stand/stand.h"
#include "mjpc/tasks/smpl/walk/walk.h"
#include "mjpc/tasks/smpl/motion/motion.h"
#include "mjpc/tasks/common_rig/walk/walk.h"
#include "mjpc/tasks/common_rig/motion/motion.h"
#include "mjpc/tasks/myosuite/walk/walk.h"

// #include "mjpc/tasks/smpl_stand/smpl_stand.h"

namespace mjpc {

std::vector<std::shared_ptr<Task>> GetTasks() {
  return {
    std::make_shared<Acrobot>(),
    std::make_shared<Cartpole>(),
    std::make_shared<Fingers>(),
    std::make_shared<Hand>(),
    std::make_shared<humanoid::Stand>(),
    std::make_shared<humanoid::Tracking>(),
    std::make_shared<humanoid::Walk>(),
    std::make_shared<manipulation::Bring>(),
    // DEEPMIND INTERNAL TASKS
    std::make_shared<Panda>(),
    std::make_shared<Particle>(),
    std::make_shared<ParticleFixed>(),
    std::make_shared<Quadrotor>(),
    std::make_shared<QuadrupedFlat>(),
    std::make_shared<QuadrupedHill>(),
    std::make_shared<Swimmer>(),
    std::make_shared<Walker>(),
    // RILAB TASKS
    std::make_shared<atlas::Stand>(),
    std::make_shared<atlas::Motion>(),
    std::make_shared<smpl::Stand>(),
    std::make_shared<smpl::Walk>(),
    std::make_shared<smpl::Motion>(),
    std::make_shared<common_rig::Walk>(),
    std::make_shared<common_rig::Motion>(),
    std::make_shared<myosuite::Walk>(),
  };
}
}  // namespace mjpc
