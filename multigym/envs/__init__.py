# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Imports all environments so that they register themselves with the Gym API.

This protocol is the same as Minigrid, and allows all environments to be
simultaneously registered with Gym as a package.
"""

# Import all environments and register them, so pylint: disable=wildcard-import
from .adversarial import *
from .cluttered import *
from .coingame import *
from .doorkey import *
from .empty import *
from .fourrooms import *
from .gather import *
from .lava_walls import *
from .maze import *
from .meetup import *
from .stag_hunt import *
from .tag import *
from .tasklist import *
