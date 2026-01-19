"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""

from typing import Dict, List, Tuple


# Dataset mixture name mapped to a list of tuples containing:
## {nakename: [(data_name, sampling_weight, robot_type)] }
DATASET_NAMED_MIXTURES = {

    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
                # ("libero_90_no_noops_lerobot", 1.0, "libero_franka"),
    ],
    "libero_goal": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_object": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_spatial": [
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_10": [
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_90": [
        ("libero_90_no_noops_lerobot", 1.0, "libero_franka"),
        # ("libero_90_no_noops_lerobot", 1.0, "libero_ur5"),
    ],

    "bridge": [
        ("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
    ],
    "bridge_local": [
        ("bridge_orig_lerobot", 1.0, "oxe_bridge"),
    ],
    "bridge_rt_1": [
        ("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
        ("fractal20220817_data_0.1.0_lerobot", 1.0, "oxe_rt1"),
    ],
    "demo_sim_pick_place": [
        ("sim_pick_place", 1.0, "demo_sim_franka_delta_joints"),
    ],

    "agilex_cobot_magic_real4": [
        ("Agilex_Cobot_Magic_classify_object_fruit", 1.0, "agilex_cobot_magic"),
        ("Agilex_Cobot_Magic_pour_water_twice", 1.0, "agilex_cobot_magic"),
        ("Agilex_Cobot_Magic_stack_block_twice", 1.0, "agilex_cobot_magic"),
        ("Agilex_Cobot_Magic_storage_object_two", 1.0, "agilex_cobot_magic"),
    ],
    "agilex_cobot_magic_fruit": [
        ("Agilex_Cobot_Magic_classify_object_fruit", 1.0, "agilex_cobot_magic"),
    ],
    "agilex_cobot_magic_pour": [
        ("Agilex_Cobot_Magic_pour_water_twice", 1.0, "agilex_cobot_magic"),
    ],
    "agilex_cobot_magic_stack": [
        ("Agilex_Cobot_Magic_stack_block_twice", 1.0, "agilex_cobot_magic"),
    ],
    "agilex_cobot_magic_storage": [
        ("Agilex_Cobot_Magic_storage_object_two", 1.0, "agilex_cobot_magic"),
    ],

    "custom_dataset": [
        ("custom_dataset_name", 1.0, "custom_robot_config"),
    ],
    "custom_dataset_2": [
        ("custom_dataset_name_1", 1.0, "custom_robot_config"),
        ("custom_dataset_name_2", 1.0, "custom_robot_config"),
    ],

    "BEHAVIOR_challenge": [
        ("BEHAVIOR_challenge", 1.0, "R1Pro"),
    ],


}
