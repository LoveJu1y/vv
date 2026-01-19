#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

import os
import cv2
import sys
sys.path.append("./")

import time
import math
import torch
import pickle
import argparse
import threading
import numpy as np
import collections
from collections import deque
from einops import rearrange

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from utils import compute_dict_mean, set_seed, detach_dict # helper functions

from typing import Dict
from PIL import Image as PIL_Image
from starVLA.model.tools import read_mode_config
from starVLA.model.framework.base_framework import baseframework
from starVLA.dataloader.gr00t_lerobot.transform.state_action import Normalizer

task_config = {'camera_names': ['cam_high', 'cam_right_wrist']}

inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None


def actions_interpolation(args, pre_action, actions, stats):
    steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['qpos_std'] + stats['qpos_mean']
    result = [pre_action]
    post_action = post_process(actions[0])
    max_diff_index = 0
    max_diff = -1
    for i in range(post_action.shape[0]):
        diff = 0
        for j in range(pre_action.shape[0]):
            if j == 6 or j == 13:
                continue
            diff += math.fabs(pre_action[j] - post_action[i][j])
        if diff > max_diff:
            max_diff = diff
            max_diff_index = i

    for i in range(max_diff_index, post_action.shape[0]):
        step = max([math.floor(math.fabs(result[-1][j] - post_action[i][j])/steps[j]) for j in range(pre_action.shape[0])])
        inter = np.linspace(result[-1], post_action[i], step+2)
        result.extend(inter[1:])
    while len(result) < args.chunk_size+1:
        result.append(result[-1])
    result = np.array(result)[1:args.chunk_size+1]
    # print("actions_interpolation2:", result.shape, result[:, 7:])
    result = pre_process(result)
    result = result[np.newaxis, :]
    return result


def get_image(observation, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(observation['images'][cam_name], 'h w c -> c h w')
    
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def get_depth_image(observation, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_images.append(observation['images_depth'][cam_name])
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def unnormalize_actions_q01_q99_with_gripper(
    normalized_actions: np.ndarray,
    action_norm_stats: Dict[str, np.ndarray],
    gripper_idxs=(6, 13),
    gripper_open_value=0.08,
    threshold=0.5,
) -> np.ndarray:
    """
    normalized_actions: shape [T, D] or [D], values roughly in [-1, 1]
    action_norm_stats: {"q01": ..., "q99": ..., "mask": ...}
    gripper dims: set to {0, 0.08} by thresholding *before* unnormalize
    """
    x = np.asarray(normalized_actions, dtype=np.float32)
    orig_was_1d = (x.ndim == 1)
    if orig_was_1d:
        x = x[None, :]  # [1, D]

    q01 = np.asarray(action_norm_stats["q01"], dtype=np.float32)
    q99 = np.asarray(action_norm_stats["q99"], dtype=np.float32)
    mask = np.asarray(action_norm_stats.get("mask", np.ones_like(q01, dtype=bool)), dtype=bool).copy()

    # 1) clip 到 [-1, 1]
    # x = np.clip(x, -1.0, 1.0)

    # 2) 先做 gripper 二值映射（注意：这里在 normalized space 上阈值）
    for idx in gripper_idxs:
        x[:, idx] = np.where(x[:, idx] > threshold, gripper_open_value, 0.0)
        mask[idx] = False  # 关键：防止后面反归一化把它覆盖掉

    # 3) 其余维度做 q01-q99 反归一化
    actions = np.where(
        mask[None, :],
        0.5 * (x + 1.0) * (q99[None, :] - q01[None, :]) + q01[None, :],
        x,
    )

    return actions[0] if orig_was_1d else actions


def inference_process(args, config, ros_operator, policy, stats, t, pre_action):
    global inference_lock
    global inference_actions
    global inference_timestep
    print_flag = True
    pre_pos_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    pre_action_process = lambda next_action: (next_action - stats["action_mean"]) / stats["action_std"]
    rate = rospy.Rate(args.publish_rate)
    while True and not rospy.is_shutdown():
        result = ros_operator.get_frame()
        if not result:
            if print_flag:
                print("syn fail")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
         puppet_arm_left, puppet_arm_right, robot_base) = result
        obs = collections.OrderedDict()
        image_dict = dict()

        image_dict[config['camera_names'][0]] = img_front
        image_dict[config['camera_names'][1]] = img_left
        image_dict[config['camera_names'][2]] = img_right

        obs['images'] = image_dict

        if args.use_depth_image:
            image_depth_dict = dict()
            image_depth_dict[config['camera_names'][0]] = img_front_depth
            image_depth_dict[config['camera_names'][1]] = img_left_depth
            image_depth_dict[config['camera_names'][2]] = img_right_depth
            obs['images_depth'] = image_depth_dict

        obs['qpos'] = np.concatenate(
            (np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
        obs['qvel'] = np.concatenate(
            (np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)), axis=0)
        obs['effort'] = np.concatenate(
            (np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort)), axis=0)
        if args.use_robot_base:
            obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
            obs['qpos'] = np.concatenate((obs['qpos'], obs['base_vel']), axis=0)
        else:
            obs['base_vel'] = [0.0, 0.0]
        # qpos_numpy = np.array(obs['qpos'])

        # 归一化处理qpos 并转到cuda
        qpos = pre_pos_process(obs['qpos'])
        qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

        # 当前图像curr_image获取图像
        curr_image = get_image(obs, config['camera_names'])

        start_time = time.time()
        all_actions = policy(curr_image, qpos)
        end_time = time.time()
        print("model cost time: ", end_time -start_time)
        inference_lock.acquire()
        inference_actions = all_actions.cpu().detach().numpy()
        if pre_action is None:
            pre_action = obs['qpos']
        # print("obs['qpos']:", obs['qpos'][7:])
        if args.use_actions_interpolation:
            inference_actions = actions_interpolation(args, pre_action, inference_actions, stats)
        inference_timestep = t
        inference_lock.release()
        break


def check_unnorm_key(norm_stats, unnorm_key):
    """
    Duplicate helper (retained for backward compatibility).
    See primary _check_unnorm_key above.
    """
    if unnorm_key is None:
        assert len(norm_stats) == 1, (
            f"Your model was trained on more than one dataset, "
            f"please pass a `unnorm_key` from the following options to choose the statistics "
            f"used for un-normalizing actions: {norm_stats.keys()}")
        unnorm_key = next(iter(norm_stats.keys()))

    assert unnorm_key in norm_stats, (
        f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
        f"please choose from: {norm_stats.keys()}")
    return unnorm_key


def ensure_pil_at_size(img, size=(224, 224)):
    """
    统一输入格式：如果是 Numpy (OpenCV)，转为 PIL 并转换颜色空间；
    如果是 PIL，直接 resize。
    """
    # 1. 如果是 Numpy 数组 (OpenCV 格式)
    if isinstance(img, np.ndarray):
        # 检查是否需要 BGR -> RGB (假设 OpenCV 默认 BGR)
        # 如果你确定输入已经是 RGB，可以去掉这一步
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = PIL_Image.fromarray(rgb_img)
    
    # 2. 如果已经是 PIL 对象
    elif isinstance(img, PIL_Image.Image):
        pil_img = img
        
    else:
        raise ValueError(f"不支持的图片格式: {type(img)}")

    # 3. 统一进行 resize
    return pil_img.resize(size, resample=PIL_Image.BILINEAR)


# -----------------------------
# Implicit reasoning prompt utils (stage4-only, simplified)
# -----------------------------
# Stage4 default: BBOX(1) + SUBTASK(1) + REASON(1) => 3 thinking tokens.
THINKING_START_TOKEN = "<|start_of_thinking|>"
THINKING_TOKEN = "<|thinking|>"
THINKING_END_TOKEN = "<|end_of_thinking|>"
THINKING_TOKEN_COUNT_STAGE4 = 3

IMG_NEXT_TOKEN = "<img_next>"
IMG_NEXT_COUNT_DEFAULT = 16


def format_instruction_with_implicit_reasoning_stage4(instruction: str) -> str:
    """
    Fixed stage4 prompt:
      "<instruction>. @ <start><thinking>*3<end> <img_next>*16"
    """
    prompt = (instruction or "").strip()
    if not prompt:
        return prompt
    span = f"{THINKING_START_TOKEN}{THINKING_TOKEN * THINKING_TOKEN_COUNT_STAGE4}{THINKING_END_TOKEN}"
    img_next_span = IMG_NEXT_TOKEN * IMG_NEXT_COUNT_DEFAULT
    return f"{prompt}. @ {span} {img_next_span}".strip()

    
def model_inference(args, ros_operator):
    global inference_lock
    global inference_actions
    global inference_timestep
    global inference_thread
    set_seed(1000)

    # 1 创建模型、统计值
    # model_path = "/media/agilex/Entertain/starvla/pretrained_models_real_world/muti-task/BAAI_real_world_2B/starvla_qwen_gr00t/checkpoints/steps_20000_pytorch_model.pt"
    # model_path = "/media/agilex/Entertain/starvla/pretrained_models_real_world/single-task/BAAI_real_world_pour_water_2B/starvla_qwen_gr00t/checkpoints/steps_15000_pytorch_model.pt"
    # model_path = "/media/agilex/Entertain/starvla/pretrained_models_real_world/single-task/BAAI_real_world_storage_banana_right_2B/starvla_qwen_gr00t/checkpoints/steps_15000_pytorch_model.pt"
    model_path = "/media/agilex/Entertain/starvla/pretrained_models_real_world/single-task/BAAI_real_world_storage_banana_right_2B/starvla_qwen_oft/checkpoints/steps_10000_pytorch_model.pt"
    infer_model = baseframework.from_pretrained(model_path)
    # instruction = ["Erase the whiteboard with an eraser"]
    # instruction = ["Fold the towel with dual arms"]
    # instruction = ["Pour water into the cup"]
    # instruction = ["Stack block"]
    instruction = ["Put the banana into the basket"]

    # instruction = ["Put all fruits into the basket"]
    # instruction = ["Pour water into two cups"]
    # instruction = ["Stack two blocks"]
    # instruction = ["Put all objects into the basket"]

    model_config, norm_stats = read_mode_config(model_path)
    unnorm_key = None
    unnorm_key = check_unnorm_key(norm_stats, unnorm_key)
    action_norm_stats = norm_stats[unnorm_key]["action"]
    # normalizer_action = Normalizer('q99', action_norm_stats)

    # ---- Implicit reasoning (ECOT) inference knobs (aligned with SimplerEnv) ----
    cot_mode = getattr(args, "cot_mode", "implicit") or "implicit"
    cot_mode = str(cot_mode).lower()
    think_max_len = int(getattr(args, "think_max_len", 64))
    think_temp = float(getattr(args, "think_temp", 0.1))
    think_topp = float(getattr(args, "think_topp", 0.9))

    # Build implicit-reasoning formatted instructions only when cot_mode=implicit (stage4-only, fixed tokens).
    if cot_mode == "implicit":
        instruction = [format_instruction_with_implicit_reasoning_stage4(ins) for ins in instruction]
    
    # 2 模型设置为cuda模式和验证模式
    infer_model.cuda()
    infer_model.eval()

    # 3. 初始位姿
    left0 = [0.00000000e+00, -9.52466670e-03, -5.05592346e-01,  4.17794436e-02, 1.11417663e+00, -3.92500013e-02, 0.08]
    right0 = [5.22809997e-02, -8.98388866e-03, -5.12866676e-01, -7.93373361e-02,  1.07642686e+00,  6.07066648e-03, 0.08]
    
    ros_operator.puppet_arm_publish_continuous(right0, left0, )

    # import pandas as pd
    # df = pd.read_parquet("/home/agilex/cobot_magic/aloha-devel_2/episode_000000.parquet")
    # actions = np.stack(df["action"].values, axis=0)
    
    # 推理
    t = 0
    rate = rospy.Rate(args.publish_rate)
    with torch.inference_mode():
        while True and not rospy.is_shutdown():
            # start_time = time.time()       
            result = ros_operator.get_frame()
            (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth, puppet_arm_left, puppet_arm_right, robot_base) = result

            img_front = ensure_pil_at_size(img_front)
            img_left  = ensure_pil_at_size(img_left)
            img_right = ensure_pil_at_size(img_right)
            
            batch_images = [[img_front, img_left, img_right]]

            # Explicitly pass inference mode flags (server-side SimplerEnv does the same).
            pred_out = infer_model.predict_action(
                batch_images,
                instruction,
                cot_mode=cot_mode,
                use_iterative_forward=(cot_mode == "implicit"),
                emit_thinking_tokens=False,
                think_max_len=think_max_len,
                think_temp=think_temp,
                think_topp=think_topp,
            )
            action_pred = pred_out["normalized_actions"][0]
            raw_actions = unnormalize_actions_q01_q99_with_gripper(action_pred, action_norm_stats)
            raw_actions = raw_actions[:25]

            # actions_chunk = actions[t: t+25]
            for action in np.array(raw_actions):
            # for action in np.array(actions_chunk):
                # left_action = action[:7].copy() 
                right_action = action[7:14].copy()
                left_action = [0., 0., 0., 0., 0., 0., 0.08]
                # input("Enter any key to continue:")
                ros_operator.puppet_arm_publish(right_action, left_action)  # puppet_arm_publish_continuous_thread
                t += 1
                # print("publish: ", t)
                # print("left_action:", left_action)
                # print("right_action:", right_action)
                rate.sleep()
            # end_time = time.time()
            # print("time:", end_time - start_time)


class RosOperator:
    def __init__(self, args):
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.puppet_arm_left_publisher = None
        self.puppet_arm_right_publisher = None
        self.robot_base_publisher = None
        self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_lock = None
        self.args = args
        self.ctrl_state = False
        self.ctrl_state_lock = threading.Lock()
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()

    def puppet_arm_publish(self, left, right):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
        joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
        joint_state_msg.position = left
        self.puppet_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.position = right
        self.puppet_arm_right_publisher.publish(joint_state_msg)

    def robot_base_publish(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def puppet_arm_publish_continuous(self, left, right):
        rate = rospy.Rate(self.args.publish_rate)
        left_arm = None
        right_arm = None
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            for i in range(len(left)):
                if left_diff[i] < self.args.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            for i in range(len(right)):
                if right_diff[i] < self.args.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.args.arm_steps_length[i]
                    flag = True
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
            joint_state_msg.position = left_arm
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = right_arm
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            step += 1
            print("puppet_arm_publish_continuous:", step)
            rate.sleep()

    def puppet_arm_publish_linear(self, left, right):
        num_step = 100
        rate = rospy.Rate(200)

        left_arm = None
        right_arm = None

        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break

        traj_left_list = np.linspace(left_arm, left, num_step)
        traj_right_list = np.linspace(right_arm, right, num_step)

        for i in range(len(traj_left_list)):
            traj_left = traj_left_list[i]
            traj_right = traj_right_list[i]
            traj_left[-1] = left[-1]
            traj_right[-1] = right[-1]
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()  # 设置时间戳
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
            joint_state_msg.position = traj_left
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = traj_right
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            rate.sleep()

    def puppet_arm_publish_continuous_thread(self, left, right):
        if self.puppet_arm_publish_thread is not None:
            self.puppet_arm_publish_lock.release()
            self.puppet_arm_publish_thread.join()
            self.puppet_arm_publish_lock.acquire(False)
            self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_thread = threading.Thread(target=self.puppet_arm_publish_continuous, args=(left, right))
        self.puppet_arm_publish_thread.start()

    def get_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0 or \
                (self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)):
            return False
        if self.args.use_depth_image:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec(),
                              self.img_left_depth_deque[-1].header.stamp.to_sec(), self.img_right_depth_deque[-1].header.stamp.to_sec(), self.img_front_depth_deque[-1].header.stamp.to_sec()])
        else:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec()])

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_depth_image and (len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            return False

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        img_left_depth = None
        if self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')

        img_front_depth = None
        if self.args.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')

        robot_base = None
        if self.args.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                puppet_arm_left, puppet_arm_right, robot_base)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def ctrl_callback(self, msg):
        self.ctrl_state_lock.acquire()
        self.ctrl_state = msg.data
        self.ctrl_state_lock.release()

    def get_ctrl_state(self):
        self.ctrl_state_lock.acquire()
        state = self.ctrl_state
        self.ctrl_state_lock.release()
        return state

    def init_ros(self):
        rospy.init_node('joint_state_publisher', anonymous=True)
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        if self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)
        self.puppet_arm_left_publisher = rospy.Publisher(self.args.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(self.args.puppet_arm_right_cmd_topic, JointState, queue_size=10)
        self.robot_base_publisher = rospy.Publisher(self.args.robot_base_cmd_topic, Twist, queue_size=10)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', default='', required=False)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', default='aloha_mobile_dummy', required=False)
    parser.add_argument('--max_publish_step', action='store', type=int, help='max_publish_step', default=10000, required=False)
    parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', default='policy_best.ckpt', required=False)
    parser.add_argument('--ckpt_stats_name', action='store', type=str, help='ckpt_stats_name', default='dataset_stats.pkl', required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', default='ACT', required=False)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=8, required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0, required=False)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', default=2000, required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', default=1e-5, required=False)
    parser.add_argument('--weight_decay', type=float, help='weight_decay', default=1e-4, required=False)
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)", required=False)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features", required=False)
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', default=10, required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512, required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200, required=False)
    parser.add_argument('--temporal_agg', action='store', type=bool, help='temporal_agg', default=True, required=False)

    parser.add_argument('--state_dim', action='store', type=int, help='state_dim', default=14, required=False)
    parser.add_argument('--lr_backbone', action='store', type=float, help='lr_backbone', default=1e-5, required=False)
    parser.add_argument('--backbone', action='store', type=str, help='backbone', default='resnet18', required=False)
    parser.add_argument('--loss_function', action='store', type=str, help='loss_function l1 l2 l1+l2', default='l1', required=False)
    parser.add_argument('--enc_layers', action='store', type=int, help='enc_layers', default=4, required=False)
    parser.add_argument('--dec_layers', action='store', type=int, help='dec_layers', default=7, required=False)
    parser.add_argument('--nheads', action='store', type=int, help='nheads', default=8, required=False)
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer", required=False)
    parser.add_argument('--pre_norm', action='store_true', required=False)

    parser.add_argument('--img_front_topic', action='store', type=str, help='img_front_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    
    parser.add_argument('--img_front_depth_topic', action='store', type=str, help='img_front_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)
    
    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, help='puppet_arm_left_cmd_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, help='puppet_arm_right_cmd_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom_raw', required=False)
    parser.add_argument('--robot_base_cmd_topic', action='store', type=str, help='robot_base_topic',
                        default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base',
                        default=False, required=False)
    parser.add_argument('--publish_rate', action='store', type=int, help='publish_rate',
                        default=40, required=False)
    parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step',
                        default=0, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size',
                        default=32, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, help='arm_steps_length',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)

    parser.add_argument('--use_actions_interpolation', action='store', type=bool, help='use_actions_interpolation',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image',
                        default=False, required=False)

    # for Diffusion
    parser.add_argument('--observation_horizon', action='store', type=int, help='observation_horizon', default=1, required=False)
    parser.add_argument('--action_horizon', action='store', type=int, help='action_horizon', default=8, required=False)
    parser.add_argument('--num_inference_timesteps', action='store', type=int, help='num_inference_timesteps', default=10, required=False)
    parser.add_argument('--ema_power', action='store', type=int, help='ema_power', default=0.75, required=False)
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    ros_operator = RosOperator(args)
    model_inference(args, ros_operator)


if __name__ == '__main__':
    main()

# python act/inference.py