"""
可视化分段结果：展示原始轨迹和分段边界（简化版，无 B-Spline）
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 添加路径以便导入
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from starVLA.data_gen.bridge_geometric_segmentation import (
    load_info,
    load_episode,
    derive_gripper_signal,
    binarize_gripper,
    detect_events,
    process_episode,
    DEFAULT_BRIDGE_ROOT,
    PRIMITIVE_TYPES,
    SEGMENT_ID_MAP
)

# 定义颜色映射
SEGMENT_COLORS = {
    "move_to_object": "#3498db",  # 蓝色
    "move_to_goal": "#2ecc71",    # 绿色
    "place_object": "#f39c12",    # 橙色
}

def visualize_episode(
    episode_idx: int,
    output_path: str = None,
    th_open: float = 0.3,
    th_close: float = 0.7,
):
    """可视化单个 episode 的分段结果"""
    
    root = Path(DEFAULT_BRIDGE_ROOT)
    info = load_info(root)
    chunk_size = int(info.get("chunks_size", 1000))
    
    # 1. 加载原始数据
    ep_data = load_episode(root, episode_idx, chunk_size)
    state, action = ep_data["state"], ep_data["action"]
    T = state.shape[0]
    
    # 2. 提取特征
    pos = state[:, :3].astype(np.float32)  # XYZ position
    z = pos[:, 2]
    
    # 计算速度（位置变化）
    vel = np.linalg.norm(np.diff(pos, axis=0, prepend=pos[[0]]), axis=1)
    
    # 提取夹爪信号
    gripper = derive_gripper_signal(state, action)
    
    # 3. 运行分段
    result = process_episode(
        episode_index=episode_idx,
        root=root,
        chunk_size=chunk_size,
        th_open=th_open,
        th_close=th_close
    )
    
    cycles = result.get("cycles", [])
    dense_labels = result.get("dense_labels", {})
    subtask_ids = dense_labels.get("subtask_id", [0] * T)
    
    # 4. 创建图形
    fig = plt.figure(figsize=(16, 10))
    time = np.arange(T)
    
    # 4.1 位置 XYZ
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(time, pos[:, 0], 'b-', label='X', alpha=0.7, linewidth=1.5)
    ax1.plot(time, pos[:, 1], 'g-', label='Y', alpha=0.7, linewidth=1.5)
    ax1.plot(time, pos[:, 2], 'r-', label='Z', alpha=0.7, linewidth=1.5)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Position')
    ax1.set_title('Position (XYZ)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    _add_segment_background(ax1, cycles, T)
    
    # 4.2 速度
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(time, vel, 'b-', label='Velocity', alpha=0.7, linewidth=1.5)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Velocity')
    ax2.set_title('Velocity (Position Change)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    _add_segment_background(ax2, cycles, T)
    
    # 4.3 夹爪状态
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(time, gripper, 'purple', label='Gripper (Normalized)', alpha=0.7, linewidth=1.5)
    ax3.axhline(y=th_open, color='green', linestyle='--', label=f'Open Threshold ({th_open})', linewidth=1.5)
    ax3.axhline(y=th_close, color='red', linestyle='--', label=f'Close Threshold ({th_close})', linewidth=1.5)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Gripper Value')
    ax3.set_title('Gripper State (1=Open, 0=Closed)')
    ax3.set_ylim(-0.1, 1.1)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    _add_segment_background(ax3, cycles, T)
    _add_gripper_events(ax3, gripper, th_open, th_close, T)
    
    # 4.4 分段标签（Dense Labels）
    ax4 = plt.subplot(3, 2, 4)
    colors_list = [SEGMENT_COLORS.get(PRIMITIVE_TYPES[sid], 'gray') for sid in subtask_ids]
    ax4.bar(time, subtask_ids, color=colors_list, alpha=0.6, width=1.0)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Subtask ID')
    ax4.set_title('Dense Segmentation Labels')
    ax4.set_yticks(range(len(PRIMITIVE_TYPES)))
    ax4.set_yticklabels(PRIMITIVE_TYPES)
    ax4.grid(True, alpha=0.3, axis='y')
    _add_segment_boundaries(ax4, cycles, T)
    
    # 4.5 3D 轨迹可视化
    ax5 = plt.subplot(3, 2, 5, projection='3d')
    ax5.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', alpha=0.5, linewidth=1.5, label='Trajectory')
    # 标记分段点
    for cycle in cycles:
        for seg in cycle['segments']:
            start, end = seg['start_t'], seg['end_t']
            if start < T and end < T:
                color = SEGMENT_COLORS.get(seg['segment_type'], 'gray')
                ax5.scatter(pos[start:end+1, 0], pos[start:end+1, 1], pos[start:end+1, 2],
                          c=color, s=20, alpha=0.7, label=seg['segment_type'] if start == seg['start_t'] else '')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')
    ax5.set_title('3D Trajectory with Segmentation')
    ax5.legend(loc='best', fontsize=8)
    
    # 4.6 分段信息表格
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    info_text = f"Episode {episode_idx} | Steps: {T} | Quality: {result.get('segmentation_quality', 'unknown')}\n"
    info_text += f"Cycles: {len(cycles)}\n\n"
    
    for cycle in cycles:
        info_text += f"Cycle {cycle['cycle_id']}:\n"
        for seg in cycle['segments']:
            duration = seg['end_t'] - seg['start_t'] + 1
            debug = seg.get('geometry_debug', {})
            info_text += f"  {seg['segment_type']:20s} [{seg['start_t']:3d}, {seg['end_t']:3d}] "
            info_text += f"(dur={duration:2d})\n"
        info_text += "\n"
    
    ax6.text(0.1, 0.9, info_text, transform=ax6.transAxes, 
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 可视化已保存到: {output_path}")
    else:
        plt.show()
    
    plt.close()

def _add_segment_background(ax, cycles, T):
    """在图上添加分段背景色"""
    for cycle in cycles:
        for seg in cycle['segments']:
            start, end = seg['start_t'], seg['end_t']
            color = SEGMENT_COLORS.get(seg['segment_type'], 'gray')
            ax.axvspan(start, end + 1, alpha=0.15, color=color)

def _add_segment_boundaries(ax, cycles, T):
    """添加分段边界线"""
    for cycle in cycles:
        for seg in cycle['segments']:
            start, end = seg['start_t'], seg['end_t']
            ax.axvline(x=start, color='black', linestyle='--', alpha=0.5, linewidth=1)
            ax.axvline(x=end, color='black', linestyle='--', alpha=0.5, linewidth=1)

def _add_gripper_events(ax, gripper, th_open, th_close, T):
    """标记夹爪事件点"""
    g_bin = binarize_gripper(gripper, th_open, th_close)
    opens, closes = detect_events(g_bin)  # opens=0->1 (place), closes=1->0 (grasp)
    
    # 标记抓取事件（1->0，红色）
    for t in closes:
        if 0 <= t < T:
            ax.axvline(x=t, color='red', linestyle=':', linewidth=2, alpha=0.7)
            ax.scatter(t, gripper[t], color='red', s=100, marker='v', zorder=5, label='Grasp' if t == closes[0] else '')
    
    # 标记放置事件（0->1，绿色）
    for t in opens:
        if 0 <= t < T:
            ax.axvline(x=t, color='green', linestyle=':', linewidth=2, alpha=0.7)
            ax.scatter(t, gripper[t], color='green', s=100, marker='^', zorder=5, label='Place' if t == opens[0] else '')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="可视化分段结果（简化版）")
    parser.add_argument("--episode_idx", type=int, default=1, help="Episode 索引")
    parser.add_argument("--output", type=str, default=None, help="输出图片路径（如不指定则显示）")
    parser.add_argument("--th_open", type=float, default=0.3)
    parser.add_argument("--th_close", type=float, default=0.7)
    
    args = parser.parse_args()
    
    visualize_episode(
        episode_idx=args.episode_idx,
        output_path=args.output,
        th_open=args.th_open,
        th_close=args.th_close
    )
