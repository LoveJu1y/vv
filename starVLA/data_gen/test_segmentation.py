"""
简单测试脚本：验证几何分段代码框架
"""
import sys
from pathlib import Path

# 添加路径以便导入
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from starVLA.data_gen.bridge_geometric_segmentation import (
    load_info,
    iter_episode_indices,
    process_episode,
    DEFAULT_BRIDGE_ROOT
)

def test_single_episode(episode_idx: int = 0):
    """测试单个episode的处理"""
    print(f"\n{'='*60}")
    print(f"测试 Episode {episode_idx}")
    print(f"{'='*60}\n")
    
    root = Path(DEFAULT_BRIDGE_ROOT)
    
    # 1. 加载info
    try:
        info = load_info(root)
        chunk_size = int(info.get("chunks_size", 1000))
        print(f"✓ 成功加载 info.json")
        print(f"  - chunk_size: {chunk_size}")
    except Exception as e:
        print(f"✗ 加载 info.json 失败: {e}")
        return False
    
    # 2. 处理单个episode
    try:
        result = process_episode(
            episode_index=episode_idx,
            root=root,
            chunk_size=chunk_size,
            th_open=0.3,
            th_close=0.7
        )
        
        print(f"✓ 成功处理 episode")
        print(f"  - episode_index: {result.get('episode_index')}")
        print(f"  - num_steps: {result.get('num_steps')}")
        print(f"  - segmentation_quality: {result.get('segmentation_quality')}")
        print(f"  - num_cycles: {len(result.get('cycles', []))}")
        
        # 打印每个cycle的详细信息
        for cycle in result.get('cycles', []):
            print(f"\n  Cycle {cycle['cycle_id']}:")
            for seg in cycle['segments']:
                print(f"    - {seg['segment_type']}: [{seg['start_t']}, {seg['end_t']}] "
                      f"(duration={seg['end_t'] - seg['start_t'] + 1})")
                print(f"      debug: {seg['geometry_debug']}")
        
        # 检查dense_labels
        dense = result.get('dense_labels', {})
        if dense:
            subtask_ids = dense.get('subtask_id', [])
            cycle_ids = dense.get('cycle_id', [])
            print(f"\n  Dense Labels:")
            print(f"    - subtask_id length: {len(subtask_ids)}")
            print(f"    - cycle_id length: {len(cycle_ids)}")
            if len(subtask_ids) > 0:
                unique_subtasks = set(subtask_ids)
                print(f"    - unique subtask_ids: {sorted(unique_subtasks)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 处理 episode 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_episodes(num_episodes: int = 3):
    """测试多个episode"""
    print(f"\n{'='*60}")
    print(f"测试 {num_episodes} 个 Episodes")
    print(f"{'='*60}\n")
    
    root = Path(DEFAULT_BRIDGE_ROOT)
    
    try:
        info = load_info(root)
        chunk_size = int(info.get("chunks_size", 1000))
        indices = list(iter_episode_indices(info, "train"))
        
        if len(indices) == 0:
            print("✗ 没有找到可用的episode indices")
            return False
        
        print(f"✓ 找到 {len(indices)} 个episodes，测试前 {min(num_episodes, len(indices))} 个")
        
        success_count = 0
        for i, idx in enumerate(indices[:num_episodes]):
            print(f"\n--- Episode {i+1}/{num_episodes} (index={idx}) ---")
            try:
                result = process_episode(
                    episode_index=idx,
                    root=root,
                    chunk_size=chunk_size,
                    th_open=0.3,
                    th_close=0.7
                )
                
                quality = result.get('segmentation_quality', 'unknown')
                num_cycles = len(result.get('cycles', []))
                num_steps = result.get('num_steps', 0)
                
                print(f"  ✓ 成功 | steps={num_steps} | quality={quality} | cycles={num_cycles}")
                success_count += 1
                
            except Exception as e:
                print(f"  ✗ 失败: {e}")
        
        print(f"\n{'='*60}")
        print(f"测试完成: {success_count}/{num_episodes} 成功")
        print(f"{'='*60}\n")
        
        return success_count == num_episodes
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_idx", type=int, default=0, help="测试单个episode的索引")
    parser.add_argument("--num_episodes", type=int, default=3, help="测试多个episode的数量")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "multiple"])
    args = parser.parse_args()
    
    if args.mode == "single":
        success = test_single_episode(args.episode_idx)
    else:
        success = test_multiple_episodes(args.num_episodes)
    
    sys.exit(0 if success else 1)

