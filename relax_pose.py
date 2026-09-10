"""초기 자세를 시뮬에서 안정화시켜 '도달 가능한' 자세로 바꾼다.
목표(target)와 실제(q)가 어긋나 있으면 PD 가 영구히 토크를 뿜고, 그게 제곱합 벌점을 지배한다."""
import isaacgym
import numpy as np, torch, hydra
from omegaconf import DictConfig, OmegaConf
from dexscrew.tasks import isaacgym_task_map
from dexscrew.utils.reformat import omegaconf_to_dict
from dexscrew.utils.misc import set_np_formatting, set_seed
for n,f in [('eq',lambda x,y:x.lower()==y.lower()),('contains',lambda x,y:x.lower() in y.lower()),
            ('if',lambda p,a,b:a if p else b),('resolve_default',lambda d,a:d if a=='' else a)]:
    OmegaConf.register_new_resolver(n,f)
@hydra.main(config_name='config', config_path='configs')
def main(config: DictConfig):
    set_np_formatting(); config.seed = set_seed(config.seed)
    env = isaacgym_task_map[config.task_name](
        config=omegaconf_to_dict(config.task), sim_device=config.sim_device,
        graphics_device_id=config.graphics_device_id, headless=config.headless)
    env.reset()
    n_settle = int(config.get('settle', 300))
    for _ in range(n_settle):
        env.step(torch.zeros((env.num_envs, env.num_actions), device=env.rl_device))
    q = env.xhand_hand_dof_pos.cpu().numpy()          # (envs, dofs)
    names = env.gym.get_asset_dof_names(env.hand_asset)
    med = np.median(q, axis=0)                        # env 노이즈에 강건하게 중앙값
    tgt = env.cur_targets[:, :env.num_xhand_hand_dofs].cpu().numpy()
    err = np.abs(tgt - q).mean(0)
    print(f"\n[안정화 전] 위치오차 평균 {err.mean():.4f} rad  최대 {err.max():.4f} ({names[int(err.argmax())]})")
    tq = env.torques[:, :, :env.num_xhand_hand_dofs].abs().mean(1).cpu().numpy().mean(0)
    print(f"[안정화 전] 토크 제곱합 {(tq**2).sum():.4f}")
    print("\n안정화된 관절각 (새 initPoseValues):")
    print("'task.env.initPoseValues=[" + ",".join(f"{v:.4f}" for v in med) + "]'")
    print("\n관절별 변화 (목표 -> 안정화):")
    for i in np.argsort(-err)[:6]:
        print(f"   {names[i]:<12}{tgt[0][i]:>9.4f} -> {med[i]:>9.4f}   (오차였던 값 {err[i]:.4f})")
if __name__=='__main__': main()
