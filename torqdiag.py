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
    T = []
    for _ in range(60):
        env.step(torch.zeros((env.num_envs, env.num_actions), device=env.rl_device))
        T.append(env.torques[:, :, :env.num_xhand_hand_dofs].abs().mean(1).cpu().numpy())
    T = np.array(T)                     # (t, envs, dofs)
    names = env.gym.get_asset_dof_names(env.hand_asset)
    m = T.mean(axis=(0,1))
    print(f"\n관절별 평균 |토크| (Nm)   총합 {m.sum():.3f}   제곱합 {(m**2).sum():.3f}")
    for i in np.argsort(-m)[:8]:
        print(f"   {names[i]:<32}{m[i]:8.4f}")
    print(f"   ... 나머지 {len(m)-8}개 합 {np.sort(m)[:-8].sum():.4f}")
    q = env.xhand_hand_dof_pos.cpu().numpy()
    tgt = env.cur_targets[:, :env.num_xhand_hand_dofs].cpu().numpy()
    err = np.abs(tgt - q).mean(0)
    print(f"\n위치오차 |target-q| 평균 {err.mean():.4f} rad  최대 {err.max():.4f} ({names[int(err.argmax())]})")
if __name__=='__main__': main()
