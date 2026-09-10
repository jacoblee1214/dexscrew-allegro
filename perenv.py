"""env별 물체 스케일 vs 엄지 접촉 — 파지가 스케일에 얼마나 민감한지."""
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
    acc = []
    for _ in range(120):
        env.step(torch.zeros((env.num_envs, env.num_actions), device=env.rl_device))
        cf = torch.norm(env.contact_forces, dim=-1)
        acc.append(np.concatenate([cf[:, env.fingertip_handles].cpu().numpy(),
                                   cf[:, [env.screw_nut_rb_handle]].cpu().numpy()], axis=1))
    A = np.array(acc)                                  # (T, envs, 5)
    scales = [env.gym.get_actor_scale(env.envs[i], 0) if hasattr(env.gym,'get_actor_scale')
              else np.nan for i in range(env.num_envs)]
    # 스케일은 priv buf 에 기록됨
    rb = env.rigid_body_states.cpu().numpy()
    nut = rb[:, env.screw_nut_rb_handle, :3]
    th  = rb[:, env.fingertip_handles[-1], :3]
    ix  = rb[:, env.fingertip_handles[0],  :3]
    nut_f = A[:,:,-1]                                  # 물체(손잡이) 접촉력
    print(f"\n{'env':>4}{'물체접촉%':>11}{'평균힘N':>10}{'엄지-너트mm':>13}{'검지-너트mm':>13}")
    for i in range(env.num_envs):
        print(f"{i:>4}{(nut_f[:,i]>0.01).mean()*100:>11.1f}{nut_f[:,i].mean():>10.2f}"
              f"{np.linalg.norm(th[i]-nut[i])*1000:>13.1f}{np.linalg.norm(ix[i]-nut[i])*1000:>13.1f}")
    ratio = (nut_f>0.01).mean(0)
    print(f"\n[요약] 물체 접촉 시간비율: 평균 {ratio.mean()*100:.1f}%  "
          f"| 전 시간 접촉 env {int((ratio>0.95).sum())}/{env.num_envs}  "
          f"| 접촉 전무 env {int((ratio<0.05).sum())}/{env.num_envs}")
if __name__=='__main__': main()
