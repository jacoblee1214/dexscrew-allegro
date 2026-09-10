"""손 위치를 옮기고 -> 이완 -> 엄지-너트 거리 측정. 고정점 반복으로 목표 76mm 를 맞춘다."""
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
    for _ in range(300):
        env.step(torch.zeros((env.num_envs, env.num_actions), device=env.rl_device))
    q = env.xhand_hand_dof_pos.cpu().numpy(); med = np.median(q, axis=0)
    rb = env.rigid_body_states.cpu().numpy()
    nut = rb[:, env.screw_nut_rb_handle, :3]
    th  = rb[:, env.fingertip_handles[-1], :3]; ix = rb[:, env.fingertip_handles[0], :3]
    dt_ = np.linalg.norm(th-nut,axis=1).mean()*1000; di_ = np.linalg.norm(ix-nut,axis=1).mean()*1000
    tq = env.torques[:,:,:env.num_xhand_hand_dofs].abs().mean(1).cpu().numpy().mean(0)
    cf = torch.norm(env.contact_forces,dim=-1).cpu().numpy()
    prox = max(0.0, min(1.0, 1-((dt_+di_)/2)/(env.reset_dist_threshold*1000)))
    print(f"RESULT|{dt_:.1f}|{di_:.1f}|{(tq**2).sum():.4f}|{prox:.3f}|{cf[:,0].max():.0f}|"
          + ",".join(f"{v:.4f}" for v in med))
if __name__=='__main__': main()
