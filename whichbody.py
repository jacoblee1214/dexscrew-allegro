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
    names = env.gym.get_actor_rigid_body_names(env.envs[0], 0)
    env.reset()
    peak = np.zeros(env.num_bodies); first = None
    for t in range(120):
        env.step(torch.zeros((env.num_envs, env.num_actions), device=env.rl_device))
        cf = torch.norm(env.contact_forces, dim=-1).cpu().numpy()   # (envs, bodies)
        peak = np.maximum(peak, cf.max(0))
        if first is None: first = cf.max(0).copy()
    lbl = list(names) + ['[obj]base','[obj]shaft','[obj]handle']
    order = np.argsort(-peak)
    print(f"\n{'body':<16}{'1스텝후 N':>12}{'120스텝 최대 N':>16}")
    for i in order[:10]:
        nm = lbl[i] if i < len(lbl) else f'body{i}'
        print(f"{nm:<16}{first[i]:>12.1f}{peak[i]:>16.1f}")
    print(f"\n손 링크 최대 {peak[:len(names)].max():.1f} N   물체 최대 {peak[len(names):].max():.1f} N")
if __name__=='__main__': main()
