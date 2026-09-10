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
    set_np_formatting(); config.seed=set_seed(config.seed)
    env=isaacgym_task_map[config.task_name](config=omegaconf_to_dict(config.task),
        sim_device=config.sim_device, graphics_device_id=config.graphics_device_id, headless=config.headless)
    env.reset(); nb=env.num_xhand_hand_bodies
    C=[]
    for _ in range(400):
        env.step(torch.zeros((env.num_envs, env.num_actions), device=env.rl_device))
        C.append(torch.norm(env.contact_forces,dim=-1).cpu().numpy())
    C=np.array(C)
    print("\n스텝구간   물체최대N   손링크최대N   리셋수")
    for a in range(0,400,50):
        b=a+50
        print(f"  {a:3d}-{b:3d}  {C[a:b,:,nb:].max():11.0f}  {C[a:b,:,:nb].max():12.0f}")
    bn=env.gym.get_actor_rigid_body_names(env.envs[0],0)
    late=C[200:]; peak=late.mean(axis=(0,1))
    print("\n후반 200스텝 접촉 상위 5:")
    for i in np.argsort(-peak)[:5]:
        nm=bn[i] if i<len(bn) else f"[obj]{i-len(bn)}"
        print(f"  {nm:<12}{peak[i]:9.1f} N")
    print(f"\nepisodeLength={config.task.env.episodeLength}")
if __name__=='__main__': main()
