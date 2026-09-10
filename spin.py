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
    for _ in range(10):
        env.step(torch.zeros((env.num_envs, env.num_actions), device=env.rl_device))
    q = env.object_rot[0].cpu().numpy()
    def q2R(q):
        x,y,z,w=q
        return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                         [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                         [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    R = q2R(q)
    z_local_world = R @ np.array([0,0,1.0])      # 드라이버 축(손잡이 방향)
    joint_axis_world = R @ np.array([0,0,-1.0])  # URDF joint axis = (0,0,-1)
    print("\n[실측]")
    print(f"  드라이버 local +z 의 월드 방향 : {np.round(z_local_world,3)}   (z성분 {z_local_world[2]:+.3f})")
    print(f"  나사 관절축(0,0,-1)의 월드 방향: {np.round(joint_axis_world,3)}   (z성분 {joint_axis_world[2]:+.3f})")
    print(f"  → 보상은 이 축 방향 각속도가 양수일 때 주어짐")
if __name__=='__main__': main()
