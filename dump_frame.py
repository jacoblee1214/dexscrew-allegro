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
    for _ in range(30):
        env.step(torch.zeros((env.num_envs, env.num_actions), device=env.rl_device))
    rb = env.rigid_body_states[0].cpu().numpy()
    base_p, base_q = rb[0,:3], rb[0,3:7]          # 손 base_link
    def q2R(q):
        x,y,z,w=q
        return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                         [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                         [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    R = q2R(base_q)
    to_base = lambda p: R.T @ (p - base_p)
    obj_p = env.object_pos[0].cpu().numpy()
    obj_q = env.object_rot[0].cpu().numpy()
    obj_axis_world = q2R(obj_q) @ np.array([0,0,1.0])   # 드라이버 축(로컬 z 가정)
    # 나사(nut) 바디 위치 = 실제 잡아야 할 지점
    nut_p = rb[env.screw_nut_rb_handle,:3]
    np.savez('/tmp/frame.npz',
             base_p=base_p, base_q=base_q,
             obj_base=to_base(obj_p), nut_base=to_base(nut_p),
             axis_base=R.T @ obj_axis_world,
             ft_base=np.array([to_base(rb[h,:3]) for h in env.fingertip_handles]))
    print("base_p", np.round(base_p,4), "base_q", np.round(base_q,4))
    print("obj(base)", np.round(to_base(obj_p),4))
    print("nut(base)", np.round(to_base(nut_p),4))
    print("axis(base)", np.round(R.T @ obj_axis_world,4))
    print("fingertips(base) [index,mid,pinky,ring,thumb]:")
    for h in env.fingertip_handles: print("  ", np.round(to_base(rb[h,:3]),4))
if __name__=='__main__': main()
