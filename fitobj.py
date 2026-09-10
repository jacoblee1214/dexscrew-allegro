"""파지 중심에 물체 축이 오도록 objectOffset 을 보정한다."""
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
    for _ in range(5):
        env.step(torch.zeros((env.num_envs, env.num_actions), device=env.rl_device))
    rb = env.rigid_body_states.cpu().numpy()
    # 파지중심 = 엄지끝 + (검지끝+중지끝)/2 의 중점
    th = rb[:, env.fingertip_handles[-1], :3]
    ix = rb[:, env.fingertip_handles[0],  :3]
    md = rb[:, env.fingertip_handles[1],  :3]
    gc = (th + (ix+md)/2)/2                       # (envs,3)
    obj = env.object_pos.cpu().numpy()
    q = env.object_rot.cpu().numpy()
    def q2R(qq):
        x,y,z,w=qq
        return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                         [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                         [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    # 각 env 에서 파지중심과 같은 높이의 축 위 점을 구하고, 그 수평 편차를 평균
    d=[]
    for i in range(env.num_envs):
        ax = q2R(q[i]) @ np.array([0,0,1.0])
        t = (gc[i]-obj[i]) @ ax
        onaxis = obj[i] + t*ax
        d.append(gc[i]-onaxis)
    d=np.array(d).mean(0)
    cur = list(config.task.env.get('objectOffset',[0.009,0.06,0.0]))
    new = [cur[0]+float(d[0]), cur[1]+float(d[1]), cur[2]]
    print(f"\n파지중심 평균 {np.round(gc.mean(0),4)}   물체축과의 수평편차 {np.round(d*1000,1)} mm")
    print(f"현재 objectOffset {np.round(cur,4).tolist()}")
    print(f"보정 objectOffset [{new[0]:.4f},{new[1]:.4f},{new[2]:.4f}]")
if __name__=='__main__': main()
