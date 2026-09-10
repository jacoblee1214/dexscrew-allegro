"""손끝이 물체에 닿는지 진단. 학습 아님."""
import isaacgym
import numpy as np, torch, hydra
from omegaconf import DictConfig, OmegaConf
from dexscrew.tasks import isaacgym_task_map
from dexscrew.utils.reformat import omegaconf_to_dict
from dexscrew.utils.misc import set_np_formatting, set_seed
OmegaConf.register_new_resolver('eq', lambda x,y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x,y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda p,a,b: a if p else b)
OmegaConf.register_new_resolver('resolve_default', lambda d,a: d if a=='' else a)

@hydra.main(config_name='config', config_path='configs')
def main(config: DictConfig):
    set_np_formatting(); config.seed = set_seed(config.seed)
    env = isaacgym_task_map[config.task_name](
        config=omegaconf_to_dict(config.task), sim_device=config.sim_device,
        graphics_device_id=config.graphics_device_id, headless=config.headless)
    env.reset()
    for _ in range(20):
        env.step(torch.zeros((env.num_envs, env.num_actions), device=env.rl_device))

    print("[BASE] hand base link world pos =", env.rigid_body_states[0, 0, :3].cpu().numpy())
    print("[BASE] hand base link world quat=", env.rigid_body_states[0, 0, 3:7].cpu().numpy())
    print("[BASE] num_bodies per env =", env.num_bodies, " fingertip_handles =", env.fingertip_handles)
    ft  = env.fingertip_pos[0].cpu().numpy().reshape(-1, 3)   # (5,3)
    obj = env.object_pos[0].cpu().numpy()
    names = list(config.task.env.asset.get('fingertipLinks',
                 ['index','mid','pinky','ring','thumb']))
    print("\n" + "="*64)
    print(f"물체 위치      : {np.round(obj,4)}")
    print(f"손 베이스      : (0, 0, 0.21), X축 {90-25}deg 회전")
    print(f"\n{'손끝':<12}{'world xyz':<34}{'물체까지 거리(mm)':>18}")
    for n, p in zip(names, ft):
        d = np.linalg.norm(p - obj) * 1000
        print(f"{n:<12}{str(np.round(p,4)):<34}{d:>18.1f}")
    # 손 베이스 프레임으로 변환. 베이스는 두 손 모두 p=(0,0,0.21), X축 65deg 회전으로 동일하다.
    th = np.pi/2 - 25*np.pi/180
    R = np.array([[1,0,0],[0,np.cos(th),-np.sin(th)],[0,np.sin(th),np.cos(th)]])
    t = np.array([0.0, 0.0, 0.21])
    ft_b  = (ft - t) @ R          # R^T (p-t) == (p-t) @ R
    obj_b = (obj - t) @ R
    print(f"\n--- 손 베이스 기준 좌표 (이게 손 고유의 기하) ---")
    for n, p_ in zip(names, ft_b):
        print(f"  {n:<12}{np.round(p_,4)}")
    print(f"  {'손끝중심':<12}{np.round(ft_b.mean(axis=0),4)}")
    print(f"  {'물체':<12}{np.round(obj_b,4)}")
    print(f"\n최소 손끝-물체 거리: {np.linalg.norm(ft-obj,axis=1).min()*1000:.1f} mm")
    # ── 접촉 진단: 손끝이 실제로 물체를 잡고 있나 ──
    cf = torch.norm(env.contact_forces, dim=-1)          # (envs, bodies)
    ft_cf  = cf[:, env.fingertip_handles].cpu().numpy()  # (envs, 5)
    obj_cf = cf[:, env.screw_base_rb_handle:env.screw_nut_rb_handle+1].cpu().numpy()
    nut_dof = env.nut_dof_pos.cpu().numpy().ravel()
    print(f"\n--- 접촉 (전 env 평균) ---")
    for n, v in zip(names, ft_cf.mean(0)):
        print(f"  {str(n):<12} 접촉력 {v:8.3f} N   접촉 env 비율 {(ft_cf[:, list(names).index(n)]>0.01).mean()*100:5.1f}%")
    print(f"  물체 3개 바디 접촉력 평균: {np.round(obj_cf.mean(0),3)}")
    print(f"  접촉 중인 손끝 개수(평균): {(ft_cf>0.01).sum(1).mean():.2f} / 5")
    print(f"  나사 관절 각도(rad) 평균: {nut_dof.mean():.5f}")
    rb = env.rigid_body_states[0].cpu().numpy()
    nut = rb[env.screw_nut_rb_handle,:3]
    th = rb[env.fingertip_handles[-1],:3]; ix = rb[env.fingertip_handles[0],:3]
    dt_, di_ = np.linalg.norm(th-nut)*1000, np.linalg.norm(ix-nut)*1000
    thr = env.reset_dist_threshold*1000
    print(f"\n[NUTDIST] 엄지-너트 {dt_:.1f} mm / 검지-너트 {di_:.1f} mm  (종료임계 {thr:.0f} mm)")
    print(f"[NUTDIST] 즉시종료? {'예 ❌' if (dt_>thr or di_>thr) else '아니오 ✅'}"
          f"   proximity_reward = {max(0.0, min(1.0, 1-((dt_+di_)/2)/thr)):.3f}")
    print("="*64)
    np.savez(f"/tmp/diag_{config.task_name}.npz", ft_base=ft_b, obj_base=obj_b,
             ft_world=ft, obj_world=obj, names=np.array(names))
    print(f"저장: /tmp/diag_{config.task_name}.npz")

if __name__ == '__main__':
    main()
