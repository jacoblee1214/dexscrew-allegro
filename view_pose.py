"""
파지 자세를 뷰어로 관찰. 학습 아님 — 액션 0 으로 초기 파지를 그대로 유지한다.
반드시 GUI 세션(원격 데스크톱) 터미널에서 실행할 것. SSH 터미널은 DISPLAY 가 없어 안 뜬다.

뷰어 조작: 마우스 드래그=시점 회전 / 휠=줌 / ESC=종료 / V=렌더 토글
"""
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
    act = torch.zeros((env.num_envs, env.num_actions), device=env.rl_device)

    steps = int(config.get('view_steps', 0))     # 0 = 무한 (ESC 로 종료)
    i = 0
    while steps == 0 or i < steps:
        env.step(act)                            # 액션 0 = 초기 파지 유지
        if i % 200 == 0:
            cf = torch.norm(env.contact_forces, dim=-1)
            obj_cf = cf[:, env.screw_base_rb_handle:env.screw_nut_rb_handle+1].mean(0)
            ft = env.fingertip_pos[0].cpu().numpy().reshape(-1,3)
            d = np.linalg.norm(ft - env.object_pos[0].cpu().numpy(), axis=1) * 1000
            print(f"[{i:6d}] 물체접촉력 {obj_cf.cpu().numpy().round(1)}  "
                  f"손끝거리(mm) {np.round(d,1)}", flush=True)
        i += 1
    print("종료")

if __name__ == '__main__':
    main()
