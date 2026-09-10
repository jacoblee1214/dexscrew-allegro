"""
XHand 초기 파지 자세 녹화 (학습 아님, 정책 없음).

액션이 델타 방식(targets = prev_targets + scale*action)이라 액션 0 = 초기 파지 유지.
Allegro 이식 후 같은 스크립트로 찍어서 파지 자세를 눈으로 비교하는 게 목적.

주의: CUDA_VISIBLE_DEVICES 를 쓰면 IsaacGym 그래픽 장치 매핑이 깨진다
      (cudaExternalMemoryGetMappedBuffer error 101). sim_device/graphics_device_id 로 직접 지정할 것.
"""
import isaacgym
import os, numpy as np, torch, hydra, imageio
from omegaconf import DictConfig, OmegaConf
from dexscrew.tasks import isaacgym_task_map
from dexscrew.utils.reformat import omegaconf_to_dict
from dexscrew.utils.misc import set_np_formatting, set_seed

OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)

@hydra.main(config_name='config', config_path='configs')
def main(config: DictConfig):
    set_np_formatting()
    config.seed = set_seed(config.seed)
    n_steps  = int(config.get('rec_steps', 150))
    out_path = str(config.get('rec_out', 'outputs/videos/xhand_baseline.mp4'))

    env = isaacgym_task_map[config.task_name](
        config=omegaconf_to_dict(config.task),
        sim_device=config.sim_device,
        graphics_device_id=config.graphics_device_id,
        headless=config.headless,
    )
    env.reset()

    # 저장소의 _create_camera 는 카메라를 (0,0.2,0.75)->(0,0,0.5) 로 고정해서 손이 너무 작게 나온다.
    # 핸들을 저장하지 않으므로 env당 첫 카메라 = 핸들 0 을 직접 다시 조준한다.
    from isaacgym import gymapi
    # env 마다 카메라를 하나씩 만들어 타일로 붙인다 (저장소 기본 카메라는 256x256 고정 + env 1개뿐)
    W, H = int(config.get('rec_w', 480)), int(config.get('rec_h', 480))
    d = float(config.get('rec_dist', 0.13))
    cams = []
    objs = env.object_pos.cpu().numpy()
    # (와이드 1대 촬영은 불가: IsaacGym 카메라 센서는 부착된 env 의 액터만 렌더한다.
    #  전체를 한눈에 보려면 headless=False 뷰어를 쓸 것. 녹화는 아래 타일 방식으로 대체.)
    for i in range(env.num_envs):
        cp = gymapi.CameraProperties(); cp.width, cp.height = W, H; cp.enable_tensors = False
        c = env.gym.create_camera_sensor(env.envs[i], cp)
        o = objs[i]
        env.gym.set_camera_location(
            c, env.envs[i],
            gymapi.Vec3(float(o[0]) + d, float(o[1]) + d, float(o[2]) + d * 0.7),
            gymapi.Vec3(float(o[0]), float(o[1]), float(o[2])))
        cams.append(c)
    ncol = int(config.get('rec_cols', 3))
    nrow = int(np.ceil(env.num_envs / ncol))
    print(f"카메라 {env.num_envs}개 x {W}x{H}, 타일 {ncol}x{nrow}")

    def grab():
        env.gym.render_all_camera_sensors(env.sim)
        ims = []
        for i, c in enumerate(cams):
            im = env.gym.get_camera_image(env.sim, env.envs[i], c, gymapi.IMAGE_COLOR)
            ims.append(im.reshape(H, W, 4)[:, :, :3].copy())
        while len(ims) < ncol * nrow:
            ims.append(np.zeros_like(ims[0]))
        rows = [np.concatenate(ims[r*ncol:(r+1)*ncol], axis=1) for r in range(nrow)]
        return np.concatenate(rows, axis=0)

    frames, traj = [], []
    for i in range(n_steps):
        act = torch.zeros((env.num_envs, env.num_actions), device=env.rl_device)  # 0 = 파지 유지
        env.step(act)
        frames.append(grab())
        traj.append(env.object_pos.clone().cpu().numpy())
        if i % 30 == 0:
            print(f"  frame {i}/{n_steps}")

    # 파지 안정성: 액션 0 상태에서 물체가 얼마나 흘렀나 (Allegro 이식 후 비교 기준)
    t = np.array(traj)                      # (steps, num_envs, 3)
    drift = np.linalg.norm(t[-1] - t[0], axis=-1) * 1000.0
    print(f"\n--- 파지 안정성 (액션 0, {n_steps} 스텝) ---")
    print(f"  env별 물체 이동량(mm): {np.round(drift,2).tolist()}")
    print(f"  평균 {drift.mean():.2f} mm / 최대 {drift.max():.2f} mm")
    print(f"  낙하(z 하락 20mm 초과) env 수: {int(((t[0,:,2]-t[-1,:,2])*1000 > 20).sum())} / {t.shape[1]}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    import subprocess, tempfile, shutil
    tmpd = tempfile.mkdtemp()
    for i, fr in enumerate(frames):
        imageio.imwrite(os.path.join(tmpd, f'{i:05d}.png'), fr)
    r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '20',
                        '-i', os.path.join(tmpd, '%05d.png'),
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_path],
                       capture_output=True, text=True)
    shutil.rmtree(tmpd)
    if r.returncode != 0:
        print("ffmpeg 실패:", r.stderr[:300])
        out_path = out_path.rsplit('.', 1)[0] + '.gif'
        imageio.mimsave(out_path, frames, fps=20)
    imageio.imwrite(out_path.rsplit('.', 1)[0] + '_frame0.png', frames[0])
    print(f"\n저장: {os.path.abspath(out_path)}  ({len(frames)} frames)")

def _save(frames, traj, out_path, n_steps):
    import subprocess, tempfile, shutil
    t = np.array(traj)
    print(f"\n프레임 {len(frames)}개, 해상도 {frames[0].shape}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmpd = tempfile.mkdtemp()
    for i, fr in enumerate(frames):
        imageio.imwrite(os.path.join(tmpd, f'{i:05d}.png'), fr)
    r = subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate','20',
                        '-i', os.path.join(tmpd,'%05d.png'),'-c:v','libx264',
                        '-pix_fmt','yuv420p', out_path], capture_output=True, text=True)
    shutil.rmtree(tmpd)
    print(("저장: " if r.returncode==0 else "ffmpeg 실패: ") + os.path.abspath(out_path))

if __name__ == '__main__':
    main()
