"""학습된 정책으로 영상 + 회전량 측정. 학습 아님(추론만)."""
import isaacgym
import os, numpy as np, torch, hydra, imageio, subprocess, tempfile, shutil
from omegaconf import DictConfig, OmegaConf
from dexscrew.algo.ppo.ppo import PPO
from dexscrew.tasks import isaacgym_task_map
from dexscrew.utils.reformat import omegaconf_to_dict
from dexscrew.utils.misc import set_np_formatting, set_seed
for n,f in [('eq',lambda x,y:x.lower()==y.lower()),('contains',lambda x,y:x.lower() in y.lower()),
            ('if',lambda p,a,b:a if p else b),('resolve_default',lambda d,a:d if a=='' else a)]:
    OmegaConf.register_new_resolver(n,f)

@hydra.main(config_name='config', config_path='configs')
def main(config: DictConfig):
    set_np_formatting(); config.seed = set_seed(config.seed)
    from isaacgym import gymapi
    env = isaacgym_task_map[config.task_name](
        config=omegaconf_to_dict(config.task), sim_device=config.sim_device,
        graphics_device_id=config.graphics_device_id, headless=config.headless)
    agent = PPO(env, 'outputs/_eval', full_config=config)
    agent.restore_test(config.checkpoint)
    agent.set_eval()

    obs = env.reset()
    W,H = int(config.get('rec_w',480)), int(config.get('rec_h',480))
    d = float(config.get('rec_dist',0.13)); ncol = int(config.get('rec_cols',3))
    objs = env.object_pos.cpu().numpy(); cams=[]
    for i in range(env.num_envs):
        cp=gymapi.CameraProperties(); cp.width,cp.height=W,H; cp.enable_tensors=False
        c=env.gym.create_camera_sensor(env.envs[i],cp); o=objs[i]
        env.gym.set_camera_location(c,env.envs[i],
            gymapi.Vec3(float(o[0]+d),float(o[1]+d),float(o[2]+d*0.7)),
            gymapi.Vec3(float(o[0]),float(o[1]),float(o[2]))); cams.append(c)
    nrow=int(np.ceil(env.num_envs/ncol))
    def grab():
        env.gym.render_all_camera_sensors(env.sim); ims=[]
        for i,c in enumerate(cams):
            im=env.gym.get_camera_image(env.sim,env.envs[i],c,gymapi.IMAGE_COLOR)
            ims.append(im.reshape(H,W,4)[:,:,:3].copy())
        while len(ims)<ncol*nrow: ims.append(np.zeros_like(ims[0]))
        return np.concatenate([np.concatenate(ims[r*ncol:(r+1)*ncol],axis=1) for r in range(nrow)],axis=0)

    steps=int(config.get('rec_steps',400)); frames=[]; ang0=env.nut_dof_pos.clone(); dones=0
    _fs=[]; _all=[]; _ang=[]; _done=[]
    for t in range(steps):
        pc = agent.point_cloud_mean_std(obs['point_cloud_info'].reshape(-1,3)).reshape(
             (obs['obs'].shape[0],-1,3)) if agent.normalize_point_cloud else obs['point_cloud_info']
        inp = {'obs':agent.running_mean_std(obs['obs']),
               'priv_info':agent.priv_mean_std(obs['priv_info']) if agent.normalize_priv else obs['priv_info'],
               'proprio_hist':obs['proprio_hist'],'point_cloud_info':pc}
        if bool(config.get('zero_action', False)):
            mu = torch.zeros((env.num_envs, env.num_actions), device=env.rl_device)
        else:
            with torch.no_grad(): mu,_,_ = agent.model.act_inference(inp)
            mu = torch.clamp(mu, -1.0, 1.0)
        obs, r, done, info = env.step(mu)
        dones += int(done.sum().item()); frames.append(grab())
        _ang.append(env.nut_dof_pos.clone().cpu().numpy().ravel())   # LONGHORIZON
        _done.append(done.clone().cpu().numpy())
        _cf = torch.norm(env.contact_forces, dim=-1)
        _fs.append(_cf[:, env.fingertip_handles].cpu().numpy())
        _all.append(_cf[:, :env.num_xhand_hand_bodies].cpu().numpy())
        if t%100==0: print(f"  {t}/{steps}  나사각 평균 {env.nut_dof_pos.mean().item():.3f} rad")
    # FINGERSTAT: 손가락별 기여도
    import collections
    FS = collections.defaultdict(list)
    ang = (env.nut_dof_pos - ang0).cpu().numpy().ravel()
    print(f"\n=== {steps} 스텝 결과 ===")
    print(f"  나사 회전량(rad): 평균 {ang.mean():+.2f}  최대 {ang.max():+.2f}  최소 {ang.min():+.2f}")
    print(f"  = 바퀴수: 평균 {ang.mean()/(2*np.pi):+.2f}  최대 {ang.max()/(2*np.pi):+.2f}")
    print(f"  중도 종료(파지 놓침) 횟수: {dones} / {env.num_envs}환경")
    A=np.array(_fs); B=np.array(_all)
    fn=list(config.task.env.asset.get('fingertipLinks',['index','mid','pinky','ring','thumb']))
    lbl=['검지','중지','소지','약지','엄지']
    # 리셋 사이 구간별 회전량 = 실제 연속 회전 능력
    A_=np.array(_ang); D_=np.array(_done)
    segs=[]
    for e in range(A_.shape[1]):
        st=0
        for t in range(len(A_)):
            if D_[t,e]:
                if t-st>20: segs.append((A_[t-1,e]-A_[st,e], t-st))
                st=t+1
        if len(A_)-st>20: segs.append((A_[-1,e]-A_[st,e], len(A_)-st))
    if segs:
        r=np.array([s[0] for s in segs]); L=np.array([s[1] for s in segs])
        print(f"\n  --- 리셋 사이 연속 회전 (구간 {len(segs)}개) ---")
        print(f"    구간 길이   평균 {L.mean():.0f} 스텝  최대 {L.max()} 스텝")
        print(f"    구간 회전   평균 {r.mean():+.2f} rad ({r.mean()/(2*np.pi):+.2f}바퀴)  "
              f"최대 {r.max():+.2f} rad ({r.max()/(2*np.pi):+.2f}바퀴)")
        print(f"    회전 속도   평균 {(r/L*100).mean():+.2f} rad/100스텝")
    print(f"\n  {'손가락':<8}{'손끝접촉%':>10}{'손끝평균N':>11}")
    for i,(n,l) in enumerate(zip(fn,lbl)):
        print(f"  {l:<8}{(A[:,:,i]>0.01).mean()*100:>10.1f}{A[:,:,i].mean():>11.2f}")
    bn=env.gym.get_actor_rigid_body_names(env.envs[0],0)
    peak=B.mean(axis=(0,1))
    print(f"\n  손 전체 링크 중 접촉 상위 8개:")
    for i in np.argsort(-peak)[:8]:
        print(f"    {bn[i]:<14}{peak[i]:>9.2f} N   접촉 시간비율 {(B[:,:,i]>0.01).mean()*100:5.1f}%")
    out=str(config.get('rec_out','outputs/videos/eval.mp4'))
    os.makedirs(os.path.dirname(out),exist_ok=True)
    td=tempfile.mkdtemp()
    for i,f_ in enumerate(frames): imageio.imwrite(os.path.join(td,f'{i:05d}.png'),f_)
    rr=subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate','30','-i',
                       os.path.join(td,'%05d.png'),'-c:v','libx264','-pix_fmt','yuv420p',out],
                      capture_output=True,text=True)
    shutil.rmtree(td)
    print(("저장: " if rr.returncode==0 else "ffmpeg실패: ")+os.path.abspath(out))
if __name__=='__main__': main()
