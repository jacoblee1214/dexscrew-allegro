"""학습 전 손 이식 검증. 이번 프로젝트에서 실제로 겪은 함정들을 전부 체크한다.
사용: python validate_port.py task=AllegroHoraScrewDriver headless=True sim_device=cuda:0 ...
"""
import isaacgym
import numpy as np, torch, hydra
from omegaconf import DictConfig, OmegaConf
from dexscrew.tasks import isaacgym_task_map
from dexscrew.utils.reformat import omegaconf_to_dict
from dexscrew.utils.misc import set_np_formatting, set_seed
for n,f in [('eq',lambda x,y:x.lower()==y.lower()),('contains',lambda x,y:x.lower() in y.lower()),
            ('if',lambda p,a,b:a if p else b),('resolve_default',lambda d,a:d if a=='' else a)]:
    OmegaConf.register_new_resolver(n,f)

R=[]
def chk(ok, name, msg):
    R.append((ok,name,msg)); print(f"  [{'PASS' if ok else 'FAIL'}] {name:<28} {msg}")

@hydra.main(config_name='config', config_path='configs')
def main(config: DictConfig):
    set_np_formatting(); config.seed = set_seed(config.seed)
    env = isaacgym_task_map[config.task_name](
        config=omegaconf_to_dict(config.task), sim_device=config.sim_device,
        graphics_device_id=config.graphics_device_id, headless=config.headless)
    nA = env.num_actions; thr = env.reset_dist_threshold*1000
    print(f"\n=== {config.task_name}  DOF={env.num_xhand_hand_dofs} ===\n")

    # 1) 차원 정합
    # obs_buf 는 최근 3스텝을 이어붙이므로 스텝당 슬라이스 폭 = num_obs//3.
    # 이게 2*numActions 보다 작으면 목표값이 조용히 잘린다(XHand 는 32>=24 라 무해).
    chk(env.num_obs//3 >= 2*nA, "관측 차원",
        f"스텝당 슬라이스 {env.num_obs//3} >= 2*numActions {2*nA} 이어야 함 (num_obs={env.num_obs})")
    pd = config.train.ppo.proprio_dim
    chk(pd == 2*nA, "proprio_dim", f"{pd}, 필요 2*numActions={2*nA}")

    env.reset()
    T=[]; C=[]
    for _ in range(200):
        env.step(torch.zeros((env.num_envs, nA), device=env.rl_device))
        T.append(env.torques[:,:,:env.num_xhand_hand_dofs].abs().mean(1).cpu().numpy())
        C.append(torch.norm(env.contact_forces,dim=-1).cpu().numpy())
    T=np.array(T); C=np.array(C)
    nb = env.num_xhand_hand_bodies
    hand_f = C[:,:,:nb]; obj_f = C[:,:,nb:]
    bn = env.gym.get_actor_rigid_body_names(env.envs[0],0)

    # 2) 자기충돌: 손 링크 힘이 물체 힘보다 훨씬 크면 손끼리 부딪히는 것
    hmax, omax = hand_f.max(), obj_f.max()
    worst = bn[int(hand_f.mean(axis=(0,1)).argmax())]
    chk(hmax <= 3*max(omax,1e-6), "자기충돌",
        f"손 링크 최대 {hmax:.0f}N vs 물체 최대 {omax:.0f}N (최다접촉 링크 {worst})")

    # 3) 지속 관통: 리셋 임펄스는 정상이나, 끝까지 큰 힘이 유지되면 기하 겹침이다.
    late = obj_f[len(obj_f)//2:].max()
    chk(late < 3000, "지속 관통",
        f"후반 100스텝 물체 최대 {late:.0f}N (리셋 임펄스 제외)")

    # 4) 종료 여유
    rb = env.rigid_body_states.cpu().numpy()
    nut = rb[:,env.screw_nut_rb_handle,:3]
    th = rb[:,env.fingertip_handles[-1],:3]; ix = rb[:,env.fingertip_handles[0],:3]
    dt_=np.linalg.norm(th-nut,axis=1).mean()*1000; di_=np.linalg.norm(ix-nut,axis=1).mean()*1000
    chk(max(dt_,di_) < thr*0.85, "종료 여유",
        f"엄지 {dt_:.0f}mm / 검지 {di_:.0f}mm, 임계 {thr:.0f}mm (85% 이내 권장)")
    prox = max(0.0, 1-((dt_+di_)/2)/thr)
    chk(prox > 0.20, "proximity 보상", f"{prox:.3f} (XHand 기준 0.275)")

    # 5) 토크: 리워드를 지배하면 에이전트가 빨리 죽으려 한다
    tq = (T.mean(axis=(0,1))**2).sum()
    ts = abs(float(config.task.env.reward.torque_penalty_scale))
    rot = float(config.task.env.reward.rotate_reward_scale)
    chk(tq*ts < rot*0.5, "토크 벌점 균형",
        f"토크^2합 {tq:.3f} x |scale| {ts} = {tq*ts:.2f}, rotate 보상 상한 {rot:.1f}")

    # 6) 손가락 참여도: 실제 접촉은 손끝이 아니라 중간 마디에서 난다.
    #    그래서 손끝만 보면 XHand 조차 0% 로 나온다. 체인 전체로 판정한다.
    import re
    def key(nm):
        m = re.match(r'^R(\d)\d', nm)
        if m: return 'R'+m.group(1)
        for k in ['thumb','index','mid','ring','pinky']:
            if k in nm: return k
        return None
    ftn = [bn[h] for h in env.fingertip_handles]
    lbl=['검지','중지','소지','약지','엄지']
    part=[]
    for tn in ftn:
        k = key(tn)
        idx = [i for i,nm in enumerate(bn) if key(nm)==k] if k else []
        part.append((C[:,:,idx]>0.01).any(axis=2).mean()*100 if idx else 0.0)
    part=np.array(part)
    chk((part>5).sum() >= 3, "손가락 참여",
        " ".join(f"{l}{p:.0f}%" for l,p in zip(lbl,part)))

    # 7) 물체 접촉 유지
    ratio = (obj_f.max(axis=2)>0.01).mean()*100
    chk(ratio > 80, "물체 접촉 유지", f"{ratio:.1f}% (XHand 기준 98%)")

    nf = sum(1 for ok,_,_ in R if not ok)
    print(f"\n=== {len(R)-nf}/{len(R)} 통과" + ("  ✅ 학습 진행 가능" if nf==0 else f"  ❌ {nf}개 실패") + " ===")
if __name__=='__main__': main()
