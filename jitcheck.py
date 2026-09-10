"""xhand_deploy.py 의 추론 경로를 하드웨어 없이 그대로 재현해 JIT 정책을 검증한다."""
import numpy as np, torch, torch.nn.functional as F
JIT="outputs/allegro_screwdriver.pt"
m=torch.jit.load(JIT, map_location="cpu"); m.eval()
print("JIT 로드 OK:", JIT)
print("\n-- 모델이 들고 있는 정규화 통계 --")
for k in ["running_mean","running_var","running_count","sa_mean","sa_var","sa_count"]:
    if hasattr(m,k):
        v=getattr(m,k)
        print(f"  {k:<14}{tuple(v.shape) if hasattr(v,'shape') else v}")
OBS_DIM = m.running_mean.shape[0]
PH, PD  = m.sa_mean.shape
print(f"\n-> 실기 코드에 넣을 값: OBS_DIM={OBS_DIM}  PROPRIO_HIST_LEN={PH}  PROPRIO_DIM={PD}")
print(f"   (XHand 원본은 96 / 30 / 24)")
# xhand_deploy.py 와 동일한 절차
obs_mean=m.running_mean.unsqueeze(0); obs_var=m.running_var.unsqueeze(0)
sa_mean=m.sa_mean.unsqueeze(0);       sa_var=m.sa_var.unsqueeze(0)
# 초기 자세(config) 로 히스토리 채우기
import yaml, re
txt=open('configs/task/AllegroThickScrewDriver.yaml').read()
q=[float(x) for x in re.search(r'initPoseValues: \[([^\]]+)\]',txt).group(1).replace('\n',' ').split(',')]
print(f"\n초기 자세 {len(q)}개 로드")
base=torch.tensor(q+q, dtype=torch.float32)                 # [관절위치, 목표] = 40
assert base.shape[0]==PD, f"proprio 차원 불일치 {base.shape[0]} vs {PD}"
hist=base.view(1,1,PD).repeat(1,PH,1)
obs=torch.zeros(1,OBS_DIM)
prev=np.array(q,dtype=np.double)
ACTION_SCALE=0.04167
lo=np.array(q)-1.0; hi=np.array(q)+1.0                       # 검증용 임시 한계
print("\n-- 20스텝 추론 --")
for t in range(20):
    with torch.no_grad():
        o=torch.clamp(obs,-5.,5.); o=F.pad(o,(0,OBS_DIM-o.shape[1]))
        no=torch.clamp((o-obs_mean)/torch.sqrt(obs_var+1e-5),-5.,5.)
        nh=torch.clamp((hist-sa_mean)/torch.sqrt(sa_var+1e-5),-5.,5.)
        mu,extrin,extrin_gt=m({"obs":no,"proprio_hist":nh,"point_cloud_info":torch.zeros(1,100,3)})
        mu=torch.clamp(mu,-1.,1.)
    a=mu.numpy().flatten()
    if t==0:
        print(f"  액션 차원 {a.shape[0]} (필요 20)  범위 [{a.min():+.3f},{a.max():+.3f}]")
        print(f"  extrin {tuple(extrin.shape)}  extrin_gt {tuple(extrin_gt.shape)}")
    tgt=np.clip(a*ACTION_SCALE+prev, lo, hi); prev=tgt.copy()
    obs=hist[:,-3:,:].reshape(1,-1).clone()
    cur=torch.cat([torch.tensor(tgt,dtype=torch.float32), torch.tensor(tgt,dtype=torch.float32)]).view(1,1,PD)
    hist=torch.cat([hist[:,1:],cur],dim=1)
print(f"  20스텝 후 관절 변화량 (rad): {np.round(prev-np.array(q),4)}")
print(f"  최대 변화 {np.abs(prev-np.array(q)).max():.4f} rad")
print("\n✅ 실기 추론 경로 통과")
