"""손 위치/방향/손가락자세를 검증기 점수로 동시 최적화.
env 를 한 번만 만들고 파라미터를 런타임에 갈아끼워 평가당 ~3초로 줄인다."""
import isaacgym
from isaacgym.torch_utils import quat_mul
import numpy as np, torch, hydra, json, xml.etree.ElementTree as ET
from scipy.optimize import minimize
from omegaconf import DictConfig, OmegaConf
from dexscrew.tasks import isaacgym_task_map
from dexscrew.utils.reformat import omegaconf_to_dict
from dexscrew.utils.misc import set_np_formatting, set_seed
for n,f in [('eq',lambda x,y:x.lower()==y.lower()),('contains',lambda x,y:x.lower() in y.lower()),
            ('if',lambda p,a,b:a if p else b),('resolve_default',lambda d,a:d if a=='' else a)]:
    OmegaConf.register_new_resolver(n,f)

URDF="/home/jake/allegro_assets/V6_Force_L/urdf/V6_Force_L.urdf"
def load_urdf():
    r=ET.parse(URDF).getroot(); J={}
    for j in r.findall('joint'):
        o=j.find('origin'); a=j.find('axis'); l=j.find('limit')
        J[j.get('name')]=dict(type=j.get('type'),parent=j.find('parent').get('link'),child=j.find('child').get('link'),
          xyz=np.array([float(v) for v in o.get('xyz','0 0 0').split()]) if o is not None else np.zeros(3),
          rpy=np.array([float(v) for v in o.get('rpy','0 0 0').split()]) if o is not None else np.zeros(3),
          axis=np.array([float(v) for v in a.get('xyz').split()]) if a is not None else np.array([0,0,1.]),
          lo=float(l.get('lower')) if l is not None else 0., hi=float(l.get('upper')) if l is not None else 0.)
    return J
J=load_urdf(); DOFS=[k for k,v in J.items() if v['type']=='revolute']
C2J={v['child']:k for k,v in J.items()}
def rpy2R(r,p,y):
    cr,sr,cp,sp,cy,sy=np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(y),np.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])
def axR(ax,q):
    ax=ax/np.linalg.norm(ax); K=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    return np.eye(3)+np.sin(q)*K+(1-np.cos(q))*K@K
def fk(link,qm):
    T=np.eye(4); ch=[]; cur=link
    while cur in C2J: ch.append(C2J[cur]); cur=J[C2J[cur]]['parent']
    for jn in reversed(ch):
        j=J[jn]; A=np.eye(4); A[:3,:3]=rpy2R(*j['rpy']); A[:3,3]=j['xyz']
        if j['type']=='revolute':
            B=np.eye(4); B[:3,:3]=axR(j['axis'],qm.get(jn,0.)); A=A@B
        T=T@A
    return T[:3,3]
CH={'th':['joint_00','joint_01','joint_02','joint_03'],'ix':['joint_10','joint_11','joint_12','joint_13'],
    'md':['joint_20','joint_21','joint_22','joint_23'],'rg':['joint_30','joint_31','joint_32','joint_33'],
    'pk':['joint_40','joint_41','joint_42','joint_43']}
TIP={'th':'thumb_3','ix':'index_3','md':'middle_3','rg':'ring_3','pk':'pinky_3'}
TH=CH['th']; TLO=np.array([J[j]['lo'] for j in TH]); THI=np.array([J[j]['hi'] for j in TH])
def make_pose(F, GAP):
    base={k:0.0 for k in DOFS}
    for fn in ['ix','md','rg']:
        for i,jn in enumerate(CH[fn]):
            base[jn]=0.0 if i==0 else J[jn]['lo']+F*(J[jn]['hi']-J[jn]['lo'])
    for i,jn in enumerate(CH['pk']):
        base[jn]=0.0 if i==0 else J[jn]['lo']+F*0.6*(J[jn]['hi']-J[jn]['lo'])
    cen=np.mean([fk(TIP[x],base) for x in ['ix','md']],axis=0)
    def cost(q):
        m=dict(base); m.update(dict(zip(TH,q)))
        gap=np.linalg.norm(fk(TIP['th'],m)-cen)
        clr=min(np.linalg.norm(fk(l,m)) for l in ['thumb_1','thumb_2','thumb_3'])
        return (gap-GAP)**2*1e4 + max(0.,0.065-clr)**2*1e5
    best=None
    for s_ in range(20):
        q0=TLO+np.random.RandomState(s_).rand(4)*(THI-TLO)
        r=minimize(cost,q0,bounds=list(zip(TLO,THI)),method='L-BFGS-B')
        if best is None or r.fun<best.fun: best=r
    out=dict(base); out.update(dict(zip(TH,best.x)))
    return np.array([out[j] for j in DOFS])

def qmul(a,b):
    x1,y1,z1,w1=a; x2,y2,z2,w2=b
    return np.array([w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2,
                     w1*z2+x1*y2-y1*x2+z1*w2, w1*w2-x1*x2-y1*y2-z1*z2])
def qaxis(ax,ang):
    ax=np.array(ax,dtype=float); ax/=np.linalg.norm(ax); s=np.sin(ang/2)
    return np.array([ax[0]*s,ax[1]*s,ax[2]*s,np.cos(ang/2)])

@hydra.main(config_name='config', config_path='configs')
def main(config: DictConfig):
    set_np_formatting(); config.seed = set_seed(config.seed)
    env = isaacgym_task_map[config.task_name](
        config=omegaconf_to_dict(config.task), sim_device=config.sim_device,
        graphics_device_id=config.graphics_device_id, headless=config.headless)
    dev=env.device; nd=env.num_xhand_hand_dofs
    Q0=np.array([0.890958,-0.349071,0.289916,-0.017075])   # 현재 handMountQuat
    P0=np.array([-0.0313,0.0265,0.2402])
    thr=env.reset_dist_threshold*1000

    def evaluate(x, steps=120):
        dx,dy,dz, a1,a2, F, GAP = x
        pose = make_pose(F, GAP)
        pose = np.clip(pose, env.xhand_dof_lower_limits, env.xhand_dof_upper_limits)
        for k in env.saved_grasping_states:
            env.saved_grasping_states[k][:, :nd] = torch.tensor(pose, device=dev, dtype=torch.float)
        env.joint_values_lst = list(pose)
        q = qmul(qmul(Q0, qaxis([1,0,0],a1)), qaxis([0,1,0],a2))
        env.hand_mount_quat = torch.tensor(q, device=dev, dtype=torch.float).unsqueeze(0)
        env.hand_base_pos  = torch.tensor(P0+np.array([dx,dy,dz]), device=dev, dtype=torch.float).unsqueeze(0)
        env.reset()
        A=[];T=[]
        for _ in range(steps):
            env.step(torch.zeros((env.num_envs, env.num_actions), device=env.rl_device))
            A.append(torch.norm(env.contact_forces,dim=-1).cpu().numpy())
            T.append(env.torques[:,:,:nd].abs().mean(1).cpu().numpy())
        A=np.array(A); T=np.array(T); nb=env.num_xhand_hand_bodies
        rb=env.rigid_body_states.cpu().numpy(); nut=rb[:,env.screw_nut_rb_handle,:3]
        dt_=np.linalg.norm(rb[:,env.fingertip_handles[-1],:3]-nut,axis=1).mean()*1000
        di_=np.linalg.norm(rb[:,env.fingertip_handles[0],:3]-nut,axis=1).mean()*1000
        palm=A[len(A)//2:,:,0].max()
        objf=A[len(A)//2:,:,nb:].max()
        tq=(T.mean(axis=(0,1))**2).sum()
        ratio=(A[:,:,nb:].max(axis=2)>0.01).mean()*100
        prox=max(0.,1-((dt_+di_)/2)/thr)
        m=dict(palm=float(palm),obj=float(objf),tq=float(tq),ratio=float(ratio),
               prox=float(prox),th=float(dt_),ix=float(di_))
        # 자기충돌 = 손 링크 접촉 중 물체 접촉으로 설명되지 않는 부분
        handmax = float(A[len(A)//2:,:,:nb].max())
        selfc = max(0.0, handmax - 1.2*objf)
        m['selfc']=selfc; m['handmax']=handmax
        # 점수(낮을수록 좋음)
        s  = 5.0*min(selfc,50000)/1000           # 자기충돌 (최우선)
        s += 3.0*min(objf,50000)/1000            # 물체 관통
        s += 3.0*tq                              # 토크 벌점
        s += 2.0*max(0., 0.30-prox)*10           # proximity
        s += 0.5*max(0., 95-ratio)               # 접촉 유지
        s += 5.0*max(0., max(dt_,di_)-0.85*thr)/10  # 종료 여유
        return s, m
    x0=np.array([0,0,0, 0,0, 0.70, 0.030])
    lo=np.array([-0.035,-0.035,-0.035, -0.7,-0.7, 0.50, 0.024])
    hi=np.array([ 0.035, 0.035, 0.035,  0.7, 0.7, 0.85, 0.036])
    rng=np.random.RandomState(0)
    mu=x0.copy(); sd=(hi-lo)/4
    best=(1e18,None,None)
    N_IT=int(config.get('cem_iters',8)); N_POP=int(config.get('cem_pop',14))
    print(f"CEM {N_IT} x {N_POP} = {N_IT*N_POP} 평가\n")
    for it in range(N_IT):
        pop=np.clip(rng.normal(mu,sd,size=(N_POP,len(x0))),lo,hi)
        if it==0: pop[0]=x0
        res=[]
        for x in pop:
            s,m=evaluate(x); res.append((s,x,m))
            if s<best[0]: best=(s,x.copy(),m)
        res.sort(key=lambda t:t[0])
        el=np.array([r[1] for r in res[:max(3,N_POP//4)]])
        mu=el.mean(0); sd=np.maximum(el.std(0), (hi-lo)*0.03)
        b=res[0]
        print(f"it{it}: best {b[0]:8.2f}  selfc {b[2]['selfc']:8.0f} obj {b[2]['obj']:8.0f} "
              f"tq {b[2]['tq']:6.2f} prox {b[2]['prox']:.3f} ratio {b[2]['ratio']:5.1f} "
              f"th/ix {b[2]['th']:.0f}/{b[2]['ix']:.0f}mm", flush=True)
    s,x,m=best
    print(f"\n=== 최적 (score {s:.2f}) ===")
    print(f"  metrics {json.dumps(m)}")
    print(f"  handStartPos=[{(P0+x[:3])[0]:.4f},{(P0+x[:3])[1]:.4f},{(P0+x[:3])[2]:.4f}]")
    q=qmul(qmul(Q0,qaxis([1,0,0],x[3])),qaxis([0,1,0],x[4]))
    print(f"  handMountQuat=[{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{q[3]:.6f}]")
    print(f"  flexion={x[5]:.3f} gap={x[6]*1000:.1f}mm")
    print("  initPoseValues=[" + ",".join(f"{v:.4f}" for v in make_pose(x[5],x[6])) + "]")
if __name__=='__main__': main()
