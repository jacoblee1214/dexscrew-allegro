"""두 학습 로그를 같은 스텝 지점에서 비교."""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob, numpy as np, sys
import sys, os
B='/mnt/sdb/jake/outputs'
RUNS = {}
for a in (sys.argv[1:] or ['XHandHoraScrewDriver_teacher/baseline',
                           'AllegroHoraScrewDriver_teacher/allegro_a0']):
    RUNS[os.path.basename(a)] = os.path.join(B, a)
KEYS = ['rotation_reward','screw/angular_velocity','screw/positive_vel_ratio',
        'episode_lengths/step','episode_rewards/step','step_all_reward','torques','rotate_penalty']
data = {}
for name, d in RUNS.items():
    fs = sorted(glob.glob(d+'/**/events.out.tfevents*', recursive=True))
    if not fs: print(f"{name}: 로그 없음"); continue
    ea = EventAccumulator(fs[-1], size_guidance={'scalars':0}); ea.Reload()
    data[name] = {k: [(s.step, s.value) for s in ea.Scalars(k)] for k in KEYS if k in ea.Tags()['scalars']}
n = min(len(v.get('rotation_reward',[])) for v in data.values()) if data else 0
print(f"공통 비교 구간: 각 런의 처음 {n}개 기록 지점\n")
print(f"{'지표':<28}" + "".join(f"{k:>16}" for k in data) + f"{'차이':>12}")
for k in KEYS:
    row = {}
    for name, d in data.items():
        v = [x[1] for x in d.get(k, [])][:n]
        if len(v) < 3: continue
        row[name] = np.mean(v[-max(3, n//5):])
    if len(row) == len(data) and row:
        vals = list(row.values())
        rel = (vals[1]-vals[0])/abs(vals[0])*100 if vals[0] else float('nan')
        print(f"{k:<28}" + "".join(f"{v:>16.4f}" for v in vals) + f"{rel:>11.1f}%")
