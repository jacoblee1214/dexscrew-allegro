# DexScrew → Allegro v6 (20-DoF) 이식

[DexScrew](https://github.com/x-robotics-lab/dexscrew) (arXiv:2512.02011) 를 XHand(12-DoF)
에서 **Allegro v6 왼손(20-DoF)** 으로 옮긴 private fork.

- 연구 결과 / 실패 분석 / 미해결 과제 → **[docs/allegro_port_report.md](docs/allegro_port_report.md)**
- 이 문서는 **환경 구축과 실행 방법**만 다룬다.

업스트림 대비 변경은 기존 파일 5개 15줄 + 신규 파일뿐이다.
`git remote add upstream https://github.com/x-robotics-lab/dexscrew.git` 후
`git diff upstream/main -- dexscrew/` 로 전체 차이를 볼 수 있다.

---

## 1. 환경

| 항목 | 값 |
|---|---|
| OS | Ubuntu 22.04 |
| 패키지 관리 | `uv` (conda 아님) |
| Python | 3.8.20 |
| torch | 2.4.1+cu118 |
| numpy | 1.23.0 (2.x 불가) |
| 시뮬레이터 | IsaacGym Preview 4 |

```bash
# IsaacGym 은 배포가 끊겨 별도 확보 필요 (~/isaacgym 에 배치)
uv venv --python 3.8
source .venv/bin/activate
uv pip install -e .
uv pip install -e ~/isaacgym/python
```

`LD_LIBRARY_PATH` 에 venv 의 `lib` 이 없으면 `libpython3.8.so.1.0` 을 못 찾는다.
`.venv/bin/activate` 끝에 아래를 넣어두면 매번 신경 안 써도 된다.

```bash
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH
```

### 손 URDF (저장소에 없음)

사내 자산이라 제외했다. 별도로 받아서 심볼릭 링크한다.

```bash
ln -s ~/allegro_assets/V6_Force_L    assets/allegro_v6_force_l   # 축 정정판 (권장)
ln -s ~/allegro_assets/V6_left_urdf  assets/allegro_v6_left      # 초기판
```

두 URDF 차이: 초기판은 엄지 축 부호가 뒤집혀 저장돼 있다. **sim 안에서는 기구학적으로 문제
없지만 실기에 그대로 쓰면 base 관절이 반대로 움직인다.** 실기용은 반드시 `V6_Force_L`.

### 학습 산출물

`outputs/`, `wandb/` 는 큰 디스크로 심볼릭 링크해서 쓴다 (`/mnt/sdb/<user>/`).

---

## 2. 실행

```bash
# stage 1 — teacher (특권정보 사용)
bash scripts/screwdriver_teacher_allegro_force.sh <GPU> <SEED> <RUN_NAME>

# stage 2 — student 증류 (고유수용감각만)
bash scripts/screwdriver_student_padapt_force.sh <GPU> <SEED> <RUN_NAME> \
  train.ppo.priv_info_ckpt_path=outputs/<teacher>/stage1_nn/best_reward_XXX.pth

# 배포용 TorchScript 변환
bash scripts/convert_student_jit_force.sh <GPU> <SEED> jitconv allegro_force.pt \
  task=AllegroForceScrewDriver train.load_path=outputs/<student>/stage2_nn/model_best.ckpt
```

> `num_envs` 를 바꾸면 `minibatch_size` 도 같이 바꿔야 한다.
> `batch = horizon_length(12) x num_envs` 이고 `batch % minibatch == 0` 을 assert 한다.
> 맞는 짝: 1024↔4096, 2048↔8192, 4096↔16384.

### 태스크 4종

| 태스크 | 드라이버 손잡이 | 비고 |
|---|---|---|
| `AllegroHoraScrewDriver` | 25mm (원본) | 핀치 자세로 수렴, 성능 낮음 |
| `AllegroThickScrewDriver` | 31mm | **최고 성능** (best 1239) |
| `AllegroXThickScrewDriver` | 36mm | 관통 발생, 실패 |
| `AllegroForceScrewDriver` | 31mm + 축 정정 URDF | **실기 배포용** |

---

## 3. 업스트림에서 고친 것

| 파일 | 내용 |
|---|---|
| `dexscrew/tasks/base/vec_task.py` | 관측 차원을 config 로. 원본은 `32*3` 하드코딩이라 20-DoF 에서 **조용히 잘렸다** |
| `dexscrew/algo/models/models.py` | student 인코더 입력 채널 `proprio_dim` 을 config 로 (24 → 40) |
| `dexscrew/algo/ppo/ppo.py`, `padapt.py` | `net_config` 에 `proprio_dim` 전달 |
| `dexscrew/tasks/__init__.py` | Allegro 태스크 4종 등록 |

### 신규

| 경로 | 내용 |
|---|---|
| `dexscrew/tasks/allegro_hora.py` | `xhand_hora.py` 의 Allegro 판 |
| `configs/task/Allegro*.yaml` | 태스크 4종 |
| `configs/train/Allegro*.yaml` | 학습 설정 (`proprio_dim: 40`) |
| `scripts/*_allegro*.sh`, `*_force.sh` | 실행 스크립트 8종 |
| `xhand-deploy/allegro_deploy.py` | 실기 배포 코드 |

`allegro_hora.py` 에서 손을 바꿀 때 반드시 건드려야 했던 곳은 리포트 §5 에 정리돼 있다.
특히 **PhysX aggregate 128 한계**(원본은 560 을 요청해 heap 손상으로 죽는다)와
**`reset_idx` 가 손 자세를 덮어쓴다**(`_init_object_pose` 수정은 효과 없음) 두 가지가
디버깅에 가장 오래 걸린 함정이었다.

---

## 4. 실기 배포

```bash
cd xhand-deploy
python allegro_deploy.py --verify-mapping   # ★ 첫 실행은 반드시 이것부터
python allegro_deploy.py --hold             # 초기 자세로만 이동
python allegro_deploy.py                    # 정책 실행
```

- `allegro_force.pt` — 축 정정 URDF student. **실기는 이것을 쓸 것**
- `allegro_screwdriver.pt` — 초기 URDF student. 실기 사용 금지

HW 도착 후 남은 일 두 가지:

1. `--verify-mapping` 으로 관절 순서 확인.
   현재 `HW_FINGER_ORDER = [index, middle, ring, thumb, pinky]` 는 **가정**이다.
   v4 SDK 가 `[Index, Middle, Ring, Thumb]` 였고 v6 는 소지가 끝에 붙는다는 안내에
   근거했다. 틀리면 엄지/검지 명령이 교차되어 위험하다.
2. `AllegroInterface.read_joint_positions()` / `send_joint_targets()` 를
   v6 ROS2 인터페이스에 연결. 지금은 `NotImplementedError` 스텁이다.

연속 회전은 정책 혼자 못 한다. 원저자 구현도 사람이 Quest VR 로 집어주고 → 버튼으로
20Hz 정책 시작 → 한 구간(약 0.09 회전) → 버튼으로 정지 → 사람이 다시 잡아주는 반복이다.

---

## 5. 진단 도구

이식 과정에서 만든 일회성 도구들. 루트에 둔 이유는 hydra 의 `config_path=configs` 가
스크립트 파일 기준 상대경로라 하위 디렉터리로 옮기면 전부 고쳐야 하기 때문이다
(업스트림의 `train.py`, `student_eval.py` 도 루트에 있다).

| 스크립트 | 용도 |
|---|---|
| `validate_port.py` | 이식 정합성 9개 항목 검사. **새 손을 꽂으면 제일 먼저 이것부터** |
| `relax_pose.py` | 초기 자세가 effort limit 안에서 유지 가능한지 확인 |
| `whichbody.py` | 접촉이 실제로 어느 링크에서 일어나는지 |
| `timeline.py` | 에피소드 내 보상/토크/종료 시점 추적 |
| `eval_video.py` | 체크포인트 평가 + 영상 녹화 |
| `record_video.py` | 자세 확인용 영상 |
| `diag_pose.py`, `view_pose.py` | 초기 자세 시각 확인 |
| `compare_runs.py` | 여러 run 의 지표 비교 |
| `optimize_grasp.py`, `optimize_grasp_force.py` | CEM 기반 파지 자세 탐색 |
| `check_pipeline.py` | 학습 전 관측/행동 차원 호환성 확인 |
| `torqdiag.py`, `perenv.py` | 토크 폭주 / env 별 이상치 추적 |
| `dump_frame.py`, `spin.py`, `fitobj.py`, `fixpoint.py` | 물체 배치·회전 확인 |
| `jitcheck.py`, `jitcheck_force.py` | TorchScript 변환 결과 차원 검증 |

> `validate_port.py` 는 **명백한 결함 선별용**이지 최적화 목표가 아니다.
> 이 점수를 올리려고 자세를 튜닝하면 실제로는 잘 되던 동작을 망가뜨린다.
> 판단은 시뮬레이션 영상을 직접 보고 한다.

---

## 6. 미해결

축 정정 URDF(`V6_Force_L`)가 초기 URDF보다 teacher 성능이 낮다 (390 vs 1060).
회전 능력 자체는 같고(1.053 vs 1.076) 각속도는 오히려 높은데 에피소드 길이가 짧다
(166 vs 451). 원인 후보 두 가지와 구분 방법은 리포트 §8 참조.
