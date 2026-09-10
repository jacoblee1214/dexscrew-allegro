# DexScrew → Allegro v6 손 이식 결과

**기간** 2026-08-27 ~ 09-09 · **장비** 4× RTX 2080 Ti · **시뮬** IsaacGym Preview 4 · **총 학습** 약 100억 스텝

XHand(12-DoF)용 in-hand 나사 회전 태스크(DexScrew, arXiv:2512.02011)를 Allegro v6 왼손(20-DoF)으로 이식했다.

---

## 1. 한 줄 요약

**성능을 가른 것은 손이 아니라 물체 크기였다.** 손잡이를 25 → 31mm(24%)로 바꾸자 성능 1.5배, 학습 속도 5배가 됐고 파지 형태가 pinch에서 tripod로 바뀌었다.

초기 결과만 보면 "Allegro는 XHand의 55% 성능"이라는 결론에 도달했을 것이다. 128mm짜리 손에 25mm 손잡이는 사람으로 치면 이쑤시개를 돌리는 격이었다.

---

## 2. 최종 성능

| 구성 | Best reward | 도달 스텝 | 파지 형태 |
|---|---:|---:|---|
| XHand 25mm (기준) | **1491** | 11억 (이후 열화) | — |
| Allegro 25mm | 705 | 21억 | pinch · 검지 61% / 중지 1% |
| **Allegro 31mm** 시드42 | **1060** | 3.9억 | **tripod** · 검지 28% / 중지 26% |
| **Allegro 31mm** 시드43 | **1239** | 5.3억 | tripod |
| Allegro 36mm | 학습 실패 (−537) | — | 관통 25만 N |
| Allegro 40mm | — | — | 유효 손 위치 없음 |
| Allegro 31mm · **축 정정 URDF** | 390 | 2.4억 (정점) | 회전 대등, 파지 유지 1/3 |

두 시드가 모두 얇은 버전을 크게 상회 → 시드 운이 아님. Allegro 31mm는 XHand 정점의 **83%** 도달.

### 증류 (stage 2, 특권정보 없이 고유수용감각만)

| teacher | student | 보존율 |
|---|---:|---:|
| 705 (25mm) | 489 | 69% |
| **1060 (31mm)** | **976** | **91%** |

물체 질량·마찰·위치를 모른 채 91% 보존. `numEnvs` 48 vs 256은 품질 동일, 속도만 4배.

---

## 3. 핵심 발견 — 물체 크기에는 "창"이 있다

```
얇으면(25mm)   두 손가락으로만 집힘 → pinch → 회전 비효율
굵으면(36mm+)  손바닥이 물체와 충돌 → 관통 → 학습 붕괴
28~36mm        두 문제 사이의 창
```

**31mm ≈ 손 크기(128mm)의 24%.** 다른 손으로 이식할 때 출발점으로 쓸 비율.

정적 스윕(손잡이 × 손 후퇴량, 각 점마다 자세 재안정화) 결과 — 셀 값은 물체가 받는 최대 접촉력(N):

| 손잡이 | 후퇴 0 | 15mm | 30mm | 45mm | 관통 없는 구간 |
|---|---:|---:|---:|---:|---|
| 31.0mm | 10,388 | **787** | **1,258** | 9,684 | 15–30mm · 넓음 |
| 32.5mm | 64,490 | **938** | **1,081** | 26,372 | 15–30mm · 넓음 |
| 34.0mm | 188,645 | 43,889 | **1,191** | 41,842 | 30mm 한 점 |
| 35.5mm | 298,677 | 68,762 | **1,058** | 55,569 | 30mm 한 점 |
| 40.0mm | 관통을 잡으면 손끝이 종료 임계(120mm)를 넘음 | | | | **없음** |

손잡이가 굵어질수록 관통 없는 손 위치 구간이 좁아진다. 창의 상한은 손바닥 관통 회피와 손끝–너트 거리 제약이 만나는 지점이다.

---

## 4. 실패는 늘 같은 사슬이었다

25mm를 제외한 모든 실패가 이 경로를 밟았다.

1. 초기 자세가 물체를 관통한다
2. PD 제어기가 위치오차만큼 토크를 계속 출력한다
3. `torque_penalty = (토크²).sum() × (−3.0)` 이 리워드를 지배한다
4. 스텝당 리워드가 **음수**가 된다
5. 에이전트에게 "빨리 끝내는 것"이 최적이 된다
6. 에피소드 길이가 붕괴한다 — **433 → 16 스텝**

**스텝당 리워드의 부호가 에피소드 길이를 결정한다.** 이 인과를 파악한 뒤 `torque_penalty_scale`을 −3.0 → −0.5로 손 크기에 맞춰 조정해 해결했다.

---

## 5. 손을 바꿀 때 조정해야 하는 파라미터

| 파라미터 | XHand → Allegro | 이유 |
|---|---|---|
| `numActions` | 12 → 20 | DOF |
| `numObservations` | 96 → 120 | 3 × (관절 + 목표) · **조용히 잘림** |
| `proprio_dim` | 24 → 40 | 상동 · **조용히 잘림** |
| `temporal_fusing_input_dim` | 24 → 40 | **stage 2에서만 발현** |
| `fingertipLinks` | 이름 교체 | URDF마다 다름 |
| `handStartPos` / `handMountQuat` | Kabsch 정렬 + 30mm 후퇴 | 베이스 프레임 축 규약이 다름 |
| `initPoseValues` | IK + **시뮬 안정화** | 토크로 유지 가능한 자세여야 함 |
| `reset_dist_threshold` | 0.10 → 0.12 | 손가락이 길어서 |
| `torque_penalty_scale` | −3.0 → −0.5 | 관절 20개 + 제곱합 |
| `randomizeScaleList` | 0.85–1.25 → 1.00–1.25 | 손 크기에 맞는 물체 |
| `begin_aggregate` 크기 | 560 → 33 | PhysX 한계 128 초과 |

**차원 3종은 에러 없이 조용히 잘린다.** 안 고치면 학습이 도는 것처럼 보이면서 목표값 절반이 버려진다. `temporal_fusing_input_dim`은 stage 2에서만 쓰여 teacher 학습 중에는 발현되지 않는다.

---

## 6. 버린 지표들

측정을 네 번 갈아엎었다.

| 지표 | 왜 무효였나 |
|---|---|
| 물체 이동량 | object가 `fix_base_link=True` — 손이 있든 없든 동일 |
| 손끝 접촉력 | 실제 접촉은 중간 마디에서 남 (XHand도 tip 접촉 0.0 N) |
| 손끝–회전축 거리 | 종료 조건은 **너트 바디 원점**까지의 거리를 씀 |
| 손바닥 접촉력 | 자기충돌과 물체접촉이 섞임 — 반작용 쌍으로 분리 필요 |

**교훈:** 최적화하기 전에 `check_termination`과 `compute_reward`가 *실제로 읽는* 변수를 전부 나열할 것. 그 목록에 없는 양은 최적화 목적함수가 될 수 없다.

사전검증 도구 `validate_port.py`도 처음 만들었을 때 XHand(정상 기준)에서 3개 FAIL을 냈다. **검증기 자체를 기준 대상으로 먼저 검증해야 한다.**

---

## 7. 겪은 함정

**PhysX aggregate 한계** — 원본이 `(bodies+2)×20 = 560`을 요청. PxAggregate 최대는 128. Allegro에서 `free(): invalid pointer`로 heap 손상. XHand는 640인데 우연히 안 터졌을 뿐이다.

**손 포즈는 `create_actor`가 아니라 `reset_idx`가 정한다** — 매 리셋마다 `root_state_tensor`로 덮어쓴다. `_init_object_pose`를 고치고 값이 들어가는 것을 print로 확인해도 손은 움직이지 않는다.

**충돌체가 바디당 볼록껍질 1개** — Allegro 손바닥은 오목한데 그 공간이 메워져, 엄지가 실제로는 빈 공간에서 "유령 손바닥"과 자기충돌한다. VHACD로 386조각 분해하면 해소되나 접촉력이 폭주해 채택하지 않았다.

**후반 드리프트** — XHand는 에피소드 길이가 439 → 117로 붕괴했다(회전 능력은 유지). **`last.pth`는 신뢰할 수 없다 — 평가·영상은 반드시 `best_reward_*.pth`로.**

**임계값이 스크립트마다 다르다** — `reset_dist_threshold`가 config 0.05, teacher 0.1, student 0.15, nutbolt 0.07. config 값으로 재면 XHand조차 시작부터 종료 조건을 위반한다.

**URDF 관절 부호는 추측하지 말 것** — 두 URDF는 같은 물리 손인데 링크 원점 배치가 달라(강체변환 잔차 11mm) "관절 몇 rad"가 같은 자세를 뜻하지 않는다. 부호 8개의 256조합을 전수탐색해 링크 간 거리행렬이 일치하는 조합을 골랐다.

---

## 8. 미해결 — 축 정정 URDF의 성능 격차

실기에 반드시 써야 하는 `V6_Force_L`(손가락 밑동 5관절 축 부호 정정, effort 15 → 0.38)로 재학습한 결과가 390에 그쳤다. 뜯어보면 **회전 능력이 아니라 파지 유지가 문제다** (2.45억 스텝 동일 시점):

| 지표 | 옛 URDF s42 | 옛 URDF s43 | 새 URDF |
|---|---:|---:|---:|
| rotation_reward | 1.076 | 1.089 | 1.053 |
| screw/angular_velocity | 0.459 | 0.605 | **0.723** |
| positive_vel_ratio | 0.681 | 0.681 | 0.697 |
| torques | 0.967 | 1.156 | **0.472** |
| **episode_lengths** | **451.0** | **439.7** | **166.2** |
| **pose_diff_penalty** | 2.49 | 2.87 | **5.33** |
| episode_rewards | 1030.8 | 953.6 | 311.2 |

나사는 오히려 더 빨리 돌리고 토크는 절반만 쓴다. **에피소드가 3배 짧아 누적이 적을 뿐이다.**

6구간 추이에서 `pose_diff_penalty`만 끝까지 단조 증가(2.7 → 7.6), 토크는 후반 2배(0.53 → 1.14). 정책이 초기 자세에서 점점 멀어지다 손끝이 종료 임계를 넘어 끊긴다.

**두 가설, 아직 못 가림**

- ① 초기 자세가 이 손에 최적이 아님 → 정책이 매번 벗어나야 회전이 됨
- ② 새 URDF 굴곡 관절 한계가 14% 좁아진 대가 (예: `joint11 [-0.110, 1.800]` → `joint_11 [-0.070, 1.571]`). ②라면 옛 URDF 성적이 과대평가였던 셈

**가리는 법:** 학습된 정책이 도달한 에피소드 중반 관절각을 새 `initPoseValues`로 삼아 짧게 재학습. ①이면 크게 좋아지고 ②면 별 차이 없음. (30분 규모)

---

## 9. 실기 배포

`xhand-deploy/xhand_deploy.py`와 `skill-teleop` 저장소를 읽어 확인한 저자들의 실기 루프:

1. **사람이 VR(Quest) 텔레옵으로 드라이버를 집어 손에 놓는다** — 집기 정책은 없다
2. 손을 하드코딩된 초기 자세로 이동 (sim의 `initPoseValues`와 같은 값)
3. 버튼 X로 정책 on, 20Hz로 delta 액션 (`target = μ × 0.04167 + prev`)
4. 정책이 돌릴 수 있는 만큼 돌린다 — **자율 재파지(gaiting)는 없다**
5. 버튼 Y로 off, 사람이 다시 잡아준다 (`grasp_position`/`release_position` 하드코딩)
6. 3–5 반복

sim 측정으로도 확인됐다: 1600스텝 장기 평가에서 리셋 사이 구간 평균 581스텝을 버티며 **0.09바퀴**(최대 0.95)만 돌린다. `reset_dist_threshold` 종료 조건이 손가락 이탈을 금지해 gaiting이 원천 차단돼 있다.

실기에 올라가는 것은 **student(stage 2) JIT**이며, 점군 입력은 `torch.zeros((1,100,3))`으로 0을 넣는다.

### Allegro 판 준비 상태

| 항목 | 상태 | 비고 |
|---|---|---|
| student → JIT 변환 | 완료 | 정규화 통계 내장 |
| 추론 경로 검증 | 완료 | 하드웨어 없이 20스텝 재현 |
| 배포 스크립트 `allegro_deploy.py` | 완료 | 차원·자세·한계·정규화·20Hz |
| 관절 순서 매핑 | **가정** | 실기에서 `--verify-mapping` 확인 필요 |
| `AllegroInterface` | 미구현 | v6 ROS2 인터페이스 연결 필요 |

**관절 순서 매핑이 유일한 위험 지점이다.** v4 SDK 순서가 [검지, 중지, 약지, 엄지]였고 v6는 소지가 끝에 붙는다는 안내를 받아 `[index, middle, ring, thumb, pinky]`로 가정했다. 틀리면 엄지 명령이 검지 모터로 가므로, 실기 첫 실행은 반드시 손가락을 하나씩 움직여 확인하는 `--verify-mapping`부터 해야 한다.

### HW 도착 후 순서

1. force teacher → student 증류 *(진행 중)*
2. student → JIT 재변환 (몇 분)
3. `--verify-mapping`으로 관절 순서 확인 ← 실물 필요
4. `AllegroInterface`에 v6 ROS2 연결 ← 실물/소스 필요

---

## 10. 재사용 가능한 도구

| 파일 | 용도 |
|---|---|
| `validate_port.py` | 학습 전 9항목 검증 (XHand 기준 9/9 통과 확인 후 사용) |
| `relax_pose.py` | IK 자세를 시뮬에서 안정화 → 토크로 유지 가능한 자세로 |
| `whichbody.py` | 링크별 접촉력 정렬 → 관통 vs 자기충돌 판별 |
| `timeline.py` | 접촉력 시간축 — 리셋 임펄스와 지속 관통 구분 |
| `eval_video.py` | 정책 영상 + 회전량 + 손가락별 접촉률 |
| `diag_pose.py` | 손끝 좌표·너트거리·접촉·나사각 |
| `compare_runs.py` | 여러 런을 같은 스텝에서 대조 |
| `optimize_grasp.py` | 손 위치·방향·자세 CEM 동시 최적화 |

---

## 11. 체크포인트

`/mnt/sdb/jake/outputs/` 아래:

```
XHandHoraScrewDriver_teacher/baseline/stage1_nn/best_reward_1491.87.pth
AllegroHoraScrewDriver_teacher/allegro_tq05/stage1_nn/best_reward_705.62.pth
AllegroThickScrewDriver_teacher/thick_v2/stage1_nn/best_reward_1060.07.pth
AllegroThickScrewDriver_teacher/thick_v2_s43/stage1_nn/best_reward_1239.17.pth
AllegroForceScrewDriver_teacher/force_v1/stage1_nn/best_reward_389.98.pth
AllegroHoraScrewDriver_student_padapt/student_thick/stage2_nn/model_best.ckpt
```

배포 자산: `~/dexscrew/xhand-deploy/allegro_deploy.py`, `allegro_screwdriver.pt`

> 현재 JIT는 **옛 URDF 기반**이라 실기에 그대로 쓰면 다섯 손가락 밑동이 반대로 움직인다. 축 정정 URDF의 student 증류 완료 후 재변환 필요.
