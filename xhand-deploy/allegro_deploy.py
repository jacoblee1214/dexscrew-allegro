"""
Allegro v6 (20-DoF) 실기 배포 — dexscrew/xhand-deploy/xhand_deploy.py 의 Allegro 판.

원본과 다른 점:
  OBS_DIM      96 -> 120      (3 스텝 x (20 관절위치 + 20 목표))
  PROPRIO_DIM  24 -> 40
  액션 차원     12 -> 20
  제어기        XHandControl(EtherCAT) -> Allegro ROS2/CAN  (아래 AllegroInterface 를 채울 것)

사용:
  python allegro_deploy.py --verify-mapping   # ★ 실기 첫 실행은 반드시 이것부터
  python allegro_deploy.py --hold             # 초기 자세로만 이동 후 정지
  python allegro_deploy.py                    # 정책 실행
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

# =========================================================================
# 정책
# =========================================================================
JIT_POLICY_PATH = "./allegro_force.pt"   # V6_Force_L(축 정정) student. 옛 URDF 판은 allegro_screwdriver.pt
DEVICE = torch.device("cpu")

OBS_DIM = 120
PROPRIO_HIST_LEN = 30
PROPRIO_DIM = 40
N_DOF = 20
ACTION_SCALE = 0.04167          # xhand_deploy 와 동일 (sim 0.05 의 5/6)
CONTROL_DECIMATION = 10         # 200Hz 루프에서 10 스텝마다 추론 = 20Hz
LOOP_DT = 0.005

# =========================================================================
# 관절 순서
# =========================================================================
# 정책(=sim) 순서: joint_00..03(엄지), 10..13(검지), 20..23(중지), 30..33(약지), 40..43(소지)
POLICY_JOINT_NAMES = [f"joint_{f}{k}" for f in range(5) for k in range(4)]
POLICY_FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

# ★★★ 검증 필요 ★★★
# v4 SDK/CAN 순서는 [Index, Middle, Ring, Thumb] 였고, v6 는 여기에 소지가 끝에 붙는 것으로 안내받음
# → [Index, Middle, Ring, Thumb, Pinky] 로 가정.
# 이 가정이 틀리면 엄지/검지 명령이 교차되어 위험하다. 반드시 --verify-mapping 으로 확인할 것.
HW_FINGER_ORDER = ["index", "middle", "ring", "thumb", "pinky"]

_pol = {f: i for i, f in enumerate(POLICY_FINGER_ORDER)}
# HW 인덱스 h 에 들어갈 정책 인덱스
HW_TO_POLICY_IDX = [
    _pol[f] * 4 + k for f in HW_FINGER_ORDER for k in range(4)
]
POLICY_TO_HW_IDX = [0] * N_DOF
for hw_i, pol_i in enumerate(HW_TO_POLICY_IDX):
    POLICY_TO_HW_IDX[pol_i] = hw_i


def policy_to_hw(vec):
    """정책 순서 20-vector -> 하드웨어 순서."""
    return np.asarray(vec)[POLICY_TO_HW_IDX]


def hw_to_policy(vec):
    """하드웨어 순서 20-vector -> 정책 순서."""
    return np.asarray(vec)[HW_TO_POLICY_IDX]


# =========================================================================
# 자세 / 한계  (configs/task/AllegroForceScrewDriver.yaml + V6_Force_L.urdf 에서 추출)
# =========================================================================
POLICY_INIT_POSE = np.array([
      0.5148,  -0.5633,   0.7519,  -0.0049,   # thumb
      0.2807,   0.7087,   0.0054,  -0.0281,   # index
      0.1312,   1.3075,  -0.0210,  -0.0638,   # middle
     -0.2177,   1.1679,   0.0111,   0.0966,   # ring
     -0.3136,   0.9443,  -0.1610,  -0.1548,   # pinky
], dtype=np.double)

DOF_LOWER = np.array([
     -0.035,  -1.658,  -0.175,  -0.175,
     -0.384,  -0.070,  -0.175,  -0.175,
     -1.134,  -0.070,  -0.175,  -0.175,
     -1.309,  -0.070,  -0.175,  -0.175,
     -1.309,  -0.070,  -0.175,  -0.175,
], dtype=np.double)

DOF_UPPER = np.array([
      1.658,   1.658,   1.309,   1.396,
      1.309,   1.571,   1.396,   1.396,
      1.134,   1.571,   1.396,   1.396,
      0.384,   1.571,   1.396,   1.396,
      0.436,   1.571,   1.396,   1.396,
], dtype=np.double)


# =========================================================================
# 하드웨어 인터페이스  — ★ 여기를 실기 ROS2/CAN 코드로 채울 것
# =========================================================================
class AllegroInterface:
    """Allegro v6 실기 인터페이스.

    v4 때 쓰던 ROS2 ForwardCommandController / safety_utils.py 에 대응하는 v6 버전을
    여기에 연결한다. read()/write() 는 모두 **하드웨어 순서** 20-vector 를 주고받는다.
    """

    def __init__(self):
        raise NotImplementedError(
            "실기 ROS2/CAN 인터페이스를 연결하세요. "
            "필요한 것: connect(), read_joint_positions()->20, "
            "send_joint_targets(20), disconnect()"
        )

    def read_joint_positions(self):
        """현재 관절 위치 (하드웨어 순서, rad) 20개."""
        raise NotImplementedError

    def send_joint_targets(self, targets_hw):
        """관절 목표 위치 (하드웨어 순서, rad) 20개 전송."""
        raise NotImplementedError

    def disconnect(self):
        pass


# =========================================================================
# 정책 실행
# =========================================================================
class PolicyRunner:
    def __init__(self, jit_path=JIT_POLICY_PATH):
        self.model = torch.jit.load(jit_path, map_location=DEVICE)
        self.model.eval()

        # 정규화 통계는 JIT 안에 들어 있다
        self.obs_mean = self.model.running_mean.unsqueeze(0)
        self.obs_var = self.model.running_var.unsqueeze(0)
        self.sa_mean = self.model.sa_mean.unsqueeze(0)
        self.sa_var = self.model.sa_var.unsqueeze(0)

        assert self.obs_mean.shape[1] == OBS_DIM, \
            f"JIT 의 obs 차원 {self.obs_mean.shape[1]} != OBS_DIM {OBS_DIM}"
        assert tuple(self.sa_mean.shape[1:]) == (PROPRIO_HIST_LEN, PROPRIO_DIM), \
            f"JIT 의 proprio {tuple(self.sa_mean.shape[1:])} != ({PROPRIO_HIST_LEN},{PROPRIO_DIM})"

        self.reset(POLICY_INIT_POSE)

    def reset(self, init_pose_policy):
        base = torch.tensor(
            np.concatenate([init_pose_policy, init_pose_policy]),
            dtype=torch.float32, device=DEVICE,
        )
        self.hist = base.view(1, 1, PROPRIO_DIM).repeat(1, PROPRIO_HIST_LEN, 1)
        self.obs = torch.zeros(1, OBS_DIM, device=DEVICE, dtype=torch.float32)
        self.prev_target = np.asarray(init_pose_policy, dtype=np.double).copy()

    def step(self, q_policy):
        """현재 관절(정책 순서) -> 다음 목표(정책 순서)."""
        with torch.no_grad():
            o = torch.clamp(self.obs, -5.0, 5.0)
            o = F.pad(o, (0, OBS_DIM - o.shape[1]))
            norm_obs = torch.clamp(
                (o - self.obs_mean) / torch.sqrt(self.obs_var + 1e-5), -5.0, 5.0)
            norm_hist = torch.clamp(
                (self.hist - self.sa_mean) / torch.sqrt(self.sa_var + 1e-5), -5.0, 5.0)

            mu, _, _ = self.model({
                "obs": norm_obs,
                "proprio_hist": norm_hist,
                # 실기에는 점군이 없다. sim 학습 시에도 이 경로는 student 가
                # 고유수용감각으로 대체하도록 증류되어 있으므로 0 을 넣는다.
                "point_cloud_info": torch.zeros(1, 100, 3, device=DEVICE),
            })
            action = torch.clamp(mu, -1.0, 1.0).cpu().numpy().flatten()

        target = np.clip(action * ACTION_SCALE + self.prev_target, DOF_LOWER, DOF_UPPER)
        self.prev_target = target.copy()

        # 관측 갱신 (xhand_deploy 와 동일한 순서)
        self.obs = self.hist[:, -3:, :].reshape(1, -1).clone()
        cur = torch.tensor(
            np.concatenate([q_policy, target]), dtype=torch.float32, device=DEVICE
        ).view(1, 1, PROPRIO_DIM)
        self.hist = torch.cat([self.hist[:, 1:], cur], dim=1)
        return target


# =========================================================================
# 매핑 검증  — 실기 첫 실행은 반드시 이것부터
# =========================================================================
def verify_mapping(hand, amplitude=0.25, dwell=1.5):
    """손가락을 하나씩 순서대로 움직여 매핑을 눈으로 확인한다.

    화면에 출력되는 손가락 이름과 실제로 움직이는 손가락이 일치해야 한다.
    다르면 HW_FINGER_ORDER 를 고칠 것.
    """
    print("\n=== 매핑 검증 ===")
    print("각 손가락의 두 번째 관절(굴곡)만 조금씩 움직입니다.")
    print("화면 이름과 실제 움직이는 손가락이 같은지 확인하세요.\n")

    home_hw = policy_to_hw(POLICY_INIT_POSE)
    hand.send_joint_targets(home_hw)
    time.sleep(2.0)

    for fi, fname in enumerate(POLICY_FINGER_ORDER):
        j = fi * 4 + 1                      # 각 손가락의 굴곡 관절
        for sign in (+1, -1, 0):
            target = POLICY_INIT_POSE.copy()
            target[j] = np.clip(
                target[j] + sign * amplitude, DOF_LOWER[j], DOF_UPPER[j])
            hand.send_joint_targets(policy_to_hw(target))
            if sign != 0:
                print(f"  >>> 지금 움직이는 손가락: {fname.upper()}  "
                      f"({POLICY_JOINT_NAMES[j]}, {sign:+d})")
            time.sleep(dwell)

    hand.send_joint_targets(home_hw)
    print("\n검증 끝. 순서가 달랐다면 HW_FINGER_ORDER 를 실제 순서로 고치세요.")
    print(f"  현재 가정: {HW_FINGER_ORDER}")


# =========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-mapping", action="store_true",
                    help="손가락을 하나씩 움직여 관절 매핑 확인 (첫 실행 필수)")
    ap.add_argument("--hold", action="store_true",
                    help="초기 자세로만 이동하고 정지")
    ap.add_argument("--policy", default=JIT_POLICY_PATH)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = 무한")
    args = ap.parse_args()

    hand = AllegroInterface()

    try:
        if args.verify_mapping:
            verify_mapping(hand)
            return

        # 초기 자세로 이동
        print("초기 자세로 이동...")
        hand.send_joint_targets(policy_to_hw(POLICY_INIT_POSE))
        time.sleep(2.0)
        if args.hold:
            print("정지 상태 유지. Ctrl-C 로 종료.")
            while True:
                time.sleep(0.5)

        runner = PolicyRunner(args.policy)
        print(f"정책 로드 완료: {args.policy}")
        print(f"제어 {1/(LOOP_DT*CONTROL_DECIMATION):.0f} Hz, "
              f"action_scale {ACTION_SCALE}")

        target = POLICY_INIT_POSE.copy()
        step = 0
        while args.max_steps == 0 or step < args.max_steps:
            q_hw = hand.read_joint_positions()
            q_policy = hw_to_policy(q_hw)

            if step % CONTROL_DECIMATION == 0:
                target = runner.step(q_policy)

            hand.send_joint_targets(policy_to_hw(target))
            time.sleep(LOOP_DT)
            step += 1

    except KeyboardInterrupt:
        print("\n중단됨 — 현재 자세 유지")
    finally:
        hand.disconnect()


if __name__ == "__main__":
    main()
