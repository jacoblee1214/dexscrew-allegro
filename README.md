> **이 저장소는 Allegro v6(20-DoF) 이식 private fork 입니다.**
> 이식 관련 환경 구축·실행 방법은 [README_ALLEGRO.md](README_ALLEGRO.md),
> 결과와 분석은 [docs/allegro_port_report.md](docs/allegro_port_report.md) 를 보세요.
> 아래는 업스트림 원본 README 입니다.

---

# Learning Dexterous Manipulation Skills from Imperfect Simulations
<a href="https://dexscrew.github.io/"><strong>Project Page</strong></a>
|
<a href="https://arxiv.org/abs/2512.02011"><strong>arXiv</strong></a>

[Elvis Hsieh*](https://elvishh77.github.io/),
[Wen-Han Hsieh*](https://wen-hanhsieh.github.io/),
[Yen-Jen Wang*](https://wangyenjen.github.io/),
[Toru Lin](https://toruowo.github.io/),
[Jitendra Malik](https://people.eecs.berkeley.edu/~malik/),
[Koushil Sreenath†](https://hybrid-robotics.berkeley.edu/koushil/) 
[Haozhi Qi†](https://haozhi.io/),<br>
∗: Equal contribution (listed in alphabetical order). †: Equal advising. <br>

![Demo](./assets/DexScrew.gif)

## Installation

See [installation instructions](docs/install.md).

## Introduction

Our method contains the following four steps.
1. Learn a oracle policy with privileged information and point-clouds with RL in simulation.
2. Learn a padapt-based student policy using the oracle policy in simulation.
3. Using the trained rotation policy as motion prior, we leverage teleoperation to collect trajectories with downward motion and tactile sensing in real world.
4. Train a behavior cloning policy with expert trajectories to fuse extra observations.

The following session only provides example script of our method. For baselines, checkout [baselines](docs/baseline.md).

## Step 1: Oracle Policy Training

To train an oracle policy $f$ with RL, run

```
# 0 is GPU is
# 42 is experiment seed
scripts/screwdriver_teacher.sh 0 42 output_name
```

After training your oracle policy, you can visualize it as follows:
```
scripts/vis_screwdriver_teacher.sh 0 42 ckpt_name
```

## Step 2: Sensorimotor policy Training

In this section, we train a sensorimotor policy by distilling from our trained oracle policy $f$.

Note we use the proprioceptive adapt to train the sensorimotor policy.

```
scripts/screwdriver_student_padapt.sh 0 42 output_name
```

## Step 3: Rotational Policy deployment in Real Hardware

To generate the rotational policy from the student policy $\pi$, run
```
scripts/convert_student_jit.sh
```

To deploy rotational policy on real hardware, please refer to `./xhand-deploy`.

## Step 4: Real-world Fine-tuning

See the following repository: [skill-teleop](https://github.com/x-robotics-lab/skill-teleop)

## Acknowledgement

This repository is built based on [penspin](https://github.com/HaozhiQi/penspin/), [Hora](https://github.com/HaozhiQi/hora) and [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs), and supported in part by the program "Design of Robustly Implementable Autonomous and Intelligent Machines (TIAMAT)", Defense Advanced Research Projects Agency award number HR00112490425. We thank Mengda Xu for his valuable feedback.

## Citation

If you find **dexscrew** or this codebase helpful in your research, please cite:

```
@article{hsieh2025learning,
  title={Learning Dexterous Manipulation Skills from Imperfect Simulations},
  author={Hsieh, Elvis and Hsieh, Wen-Han and Wang, Yen-Jen and Lin, Toru and Malik, Jitendra and Sreenath, Koushil and Qi, Haozhi},
  journal={arXiv:2512.02011},
  year={2025}
}
```
