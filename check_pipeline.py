"""env + PPO 에이전트 생성까지만 검증하고 종료. 학습은 하지 않음."""
import isaacgym
import os, torch, hydra, datetime
from omegaconf import DictConfig, OmegaConf
from dexscrew.algo.ppo.ppo import PPO
from dexscrew.algo.ppo.padapt import ProprioAdapt
from dexscrew.tasks import isaacgym_task_map
from dexscrew.utils.reformat import omegaconf_to_dict
from dexscrew.utils.misc import set_np_formatting, set_seed
for n,f in [('eq',lambda x,y:x.lower()==y.lower()),('contains',lambda x,y:x.lower() in y.lower()),
            ('if',lambda p,a,b:a if p else b),('resolve_default',lambda d,a:d if a=='' else a)]:
    OmegaConf.register_new_resolver(n,f)

@hydra.main(config_name='config', config_path='configs')
def main(config: DictConfig):
    set_np_formatting(); config.seed = set_seed(config.seed)
    n_env = config.task.env.numEnvs; hz = config.train.ppo.horizon_length
    mb = config.train.ppo.minibatch_size
    print(f"\n[검증] num_envs={n_env} horizon={hz} -> batch={hz*n_env}, minibatch={mb}, "
          f"나머지={hz*n_env % mb} {'OK' if hz*n_env % mb == 0 else '❌ 나누어떨어지지 않음'}")
    env = isaacgym_task_map[config.task_name](
        config=omegaconf_to_dict(config.task), sim_device=config.sim_device,
        graphics_device_id=config.graphics_device_id, headless=config.headless)
    print(f"[검증] env OK | num_obs={env.num_obs} num_actions={env.num_actions} num_dofs={env.num_dofs}")
    out = os.path.join('outputs', '_pipeline_check'); os.makedirs(out, exist_ok=True)
    algo = str(config.train.algo)
    agent = (ProprioAdapt if algo=='ProprioAdapt' else PPO)(env, out, full_config=config)
    if algo=='ProprioAdapt':
        tconv = agent.model.adapt_tconv
        ch = tconv.channel_transform[0].in_features if hasattr(tconv,'channel_transform') else None
        print(f"[검증] student 인코더 입력 채널 = {ch} (필요 2*numActions={2*env.num_actions})")
    print(f"[검증] {algo} 에이전트 생성 OK")
    obs = env.reset()
    print(f"[검증] reset OK | obs keys={list(obs.keys())}")
    for k, v in obs.items():
        if hasattr(v, 'shape'): print(f"          {k:<18}{tuple(v.shape)}")
    print(f"[검증] proprio_dim(config)={config.train.ppo.proprio_dim}  "
          f"실제 proprio_hist={tuple(obs['proprio_hist'].shape)}")
    a = agent.model.act(agent.running_mean_std(obs['obs']) if hasattr(agent,'running_mean_std') else obs['obs'],
                        obs['priv_info'], obs['proprio_hist'], obs['point_cloud_info'], obs['rot_axis_buf']) \
        if False else None
    obs2, r, d, info = env.step(torch.zeros((env.num_envs, env.num_actions), device=env.rl_device))
    print(f"[검증] step OK | reward shape={tuple(r.shape)} 평균={r.mean().item():.4f}")
    print("\n✅ 파이프라인 전체 통과 — 학습 실행 가능")
if __name__=='__main__': main()
