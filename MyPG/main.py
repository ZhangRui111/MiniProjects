import os
import time
from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate


def main():
    args = get_args()
    args.env_name = 'PongNoFrameskip-v4'
    # args.env_name = 'BreakoutNoFrameskip-v4'

    # # Necessary params for Windows
    args.num_processes = 1
    args.num_mini_batch = 4

    # args.algo = 'ppo'
    # args.use_gae = True
    # args.render = True
    # args.lr = 2.5e-4
    # args.clip_param = 0.1
    # args.value_loss_coef = 0.5
    # args.num_steps = 128
    # args.entropy_coef = 0.01
    # args.use_linear_lr_decay = True
    # args.num_processes = 8
    # args.num_mini_batch = 16

    print("{} GPUs in the system".format(torch.cuda.device_count()))
    torch.manual_seed(args.seed)  # Sets the seed on the current GPU
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(args.seed)  # Sets the seed on all GPUs

    # print(args.cuda, torch.cuda.is_available(), args.cuda_deterministic)
    # print(torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    if args.cuda and torch.cuda.is_available():
        if args.cuda_benchmark:
            torch.backends.cudnn.benchmark = True  # for speed
        if args.cuda_deterministic:
            torch.backends.cudnn.deterministic = True  # for replicate

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "eval"
    tb_log_dir = log_dir + "tb"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)
    utils.cleanup_log_dir(tb_log_dir)
    logger = SummaryWriter(tb_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # env(s) that wrapped by OpenAI/baselines
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, allow_early_resets=False)

    # build actor critic networks
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy}
    )
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm
        )
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm
        )
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            acktr=True
        )
    else:
        raise NotImplemented

    if args.gail:
        pass

    rollouts = RolloutStorage(
        args.num_steps,
        args.num_processes,
        envs.observation_space.shape,
        envs.action_space,
        actor_critic.recurrent_hidden_state_size
    )

    obs = envs.reset()

    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):
        # decrease learning rate linearly
        if args.use_linear_lr_decay:
            utils.update_linear_schedule(
                agent.optimizer, epoch=j, total_num_epochs=num_updates,
                initial_lr=agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step],
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step]
                )

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            # envs.render()

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.tensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            # Get V(s_{t+1})
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            pass

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        # print log info
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: "
                "mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
            logger.add_scalar('loss/value_loss', value_loss, total_num_steps)
            logger.add_scalar('loss/action_loss', action_loss, total_num_steps)
            logger.add_scalar('loss/dist_entropy', dist_entropy, total_num_steps)
            logger.add_scalar('episode_rewards/mean', np.mean(episode_rewards), total_num_steps)
            logger.add_scalar('episode_rewards/median', np.median(episode_rewards), total_num_steps)
            logger.add_scalar('episode_rewards/min', np.min(episode_rewards), total_num_steps)
            logger.add_scalar('episode_rewards/max', np.max(episode_rewards), total_num_steps)

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
