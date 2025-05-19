import os
import pickle
import sys

import time
import torch

import pandas as pd

from copy import deepcopy
from datetime import datetime
import torch.nn as nn
import torch.onnx

import wandb
from Environment import ESSEnv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import gc
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PrioritizedReplayBuffer:
    def __init__(self, max_len, num_agents, state_dim, action_dim, alpha=0.6, epsilon=1e-6):
        self.max_len = max_len
        self.ptr = 0
        self.size = 0
        self.num_agents = num_agents
        self.alpha = alpha
        self.epsilon = epsilon
        self.states = [np.zeros((max_len, state_dim[i])) for i in range(num_agents)]
        self.actions = [np.zeros((max_len, action_dim[i])) for i in range(num_agents)]
        self.rewards = np.zeros((max_len, num_agents))
        self.next_states = [np.zeros((max_len, state_dim[i])) for i in range(num_agents)]
        self.dones = np.zeros((max_len, num_agents))
        self.priorities = np.zeros((max_len,), dtype=np.float32)
        self.max_priority = 1.0

    def add(self, states, actions, rewards, next_states, dones):
        for i in range(self.num_agents):
            self.states[i][self.ptr] = states[i]
            self.actions[i][self.ptr] = actions[i]
            self.next_states[i][self.ptr] = next_states[i]
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.priorities[self.ptr] = self.max_priority
        self.ptr = (self.ptr + 1) % self.max_len
        self.size = min(self.size + 1, self.max_len)

    def sample(self, batch_size, beta=0.4):
        if self.size == 0:
            raise ValueError("The replay buffer is empty!")

        priorities = self.priorities[:self.size] ** self.alpha
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(self.size, batch_size, p=probabilities)
        experiences = (
            [self.states[i][indices] for i in range(self.num_agents)],
            [self.actions[i][indices] for i in range(self.num_agents)],
            self.rewards[indices],
            [self.next_states[i][indices] for i in range(self.num_agents)],
            self.dones[indices]
        )

        total = self.size
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        return experiences, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            priority_value = priority.item()  # 将 NumPy 数组转换为标量
            self.priorities[idx] = priority_value + self.epsilon
            self.max_priority = max(self.max_priority, priority_value + self.epsilon)

    def __len__(self):
        return self.size


class Arguments:
    def __init__(self, agent=None, env=None):
        self.agent = agent
        self.env = env
        self.cwd = None
        self.if_remove = False
        self.visible_gpu = '0,1,2,3'
        self.worker_num = 10
        self.num_threads = 8
        self.num_episode = 8000
        self.gamma = 0.995
        self.num_agents = 5
        self.state_dim = [8, 8, 6, 8, 8]  # 每个agent的状态维度
        self.action_dim = [3, 4, 2, 3, 4]  # 每个agent的动作维度0
        self.net_dim = 256
        self.batch_size = 24*300
        self.repeat_times = 2 ** 3
        self.target_step = 24*50        # 24*50
        self.max_memo = 24 * 2000
        self.if_per_or_gae = False
        self.update_training_data = True
        self.explorate_decay = 0.98 #0.98
        self.explore_noise = 0.1
        self.explore_rate = 1.0
        self.explorate_min = 0.3
        self.random_seed_list = [1234]
        now = datetime.now()
        self.run_name = 'experiments_' + str(now).replace(" ", "_").replace(":", "_").replace("-", "_").replace(".", "_")
        self.train = True
        self.save_network = True

    def init_before_training(self, if_main):
        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            self.cwd = f'./{agent_name}/{self.run_name}'

        if if_main:
            import shutil
            if self.if_remove is None:
                self.if_remove = bool(input(f"| PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
            elif self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print(f"| Remove cwd: {self.cwd}")
            os.makedirs(self.cwd, exist_ok=True)

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpu)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.attention = Attention(hidden_dim)      ###***
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        # self.relu = nn.ReLU()   ###***

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # x = self.attention(x.unsqueeze(1)).squeeze(1)      ###***
        action = torch.tanh(self.fc3(x))
        return action

class Critic(nn.Module):
    def __init__(self, state_dims, action_dims, num_agents, hidden_dim=128):
        super(Critic, self).__init__()
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.num_agents = num_agents

        input_dim = sum(state_dims) + sum(action_dims)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_heads = nn.Linear(hidden_dim, 1)
        # Independent heads
        # self.q_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_heads)])
    def forward(self, states, actions):
        # 确认输入的状态和动作的维度
        x = torch.cat([states, actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.q_heads(x)
        # q_values = torch.cat([q_head(x) for q_head in self.q_heads], dim=-1)  # Shape: (batch_size, num_heads)
        return q_values

class MASCAgent:
    def __init__(self, num_agents, state_dims, action_dims, hidden_dim, mixing_dim, actor_lrs, critic_lr,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=6, critic_num=4):
        self.tau = 0.001
        self.num_agents = num_agents
        self.num_critic = critic_num
        self.action_dim = action_dims
        self.state_dims = state_dims
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actors = [Actor(state_dims[i], action_dims[i], hidden_dim).to(self.device) for i in range(num_agents)]
        self.target_actors = [deepcopy(actor).to(self.device) for actor in self.actors]

        self.critics = [Critic(state_dims, action_dims, num_agents, hidden_dim).to(self.device) for _ in range(critic_num)]
        self.critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=critic_lr) for critic in self.critics]

        self.target_critics = [deepcopy(critic).to(self.device) for critic in self.critics]

        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=actor_lrs[i]) for i, actor in enumerate(self.actors)]

        self.criterion = nn.MSELoss()
        self.explore_noise = 0.1
        self.explore_rate = 1.0
        self.last_actor_loss = 0

    def select_action(self, states):
        actions = []
        for i, state in enumerate(states):
            if np.random.rand() < self.explore_rate:
                action = np.random.uniform(-1, 1, size=self.action_dim[i])
            else:
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                action = self.actors[i](state).detach().cpu().numpy()[0]
                action += np.random.normal(0, self.explore_noise, size=action.shape)
            action = np.round(action, 4)
            actions.append(np.clip(action, -1, 1))
        return actions

    def update(self, normal_buffer, batch_size, gamma, tau=0.001, critic_num=4):
        self.total_it += 1
        critic_losses = []
        actor_losses = []
        other_batch_size = batch_size

        # 从 normal_buffer 中抽取样本
        normal_samples, normal_indices, normal_weights = normal_buffer.sample(other_batch_size)
        normal_states, normal_actions, normal_rewards, normal_next_states, normal_dones = normal_samples

        states = [(torch.tensor(normal_states[i], dtype=torch.float32, device=self.device).clone().detach()) for i
                  in range(self.num_agents)]

        actions = [(torch.tensor(normal_actions[i], dtype=torch.float32, device=self.device).clone().detach()) for i
                   in range(self.num_agents)]

        rewards = (torch.tensor(normal_rewards, dtype=torch.float32, device=self.device).clone().detach())
        sum_rewards = torch.sum(rewards, dim=1).view(-1, 1)

        next_states = [
            (torch.tensor(normal_next_states[i], dtype=torch.float32, device=self.device).clone().detach()) for i in
            range(self.num_agents)]

        dones = (torch.tensor(normal_dones, dtype=torch.float32, device=self.device).clone().detach())

        weights = (normal_weights.clone().detach().to(self.device))
        first_dones = dones[:, 0].unsqueeze(1)
        with torch.no_grad():
            noise = [(torch.randn_like(actions[i]) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip) for
                     i in range(self.num_agents)]
            next_actions = [(self.target_actors[i](next_states[i]) + noise[i]).clamp(-1, 1) for i in
                            range(self.num_agents)]

            next_states_flat = torch.cat(next_states, dim=1)
            next_actions_flat = torch.cat(next_actions, dim=1)
            random_critics = random.sample(self.target_critics, 2)
            target_q_values_1 = random_critics[0](next_states_flat, next_actions_flat)
            # target_q_weight_1 = torch.rand((target_q_values_1.shape[0], num_heads)).to(device)
            # target_q_weight_1 = target_q_weight_1 / target_q_weight_1.sum(dim=-1, keepdim=True)
            # target_q_1 = (target_q_weight_1 * target_q_values_1).sum(dim=-1, keepdim=True)

            target_q_values_2 = random_critics[1](next_states_flat, next_actions_flat)
            # target_q_weight_2 = torch.rand((target_q_values_2.shape[0], num_heads)).to(device)
            # target_q_weight_2 = target_q_weight_2 / target_q_weight_2.sum(dim=-1, keepdim=True)
            # target_q_2 = (target_q_weight_2 * target_q_values_2).sum(dim=-1, keepdim=True)

            target_q_values = torch.min(target_q_values_1, target_q_values_2)
            target_q_values = sum_rewards + (1 - first_dones) * gamma * target_q_values

        states_flat = torch.cat(states, dim=1)
        actions_flat = torch.cat(actions, dim=1)

        current_q_values = [critic(states_flat, actions_flat) for critic in self.critics]
        critic_losses = [(weights * self.criterion(current_q_values[i], target_q_values.detach())).mean() for i in range(critic_num)]

        # 清零梯度
        for i, optimizer in enumerate(self.critic_optimizers):
            optimizer.zero_grad()
            critic_losses[i].backward()
            optimizer.step()

        if self.total_it % self.policy_freq == 0:
            pred_actions = [self.actors[i](states[i]) for i in range(self.num_agents)]
            pred_actions_flat = torch.cat(pred_actions, dim=1)
            q_values = [critic(states_flat, pred_actions_flat) for critic in self.critics]

            q_values1 = torch.min(torch.stack(q_values), dim=0).values
            q_values_sum = torch.sum(q_values1, dim=1)

            actor_loss = -torch.mean(q_values_sum)

            for optimizer in self.actor_optimizers:
                optimizer.zero_grad()
            actor_loss.backward()
            for optimizer in self.actor_optimizers:
                optimizer.step()
            actor_losses.append(actor_loss.item())

            self.last_actor_loss = actor_loss.item()  # 记录最新的actor_loss
        else:
            actor_losses.append(self.last_actor_loss)  # 沿用上一次的actor_loss

        avg_critic_loss = sum(critic_losses) / len(critic_losses)
        avg_actor_loss = sum(actor_losses) / len(actor_losses)
        with torch.no_grad():

            new_q_values = [critic(states_flat, actions_flat) for critic in self.critics]

            new_q_tot_stack = torch.stack(new_q_values, dim=0)

            new_q_tot_min = torch.min(new_q_tot_stack, dim=0)[0]

            new_td_errors = torch.abs(target_q_values - new_q_tot_min).cpu().detach().numpy()

        normal_buffer.update_priorities(normal_indices, new_td_errors[:other_batch_size])

        for i in range(self.num_agents):
            self.soft_update(self.target_actors[i], self.actors[i], tau)
        for i in range(critic_num):
            self.soft_update(self.target_critics[i], self.critics[i], tau)

        return avg_critic_loss, avg_actor_loss, critic_losses, actor_losses

    # 软更新
    def soft_update(self, target, source, tau = 0.001):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)



    def explore_env(self, env, target_step):
        trajectory = []
        for _ in range(target_step//24):
            # env.current_day = random.randint(0, 499)
            states = env.reset( is_test=False)
            for _ in range(24):
                actions = self.select_action(states)
                # corrected_actions是修正后的动作值
                next_states, rewards, dones, corrected_actions = env.step(actions)
                # 确保返回的是每个智能体的状态、奖励和 done
                trajectory.append((states, actions, rewards, next_states, dones))
                # print(trajectory)
                # 如果任何一个智能体达到 done 状态，重置环境
                if any(dones):
                    states = env.reset( is_test=False)
                else:
                    states = next_states
        return trajectory

    # 探索过程更新
    def _update_exploration_rate(self, explorate_decay, explore_rate_min):
        self.explore_rate = max(self.explore_rate * explorate_decay, explore_rate_min)

    def exchange_parameters(self, agent_pairs):
        for agent_1_idx, agent_2_idx in agent_pairs:
            # 获取两个智能体的参数
            actor_1_params = self.actors[agent_1_idx].state_dict()
            actor_2_params = self.actors[agent_2_idx].state_dict()

            # 计算平均参数
            average_params = {}
            for param_name in actor_1_params.keys():
                average_params[param_name] = 0.5 * (actor_1_params[param_name] + actor_2_params[param_name])

            # 更新智能体的参数为平均参数
            self.actors[agent_1_idx].load_state_dict(average_params)
            self.actors[agent_2_idx].load_state_dict(average_params)

def update_buffer(buffer, trajectory):
    for states, actions, rewards, next_states, dones in trajectory:
        buffer.add(states, actions, rewards, next_states, dones)
    _steps = len(trajectory)
    _r_exp = sum([sum(reward) for _, _, reward, _, _ in trajectory]) / _steps
    return _steps, _r_exp


def get_episode_return(env, agents, device):
    episode_return = 0.0
    original_day = env.current_day
    env.current_day = 615
    states = env.reset(is_test=False)
    original_explore_noise = agent.explore_noise
    original_explore_rate = agent.explore_rate
    # 临时设置 explore_noise 为 0
    agent.explore_noise = 0
    agent.explore_rate = 0
    for i in range(24):
        actions = agents.select_action(states)
        next_states, rewards, dones, corrected_actions = env.step(actions)
        states = next_states
        episode_return = np.sum(rewards) + episode_return
        if all(dones):
            break
    # 恢复 explore_noise 的原值
    agent.explore_noise = original_explore_noise
    agent.explore_rate = original_explore_rate
    env.current_day = original_day
    return episode_return  #episode_unbalance, episode_operation_cost


def get_episode_return_test(env, agent, device, day, prefix=""):
    env.current_day = day
    states = env.reset(is_test=True)
    episode_return = 0.0

    # 动态生成列名
    columns = ['agent','time_step', 'Reward']
    state_dims = [8, 8, 6, 8, 8]  # 每个智能体的状态维度
    action_dims = [3,4,2,3,4]
    for i in range(5):
        for h in range(action_dims[i]):
            columns.append(f'Action_{i}_{h}')
    for i in range(5):
        for j in range(state_dims[i]):
            columns.append(f'state_{i}_{j}')
    for i in range(5):
        for j in range(state_dims[i]):
            columns.append(f'next_state_{i}_{j}')

    # save_data = pd.DataFrame(columns=columns)
    all_rows = []
    for i in range(24):
        actions = agent.select_action(states)
        next_states, rewards, dones, corrected_actions = env.step(actions)

        for agent_idx in range(len(states)):
            state = states[agent_idx]
            corrected_action = corrected_actions[agent_idx]
            next_state = next_states[agent_idx]
            reward = rewards[agent_idx]
            # done = dones[agent_idx]

            new_row = {'agent':agent_idx, 'time_step': i,   'Reward': reward}
            for h in range(action_dims[agent_idx]):
                new_row[f'Action_{agent_idx}_{h}'] = corrected_action[h]
            for j in range(state_dims[agent_idx]):
                new_row[f'state_{agent_idx}_{j}'] = state[j]
            for j in range(state_dims[agent_idx]):
                new_row[f'next_state_{agent_idx}_{j}'] = next_state[j]
            all_rows.append(new_row)
            # print(f'state:{state}, next_state:{next_state}, reward:{reward}, corrected_actions:{actions}')

        states = next_states
        episode_return += sum(rewards)  # 累积每个智能体的奖励

        if all(dones):
            break
    print("文件已经测试完成", day)
    # 创建 DataFrame
    save_data = pd.DataFrame(all_rows, columns=columns)
    save_data.to_csv(f"result/{prefix}output.csv", index=False)
    print("training data have been saved")
    print(episode_return)


if __name__ == '__main__':
    train_flag = True
    num_agents = 5
    plot_rewards = []
    plot_average_rewards = []
    best_average_reward = -100
    stable_steps = 0
    i_episode_count = 0
    critic_num = 8
    if train_flag:
        os.environ["WANDB_DISABLED"] = "true"
        args = Arguments()
        reward_record = {'episode': [], 'steps': [], 'mean_episode_reward': [], 'episode_operation_cost_agent_0': [],
                         'episode_operation_cost_agent_1': [], 'episode_operation_cost_agent_2': [], "time_gap": []}

        loss_record = {'episode': [], 'steps': [], 'critic_loss': [], 'actor_loss': [], 'entropy_loss': []}

        args.visible_gpu = '0'
        results = []
        args.random_seed = 1234
        args.env = ESSEnv()

        args.agent = MASCAgent(
            num_agents=num_agents,
            state_dims=[8, 8, 6, 8, 8],  # 每个agent的状态维度
            action_dims=[3, 4, 2, 3, 4],  # 每个agent的动作维度
            hidden_dim=args.net_dim,
            mixing_dim=256,
            actor_lrs=[1e-4, 1e-4, 1e-4, 1e-4, 1e-4],  # 每个actor的学习率
            critic_lr=1.5e-4,
            critic_num=critic_num
        )

        start_time = time.time()

        args.init_before_training(if_main=True)

        agent = args.agent
        env = args.env
        normal_buffer = PrioritizedReplayBuffer(max_len=args.max_memo, num_agents=args.num_agents, state_dim=[8, 8, 6, 8, 8],
                                                action_dim=[3, 4, 2, 3, 4])

        agent.state = env.reset(is_test=False)

        num_episode = args.num_episode
        args.train = True
        args.save_network = True

        wandb.init(project='REDQ_experiments', name=args.run_name)
        wandb.config = {
            "epochs": num_episode,
            "batch_size": args.batch_size}
        wandb.define_metric('custom_step')


        if args.train:
            collect_data = True

            while collect_data:
                print(f'buffer size: {normal_buffer.size}')
                with torch.no_grad():
                    trajectory = agent.explore_env(env, args.target_step)
                    steps, r_exp = update_buffer(normal_buffer, trajectory)
                if normal_buffer.size >= 24 * 800:
                    collect_data = False

            best_episode_reward = -float('inf')
            best_actor_state_dicts = [None] * num_agents
            best_critic_state_dict = [None] * critic_num
            best_params = None

            for i_episode in range(num_episode):
                episode_start_time = time.time()

                total_critic_loss = 0
                total_actor_loss = 0
                individual_actor_losses = []

                for _ in range(args.repeat_times):
                    avg_critic_loss, avg_actor_loss, critic_losses, actor_losses = agent.update(normal_buffer,args.batch_size,gamma=args.gamma,critic_num=critic_num)
                    total_critic_loss += avg_critic_loss
                    total_actor_loss += avg_actor_loss
                    individual_actor_losses.extend(actor_losses)

                total_critic_loss /= args.repeat_times
                total_actor_loss /= args.repeat_times

                loss_record['critic_loss'].append(total_critic_loss)
                loss_record['actor_loss'].append(total_actor_loss)

                with torch.no_grad():
                    episode_reward = get_episode_return(env, agent, agent.device)
                    current_time = time.time()
                    time_gap = current_time - start_time
                    plot_rewards.append(episode_reward)
                    plot_average_rewards.append(np.mean(plot_rewards[-50:]))  # 计算最近50个回合的平均奖励

                    if episode_reward > best_average_reward:
                        best_average_reward = episode_reward
                        best_actor_state_dicts = [deepcopy(actor.state_dict()) for actor in agent.actors]
                        best_critic_state_dict = [deepcopy(critic.state_dict()) for critic in agent.critics]
                        # 保存最佳模型参数
                        best_params = (best_actor_state_dicts, best_critic_state_dict)

                    if i_episode % 20 == 0:
                        wandb.log({
                            'mean_episode_reward': episode_reward,
                            'time_gap': time_gap,
                            'total_critic_loss': total_critic_loss,
                            'total_actor_loss': total_actor_loss
                        })
                        for agent_idx in range(args.num_agents):
                            wandb.log({f'agent_{agent_idx}_actor_loss': individual_actor_losses[agent_idx]})

                    reward_record['mean_episode_reward'].append(episode_reward)
                    reward_record['time_gap'].append(time_gap)

                print(
                    f'current episode is {i_episode}, reward: {episode_reward}, buffer_length: {normal_buffer.size}, time_gap: {time_gap}')

                if i_episode % 20 == 0:
                    with torch.no_grad():
                        if i_episode < 5500: #5500:
                            agent._update_exploration_rate(args.explorate_decay, 0.5)
                        elif 5500 <= i_episode <= 7500:
                            agent._update_exploration_rate(args.explorate_decay, 0.3)  # 0
                        # if i_episode < 7000 and i_episode % 60 == 0: #5500:
                        #     agent._update_exploration_rate(args.explorate_decay, 0.2)
                        elif 7500 <= i_episode <= 8000:
                            value = 0.3 * 0.5 * (1+np.cos(np.pi * (i_episode - 7500) / 500))
                            agent._update_exploration_rate(args.explorate_decay, value)
                        else:
                            agent._update_exploration_rate(args.explorate_decay, 0.001)

                        trajectory = agent.explore_env(env, args.target_step)
                        update_buffer(normal_buffer, trajectory)

                    torch.cuda.empty_cache()
                    gc.collect()

        wandb.finish()

        # 绘制奖励图表
        plt.figure(figsize=(10, 5))
        plt.plot(plot_rewards, label='Total Rewards')
        # plt.plot(plot_average_rewards, label='Average Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.title('Total and Average Rewards per Episode')
        plt.savefig('rewards_plot.pdf',  format='pdf')
        plt.show()
        plot_rewards = np.array(plot_rewards)  # 如果它已经是 NumPy 数组，可以忽略这行
        df = pd.DataFrame(plot_rewards, columns=['Average Reward'])
        df.to_excel('plot_rewards.xlsx', index=False)
        plot_average_rewards = np.array(plot_average_rewards)  # 如果它已经是 NumPy 数组，可以忽略这行
        df = pd.DataFrame(plot_average_rewards, columns=['Average Reward'])
        df.to_excel('plot_average_rewards.xlsx', index=False)

        get_episode_return_test(env, agent, agent.device, 600, prefix="600")
        get_episode_return_test(env, agent, agent.device, 601, prefix="601")
        get_episode_return_test(env, agent, agent.device, 602, prefix="602")
        get_episode_return_test(env, agent, agent.device, 603, prefix="603")
        get_episode_return_test(env, agent, agent.device, 604, prefix="604")
        get_episode_return_test(env, agent, agent.device, 605, prefix="605")
        get_episode_return_test(env, agent, agent.device, 606, prefix="606")
        get_episode_return_test(env, agent, agent.device, 607, prefix="607")
        get_episode_return_test(env, agent, agent.device, 608, prefix="608")
        get_episode_return_test(env, agent, agent.device, 609, prefix="609")
        get_episode_return_test(env, agent, agent.device, 610, prefix="610")
        get_episode_return_test(env, agent, agent.device, 611, prefix="611")
        get_episode_return_test(env, agent, agent.device, 612, prefix="612")
        get_episode_return_test(env, agent, agent.device, 613, prefix="613")
        get_episode_return_test(env, agent, agent.device, 614, prefix="614")
        get_episode_return_test(env, agent, agent.device, 615, prefix="615")
        get_episode_return_test(env, agent, agent.device, 616, prefix="616")
        get_episode_return_test(env, agent, agent.device, 617, prefix="617")
        get_episode_return_test(env, agent, agent.device, 618, prefix="618")
        get_episode_return_test(env, agent, agent.device, 619, prefix="619")


        # 保存网络参数
        act_save_path = f'{args.cwd}/actor.pth'
        cri_save_path = f'{args.cwd}/critic.pth'
        for agent_idx in range(num_agents):
            torch.save(agent.actors[agent_idx].state_dict(), f'{act_save_path}_{agent_idx}')
        for i in range(critic_num):
            torch.save(agent.critics[i].state_dict(), f'{cri_save_path}_{i}')
        print('training finished and actor and critic parameters have been saved')



        # 加载最优模型参数
        for idx, actor in enumerate(agent.actors):
            actor.load_state_dict(best_actor_state_dicts[idx])
        for idx, critic in enumerate(agent.critics):
            critic.load_state_dict(best_critic_state_dict[idx])

        # 测试最优模型
        get_episode_return_test(env, agent, agent.device, 600, prefix="best600")
        get_episode_return_test(env, agent, agent.device, 601, prefix="best601")
        get_episode_return_test(env, agent, agent.device, 602, prefix="best602")
        get_episode_return_test(env, agent, agent.device, 603, prefix="best603")
        get_episode_return_test(env, agent, agent.device, 604, prefix="best604")
        get_episode_return_test(env, agent, agent.device, 605, prefix="best605")
        get_episode_return_test(env, agent, agent.device, 606, prefix="best606")
        get_episode_return_test(env, agent, agent.device, 607, prefix="best607")
        get_episode_return_test(env, agent, agent.device, 608, prefix="best608")
        get_episode_return_test(env, agent, agent.device, 609, prefix="best609")
        get_episode_return_test(env, agent, agent.device, 610, prefix="best610")
        get_episode_return_test(env, agent, agent.device, 611, prefix="best611")
        get_episode_return_test(env, agent, agent.device, 612, prefix="best612")
        get_episode_return_test(env, agent, agent.device, 613, prefix="best613")
        get_episode_return_test(env, agent, agent.device, 614, prefix="best614")
        get_episode_return_test(env, agent, agent.device, 615, prefix="best615")
        get_episode_return_test(env, agent, agent.device, 616, prefix="best616")
        get_episode_return_test(env, agent, agent.device, 617, prefix="best617")
        get_episode_return_test(env, agent, agent.device, 618, prefix="best618")
        get_episode_return_test(env, agent, agent.device, 619, prefix="best619")

        # 保存训练数据
        if args.update_training_data:
            loss_record_path = f'{args.cwd}/loss_data.pkl'
            reward_record_path = f'{args.cwd}/reward_data.pkl'
            with open(loss_record_path, 'wb') as tf:
                pickle.dump(loss_record, tf)
            with open(reward_record_path, 'wb') as tf:
                pickle.dump(reward_record, tf)
        act_save_path = f'{args.cwd}/actor.pth'
        cri_save_path = f'{args.cwd}/critic.pth'

        print('training data have been saved')



        # 保存最高奖励及相关参数
        best_params_path = f'{args.cwd}/best_params.pkl'
        best_params_data = {
            'best_actor_state_dicts': [actor.state_dict() for actor in agent.actors],
            'best_critic_state_dict': [critic.state_dict() for critic in agent.critics],
        }
        with open(best_params_path, 'wb') as f:
            pickle.dump(best_params_data, f)
        print('Best parameters have been saved')
        sys.exit()
