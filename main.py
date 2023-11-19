import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
#%matplotlib inline #for google colab

import DDPG
import utils

import environment


def whiten(state):
    return (state - np.mean(state)) / np.std(state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Choose the type of the experiment
    parser.add_argument('--experiment_type', default='custom', choices=['custom', 'power', 'rsi_elements', 'learning_rate', 'decay'],
                        help='Choose one of the experiment types to reproduce the learning curves given in the paper')

    # Training-specific parameters
    parser.add_argument("--policy", default="DDPG", help='Algorithm (default: DDPG)')
    parser.add_argument("--env", default="RIS_DDPG", help='OpenAI Gym environment name')
    parser.add_argument("--seed", default=0, type=int, help='Seed number for PyTorch and NumPy (default: 0)')
    parser.add_argument("--gpu", default="0", type=int, help='GPU ordinal for multi-GPU computers (default: 0)')
    parser.add_argument("--start_time_steps", default=0, type=int, metavar='N', help='Number of exploration time steps sampling random actions (default: 0)')
    parser.add_argument("--buffer_size", default=1000, type=int, help='Size of the experience replay buffer (default: 100000)')
    parser.add_argument("--batch_size", default=16, metavar='N', help='Batch size (default: 16)')
    parser.add_argument("--save_model", action="store_true", help='Save model and optimizer parameters')
    parser.add_argument("--load_model", default="", help='Model load file name; if empty, does not load')

    # Environment-specific parameters
    parser.add_argument("--num_RIS", default=2, type=int, metavar='N', help='Number of antennas in the BS')
    parser.add_argument("--num_RIS_elements", default=10, type=int, metavar='N', help='Number of RIS elements')
    parser.add_argument("--num_users", default=2, type=int, metavar='N', help='Number of users')
    parser.add_argument("--power_t", default=0, type=float, metavar='N', help='Transmission power for the constrained optimization in dBm (default: 30)')
    parser.add_argument("--num_time_steps_per_eps", default=500, type=int, metavar='N', help='Maximum number of steps per episode (default: 10000)')
    parser.add_argument("--num_eps", default=10, type=int, metavar='N', help='Maximum number of episodes (default: 5000)')
    parser.add_argument("--awgn_var", default=1e-2, type=float, metavar='G', help='Variance of the additive white Gaussian noise (default: 1e-2)')
    parser.add_argument("--channel_est_error", default=False, type=bool, help='Noisy channel estimate? (default: False)')
    parser.add_argument("--bandwidth", default=2*1000000, type=float, help='Channel bandwidth (default: 5 MHz)')

    # Algorithm-specific parameters
    parser.add_argument("--exploration_noise", default=0.0, metavar='G', help='Std of Gaussian exploration noise')
    parser.add_argument("--discount", default=0.99, metavar='G', help='Discount factor for reward (default: 0.99)')
    parser.add_argument("--tau", default=1e-3, type=float, metavar='G',  help='Learning rate in soft/hard updates of the target networks (default: 0.001)')
    parser.add_argument("--lr", default=1e-3, type=float, metavar='G', help='Learning rate for the networks (default: 0.001)')
    parser.add_argument("--decay", default=1e-5, type=float, metavar='G', help='Decay rate for the networks (default: 0.00001)')

    args = parser.parse_args()

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    file_name = f"{args.num_RIS}_{args.num_RIS_elements}_{args.num_users}_{args.power_t}_{args.lr}_{args.decay}"

    if not os.path.exists(f"./Learning Curves/{args.experiment_type}"):
        os.makedirs(f"./Learning Curves/{args.experiment_type}")

    if args.save_model and not os.path.exists("./Models"):
        os.makedirs("./Models")

    env = environment.RIS_DDPG(args.num_RIS, args.num_RIS_elements, args.num_users, args.power_t, AWGN_var=args.awgn_var, bandwidth=args.bandwidth)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = 1

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "power_t": args.power_t,
        "max_action": max_action,
        "M": args.num_RIS,
        "N": args.num_RIS_elements,
        "K": args.num_users,
        "bandwidth": args.bandwidth,
        "actor_lr": args.lr,
        "critic_lr": args.lr,
        "actor_decay": args.decay,
        "critic_decay": args.decay,
        "device": device,
        "discount": args.discount,
        "tau": args.tau
    }

    # Initialize the algorithm
    agent = DDPG.DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        agent.load(f"./models/{policy_file}")

    replay_buffer = utils.ExperienceReplayBuffer(state_dim, action_dim, max_size=args.buffer_size)
    #replay_buffer = ReplayBuffer(max_size=args.buffer_size)

    # Initialize the instant rewards recording array
    instant_rewards = []

    max_reward = 0

    # Define the interval for saving aggregated statistics ---edatsika
    save_interval = 1
    rho_k_log = []
    theta_kmn_log = []
    sum_rate_log = []
    cumulative_rewards = []

    for eps in range(int(args.num_eps)):
        state, done = env.reset(), False
        episode_reward = 0
        episode_num = 0
        episode_time_steps = 0

        state = whiten(state)

        eps_rewards = []

        #edatsika
        episode_rho_k_log = []
        episode_theta_kmn_log = []
        episode_sum_rate = 0
        episode_rewards = []
        cumulative_reward = 0

        for t in range(int(args.num_time_steps_per_eps)):
            # Choose action from the policy
            action = agent.select_action(np.array(state))
            #print(f">>>>>>>>>>>>>>>>>>>>>> Episode {eps}, Step {t}, Action shape {action.shape}, Action before: {action[:, :env.K]}")
            #print(f">>>>>>>>>>>>>>>>>>>>>> State shape {state.shape}, State before: {state[:env.K]}")
            #if np.any(state[:env.K] < 0):
            #    input("Press Enter to continue...")
            # Take the selected action
            next_state, reward, done, _ = env.step(action, args.power_t)
            #print(f">>>>>>>>>>>>>>>>>>>>>> Episode {eps}, Step {t}, Action shape {action.shape}, Action after env.step in main: {action}")
            #print(f">>>>>>>>>>>>>>>>>>>>>> State shape {state.shape}, State after env.step in main: {state[:env.K]}")
            #print("rho_k values after step:", env.rho_k)
            #input("Press Enter to continue...")
            #print("theta_kmn values after step:", env.theta_kmn)

            #edatsika
            episode_rho_k_log.append(env.rho_k)
            episode_theta_kmn_log.append(env.theta_kmn)

            nan_indices = np.isnan(action)
            if nan_indices.any():
                nan_positions = np.argwhere(nan_indices)
                #print("AFTER STEP NaN values found at positions:", nan_positions)
                print("rho_k values right after step:", env.rho_k)
                input("NaN - Press Enter to continue...")
            
            done = 1.0 if t == args.num_time_steps_per_eps - 1 else float(done)

            # Store data in the experience replay buffer
            replay_buffer.add(state, action, next_state, reward, done)

            state = next_state
            episode_reward += reward
            #edatsika
            episode_sum_rate += reward

            state = whiten(state)

            if reward > max_reward:
                max_reward = reward

            # Train the agent
            agent.update_parameters(replay_buffer, args.batch_size)

            #print(f"Time step: {t + 1} Episode Num: {episode_num + 1} Reward: {reward:.3f}")

            eps_rewards.append(reward)

            episode_time_steps += 1

            #edatsika
            #cumulative_reward += max_reward

            if done:
                #print(f"\nTotal T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_time_steps} Max. Reward: {max_reward:.3f}\n")
                #print(f"\nEpisode Num: {eps} Episode T: {t} Max. Reward: {max_reward:.3f}\n")
                # Reset the environment
                state, done = env.reset(), False
                episode_reward = 0
                episode_time_steps = 0
                episode_num += 1

                state = whiten(state)

                instant_rewards.append(eps_rewards)

                # commented by edatsika
                #np.save(f"./Learning Curves/{args.experiment_type}/{file_name}_episode_{episode_num + 1}", instant_rewards)
        
        cumulative_rewards.append(episode_reward)
    
        rho_k_log.append(episode_rho_k_log)
        theta_kmn_log.append(episode_theta_kmn_log)
        sum_rate_log.append(episode_sum_rate)
        print(f"\nTotal T steps completed: {t} Episode Num: {eps} Max. Reward: {max_reward:.6f}\n")
        
        # Save aggregated statistics every N episodes
        """episode_rewards.append(cumulative_reward)
        if (episode_num + 1) % save_interval == 0:
            mean_reward = np.mean(episode_rewards[-save_interval:])  # Calculate mean of last N episodes
            std_reward = np.std(episode_rewards[-save_interval:])    # Calculate standard deviation
            aggregated_stats = {
                'episode': episode_num + 1,
                'mean_reward': mean_reward,
                'std_reward': std_reward
            }
            #np.save('aggregated_stats.npy', aggregated_stats)
            try:
                np.save(f"./Learning Curves/{args.experiment_type}/{file_name}_episode_{episode_num + 1}", aggregated_stats)
            except Exception as e:
                # Print the exception or handle it as needed
                print(f"Error saving results: {e}")"""
    
    #edatsika
    # After training, identify the episode with the maximum sum_rate
    max_sum_rate_episode = np.argmax(sum_rate_log) #------------------- print optimal values

    # Extract the corresponding values of rho_k and theta_kmn
    max_rho_k = rho_k_log[max_sum_rate_episode]
    max_theta_kmn = theta_kmn_log[max_sum_rate_episode]

    # Print or use the values as needed
    print(f"Max_sum_rate_episode: {max_sum_rate_episode}")
    print(f"Max Sum Rate: {sum_rate_log[max_sum_rate_episode]}")
    print(f"Optimal rho_k: {max_rho_k[max_sum_rate_episode]}")
    #print(f"Optimal theta_kmn: {max_theta_kmn}")

    #file_path = f"./Learning Curves/{args.experiment_type}/{file_name}_episode_{episode_num + 1}.npy"
    #loaded_data = np.load(file_path, allow_pickle=True) put back causes error
    #print(loaded_data)
    
    # edatsika
    # Convert the rewards list to a NumPy array for further analysis or plotting
    #rewards_array = np.array(instant_rewards)
    #print("Rewards array:", rewards_array.shape)
    #print("Rewards array:", cumulative_rewards)

    # Create x-axis values for all steps and episodes
    # Flatten rewards_array
    #flat_rewards = rewards_array.flatten()
    # Plot the rewards for all steps and episodes
    #x_values = np.arange(flat_rewards.shape[0])
    #plt.plot(x_values, flat_rewards)
    #plt.xlabel("Step and Episode")
    #plt.ylabel("Reward")
    #plt.title("Reward for All Steps and Episodes")
    #plt.show()

    # Plot cumulative reward over episodes
    #plt.plot(range(int(args.num_eps)), instant_rewards)#cumulative_rewards
    avg_reward = np.zeros_like(instant_rewards)

    for i in range(len(instant_rewards)):
        avg_reward[i] = np.sum(instant_rewards[:(i + 1)]) / (i + 1)
    #plt.plot(range(len(instant_rewards)), instant_rewards)


    plt.plot(range(len(avg_reward)), avg_reward)
    #plt.plot(range(len(cumulative_rewards)), cumulative_rewards)#cumulative_rewards
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward over Episodes")
    plt.savefig("plot.png")
    plt.show()

    if args.save_model:
        agent.save(f"./Models/{file_name}")
