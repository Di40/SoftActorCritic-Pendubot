import os
import matplotlib.pyplot as plt
import tensorflow as tf

def extract_scalars_from_event_file(event_file_path):
    scalar_data_dict = {}
    for event in tf.compat.v1.train.summary_iterator(event_file_path):
        for value in event.summary.value:
            if value.tag not in scalar_data_dict:
                scalar_data_dict[value.tag] = []
            scalar_data_dict[value.tag].append((event.step, value.simple_value))
    return scalar_data_dict

def get_baseline_reward():
    return 103320.86


robot = 'pendubot'
log_dir = os.path.join(os.getcwd(), 'log_data', robot + '_training', 'tb_logs', 'SAC_1')
files = os.listdir(log_dir)
event_filename = files[0] if files else None

if event_filename:
    path_to_event_file = os.path.join(log_dir, event_filename)
    # print(path_to_event_file)
    
    scalar_data = extract_scalars_from_event_file(path_to_event_file)
    
    # Extract data for both tags
    time_rollout, value_rollout = [], []
    time_eval, value_eval = [], []
    
    for tag, values in scalar_data.items():
        if tag == 'rollout/ep_rew_mean':
            time_rollout, value_rollout = zip(*values)
        elif tag == 'eval/mean_reward':
            time_eval, value_eval = zip(*values)
    
    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    baseline_reward = get_baseline_reward()
    
    # Plot rollout/ep_rew_mean
    if time_rollout and value_rollout:
        axs[0].plot(time_rollout, value_rollout, label='Mean Reward')
        axs[0].axhline(y=0, color='k', linewidth=0.5)
        axs[0].axhline(y=baseline_reward, color='r', linestyle='--', label=f'Baseline Reward ({baseline_reward:.2e})')
        axs[0].text(x=max(time_rollout), y=baseline_reward*1.1, s=f'{baseline_reward:.2e}', color='r', ha='right', va='bottom')
        axs[0].set_ylabel('Mean Reward')
        axs[0].set_title('Training Progress')
        axs[0].legend()
        axs[0].grid(True)
        axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        
    # Plot eval/mean_reward
    if time_eval and value_eval:
        axs[1].plot(time_eval, value_eval, label='Mean Reward (Eval)', color='g')
        axs[1].axhline(y=0, color='k', linewidth=0.5)
        axs[1].axhline(y=baseline_reward, color='r', linestyle='--', label=f'Baseline Reward ({baseline_reward:.2e})')
        axs[1].text(x=max(time_eval), y=baseline_reward*0.7, s=f'{baseline_reward:.2e}', color='r', ha='right', va='bottom')
        axs[1].set_xlabel('Step')
        axs[1].set_ylabel('Mean Reward (Eval)')
        axs[1].set_title('Evaluation Progress')
        axs[1].legend()
        axs[1].grid(True)
        axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    
    # Save the figure
    destination_folder = os.path.join(os.getcwd(), 'results')
    os.makedirs(destination_folder, exist_ok=True)
    output_path = os.path.join(destination_folder, 'training_progress.png')
    plt.savefig(output_path)
    print(f"Figure saved to {output_path}")
else:
    print("No event files found in the directory.")
