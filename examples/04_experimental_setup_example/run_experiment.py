import os
import time
import argparse
from datetime import datetime
import gc
import torch   # You will need to install torch separately. I used to to clean up the VRAM.

import llm_layer.layer_LLM as layer_LLM
from tqdm import tqdm


from dotenv import load_dotenv
from openai import OpenAI
from control_layer.simulation import RobotAgent
from parser_layer import save_behavior_tree_with_metadata

load_dotenv()

# Model name, hf url, and chat format. Only run 1B models in this example for brevity. 
# If you add 4B and 12B you get the setup for my thesis results.
models_v = [   
    ["Gemma_3_1B_QAT", "hf.co/bartowski/google_gemma-3-1b-it-qat-GGUF:Q8_0", "gemma-3"],
           
]

models_a = [    
    ["Gemma_3_1B_QAT_A", "hf.co/Andresgr96/gemma-3-1b-it-qat-A:Q8_0", "gemma-3"],
          
]

models_b = [    
    ["Gemma_3_1B_QAT_B", "hf.co/Andresgr96/gemma-3-1b-it-qat-B:Q8_0", "gemma-3"],    
]

agents = ["RobotAgent"]
envs = ["RobotEnvironment"]

task_prompt_style = ["layman", 
                     "technical", 
                     "spoonfed"
                     ]

tasks = [
    ["find", """Find a good part and pick it up""", 
             """Find a good part. If you find one, pick it up.""",
            """Find a good part. If you detect a good part, pick up the part."""],

    ["clean", """Find as many scrap parts as you can and bring them to the waste""",
              """Find scrap parts, if you find one bring them to the waste. If you are in the waste then drop it there.""",
              """Find scrap parts. If you are holding a scrap part and are in the waste area then drop the part. If you are holding a scrap part then seek the waste area.
                    If you find a scrap part then pick it up. Otherwise walk randomly."""],

    ["maintain", """Find as many parts as you can, bring good parts to the storage, while taking any scrap parts you find to the waste""",
                 """Find all the parts in the environment. If you find a good part, bring it to the storage. If you find a scrap part, bring it to the waste.""",
                 """Find all the parts in the environment. If you are holding a good part and are in the storage area then drop the part. If you are holding a good part then seek the storage area.
                        If you detect a good part then pick it up. If you are holding a scrap part and are in the waste area then drop the part. If you are holding a scrap part then seek the waste area. 
                        If you detect a scrap part then pick it up. Otherwise seek the source area."""]
]


prompt_techniques = [
    ["zero_shot", 1],
    ["one_shot", 2],
    ["two_shot", 3]
]


def run_experiment(models: list, backend: str, n_trials: int, result_dir: str, seed: int) -> None:
    """
    Run behavior tree generation experiments across multiple models, tasks, and configurations.
    
    Args:
        models: List of model configurations, each containing [name, path/url, chat_format]
        backend: Backend to use for LLM inference ("ollama" or "llamacpp")
        n_trials: Number of trials to run for each configuration
        result_dir: Directory to save the generated behavior trees
        seed: Random seed for reproducible results
        
    Returns:
        None - Results are saved to files in result_dir
    """
    total_trials = len(models) * len(tasks) * len(task_prompt_style) * len(prompt_techniques) * n_trials 
    agent_class = agents[0]
    env_name = envs[0]
    
    client = OpenAI()

    # Create progress bar for all trials
    with tqdm(total=total_trials, desc="Running experiments", unit="trial") as pbar:
        model = None
        for model_name, model_path, chat_format in models:
            # Explicitly delete the previous model to free up VRAM
            if model is not None:
                del model
                gc.collect()
                if backend == "llamacpp" and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Updated config to match new BehaviorTreeGenerator parameters
            config = {
                "agent_class": RobotAgent,
                "backend": backend,
                "model_path_or_url": model_path,
                "chat_format": chat_format,
                "context_length": (1024*4),
                "gpu_layers": -1, 
                "n_threads": os.cpu_count(),   
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 64,
                "repeat_penalty": 1.1,
                "seed": seed,
                "client": client,
            }

            model = layer_LLM.BehaviorTreeGenerator(**config)

            for task in tasks:
                task_name = task[0]
                for style_index, style_name in enumerate(task_prompt_style):
                    prompt = task[style_index + 1] 
                    for technique, tech_number in prompt_techniques:
                        for trial_nr in range(n_trials):
                            # Update progress bar description with current trial info
                            pbar.set_description(f"Model: {model_name} | Task: {task_name} | Style: {style_name} | Technique: {technique}")
                            
                            file_name = f"{model_name}_{task_name}_{style_name}_{technique}_trial{trial_nr + 1}.xml"
                            bt_result_path = os.path.join(result_dir, file_name)
                            
                            # Run the model
                            _, behavior_tree = model.generate_behavior_tree(prompt=prompt, fix_mistake="", which_prompt=tech_number, log_prompt=False)
                            behaviors = model.call_behaviors()
                            save_behavior_tree_with_metadata(behavior_tree, behaviors, agent_class, env_name, task_name, model_name, technique, style_name, bt_result_path)

                            # Update progress bar
                            pbar.update(1)
                            pbar.set_postfix({"Saved": os.path.basename(bt_result_path)})

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run behavior tree generation experiments.")
    parser.add_argument(
        '--backend', 
        type=str, 
        default="ollama", 
        help="Backend to use for the experiments. Options are ollama and llamacpp. Default is ollama.")
    parser.add_argument(
        '--n_trials',
        type=int,
        default=3, 
        help="Number of trials to run for each configuration. Default is 3")
    parser.add_argument(
        '--seed',
        type=int,
        default=42, 
        help="Seed for the random number generator. Default is 42")
    
    parser.add_argument(
        '--results_path', 
        type=str, 
        default=None, 
        help="Path to save the results. Default is a timestamped folder in 'results'.")
    
    parser.add_argument(
        '--model_bit', 
        type=int, 
        choices=[0, 1, 2], 
        default=0, 
        help="Select whether to use the vanilla models(0), or finetuned with dataset a(1) or dataset b(2). Default is 0.")

    args = parser.parse_args()

    if args.results_path is None:
        # Ensure base results directory exists
        base_results_dir = os.path.join("experiments", "results")
        os.makedirs(base_results_dir, exist_ok=True)
        
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        result_dir = os.path.join(base_results_dir, f"experiment_{current_time}")
    else:
        result_dir = args.results_path

    # Create the specific experiment results directory
    os.makedirs(result_dir, exist_ok=True)
    print(f"Results will be saved to: {result_dir}")

    if args.model_bit == 1:
        models = models_a
    elif args.model_bit == 2:
        models = models_b
    else:
        models = models_v

    run_experiment(models, args.backend, args.n_trials, result_dir, args.seed)
