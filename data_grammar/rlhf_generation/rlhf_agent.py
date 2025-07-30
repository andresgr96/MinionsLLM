import os
from typing import Optional, Literal, Tuple
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from dataclasses import dataclass
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import tkinter as tk
from tkinter import ttk

from data_grammar.dataset_generation.sys_prompt import system_prompt_b2
from tree_parser import BehaviorTreeGrammarValidator
from tree_parser.primitives_validator import validate_primitives
from agent_control import RobotAgent
from utils.run_robot_sim import run_robot_sim
from utils.save_data_point import save_datapoint


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# ------------------------------ Define Graph Schemas ------------------------------
class GraphInput(BaseModel):
    """Schema for the input of the graph"""
    dataset_path: str = Field(description="The path to the dataset")
    dataset_size_goal: int = Field(description="The number of samples to generate")

class GraphOutput(BaseModel):
    """Schema for the output of the graph"""
    dataset_path: str = Field(description="The path to the dataset")
    dataset_size: int = Field(description="The number of samples generated")

class GraphState(BaseModel):
    """Represents the state of our graph."""
    task_definition: str = Field(description="The natural language definition of the task to be executed")
    task_metrics_goal: str = Field(description="The user defined metrics the tree should achieve in the simulation")
    behaviour_tree: Optional[str] = Field(default=None, description="The bahavior tree returned by the LLM")
    passed_validator: bool = Field(default=False, description="Whether the tree passed both the grammar and primitive validator checks")
    validator_feedback: Optional[str] = Field(default=None, description="If the tree doesnt pass, the feedback of both the grammar and primitive validator checks")
    task_metrics_result: Optional[dict] = Field(default=None, description="The metrics returned by executing the tree on the simulator")
    human_feedback: Optional[str] = Field(default=None, description="The feedback of the human on the task execution that will be given to the LLM in case of retries")
    dataset_size: int = Field(default=0, description="The number of samples generated")
    dataset_path: str = Field(description="The path to the dataset")
    dataset_size_goal: int = Field(description="The number of samples to generate")


# ------------------------------ Define Graph Nodes ------------------------------

# Human Task Input Node ----------------------------------------------------------
def human_task_input_node(input: GraphInput, state: Optional[GraphState] = None) -> Command[Literal["tree_generator_node"]]:
    # Create a simple UI to get user input
    task_definition = ""
    task_metrics_goal = ""

    current_dataset_size = state.dataset_size if state else 0
    
    def on_submit():
        nonlocal task_definition, task_metrics_goal
        task_definition = task_def_entry.get("1.0", tk.END).strip()
        task_metrics_goal = task_metrics_entry.get("1.0", tk.END).strip()
        root.destroy()
    
    # Create the main window
    root = tk.Tk()
    root.title("Task Input")
    root.geometry("600x400")
    root.resizable(True, True)
    
    # Main frame
    main_frame = ttk.Frame(root, padding="20")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    
    # Task Definition section
    ttk.Label(main_frame, text="Task Definition:", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
    task_def_entry = tk.Text(main_frame, height=8, width=60, wrap=tk.WORD)
    task_def_entry.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
    
    # Task Metrics Goal section
    ttk.Label(main_frame, text="Task Metrics Goal:", font=("Arial", 12, "bold")).grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
    task_metrics_entry = tk.Text(main_frame, height=6, width=60, wrap=tk.WORD)
    task_metrics_entry.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
    
    # Submit button
    submit_button = ttk.Button(main_frame, text="Prompt", command=on_submit)
    submit_button.grid(row=4, column=0, pady=10)
    
    # Configure grid weights for resizing
    main_frame.grid_rowconfigure(1, weight=2)
    main_frame.grid_rowconfigure(3, weight=1)
    main_frame.grid_columnconfigure(0, weight=1)
    
    # Center the window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Run the UI
    root.mainloop()
    
    return Command(
        update={
            "task_definition": task_definition,
            "task_metrics_goal": task_metrics_goal,
            "dataset_size": current_dataset_size,
            "dataset_path": input.dataset_path,
            "dataset_size_goal": input.dataset_size_goal,
        },
        goto="tree_generator_node",
    )


# Tree Generator Node ----------------------------------------------------------
tree_generator_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", system_prompt_b2
        ),
        ("placeholder", "{user_prompt}"),
    ]
)

tree_generator_llm = ChatOpenAI(model="gpt-4o", temperature=0)

class TreeGeneratorOutput(BaseModel):
    """Schema for the structured output of the tree generator tool"""
    behaviour_tree: str = Field(description=" The raw behaviour tree in XML format without any quotes or markdown formatting")

def tree_generator_node(state: GraphState) -> Command[Literal["tree_validator_node"]]:  # type: ignore , ignore the END warning
    task_definition = state.task_definition
    task_metrics_goal = state.task_metrics_goal

    prompt = "Please generate a behaviour tree for the following task: " + task_definition + " \nThe task metrics goal is: " + task_metrics_goal

    tree_gen_chain_oai = tree_generator_prompt | tree_generator_llm.with_structured_output(TreeGeneratorOutput)
    result = tree_gen_chain_oai.invoke({"user_prompt": [("user", prompt)]})
    print(f"Behaviour Tree: {result.behaviour_tree}")

    return Command(
        update={
            "behaviour_tree": result.behaviour_tree,
        },
        goto="tree_validator_node",
    )

# Tree Validator Node ----------------------------------------------------------

def tree_validator_node(state: GraphState) -> Command[Literal["environment_simulator_node"]]:  # type: ignore , ignore the END warning

    behaviour_tree = state.behaviour_tree

    # Define custom rules for the grammar
    grammar_rules = {                                                                 
        "B":   [["b", ["SEL"]], ["b", ["SEQ"]]],                                                          
        "SEL": [["sel", ["SEQn", "As"]], ["sel", ["SEQn"]]],                                               
        "SEQn":[["SEQ", "SEQn"], ["SEQ"]], 
        "SEQ": [["seq", ["Pn", "A"]], ["seq", ["As", "Pn", "A"]]],
        "b":   ["BehaviorTree", ["children_nodes"]],     
        "sel": ["Selector", ["children_nodes"]],
        "seq": ["Sequence", ["children_nodes"]],                                            
        "A":   [["aa", "sa"], ["aa"], ["sa"]],                                                                  
        "As":  [["aa"], ["sa"]],                                                                  
        "aa":  ["ActuatorAction"],                                                    
        "sa":  ["StateAction"],
        "Pn":  [["p", "Pn"], ["p"], []], 
        "p":   ["Condition"]
    }

    passed_validators = False
    feedback = None
    grammar_validator = BehaviorTreeGrammarValidator(grammar_rules)

    passed_grammar_validator, grammar_feedback = grammar_validator.validate_tree(behaviour_tree)
    passed_primitive_validator, primitive_feedback = validate_primitives(behaviour_tree, RobotAgent)

    passed_validators = passed_grammar_validator and passed_primitive_validator

    if not passed_validators:
        feedback = f"The tree failed either the grammar or the primitive validator checks. \nGrammar feedback: {grammar_feedback} \nPrimitive feedback: {primitive_feedback}"

    print(f"Passed validators: {passed_validators}")
    print(f"Feedback: {feedback}")

    return Command(
        update={
            "passed_validator": passed_validators,
            "validator_feedback": feedback,
        },
        goto="environment_simulator_node",
    )

# Environment Simulator Node ----------------------------------------------------------

def environment_simulator_node(state: GraphState) -> Command[Literal["datapoint_saver_node"]]:  # type: ignore , ignore the END warning
    behaviour_tree = state.behaviour_tree
    task_metrics_goal = state.task_metrics_goal

    # Create a UI to display simulation progress and results
    task_metrics_result = {}
    
    def run_simulation():
        nonlocal task_metrics_result
        # Update the metrics label to show "Running simulation..."
        metrics_achieved_text.config(state="normal")
        metrics_achieved_text.delete("1.0", tk.END)
        metrics_achieved_text.insert("1.0", "Running simulation...")
        metrics_achieved_text.config(state="disabled")
        root.update()
        
        # Run the actual simulation
        task_metrics_result = run_robot_sim(behaviour_tree)
        
        # Force pygame to quit and close all windows
        try:
            import pygame
            pygame.quit()
        except:
            pass
        
        # Format the metrics dictionary for display
        metrics_display = "\n".join([f"{key}: {value}" for key, value in task_metrics_result.items()])
        
        # Update the UI with results
        metrics_achieved_text.config(state="normal")
        metrics_achieved_text.delete("1.0", tk.END)
        metrics_achieved_text.insert("1.0", metrics_display)
        metrics_achieved_text.config(state="disabled")
        
        # Enable the close button
        close_button.config(state="normal")
    
    def on_close():
        root.destroy()
    
    # Create the main window
    root = tk.Tk()
    root.title("Tree Simulation")
    root.geometry("700x500")
    root.resizable(True, True)
    
    # Main frame
    main_frame = ttk.Frame(root, padding="20")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    
    # Success message
    success_label = ttk.Label(main_frame, text="Tree Validation Passed! Simulating Tree:", 
                             font=("Arial", 14, "bold"), foreground="green")
    success_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
    
    # Behavior Tree section
    ttk.Label(main_frame, text="Behavior Tree:", font=("Arial", 12, "bold")).grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
    tree_text = tk.Text(main_frame, height=8, width=80, wrap=tk.WORD, state="disabled")
    tree_text.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
    tree_text.config(state="normal")
    tree_text.insert("1.0", behaviour_tree)
    tree_text.config(state="disabled")
    
    # Desired Metrics section
    ttk.Label(main_frame, text="Desired Metrics:", font=("Arial", 12, "bold")).grid(row=3, column=0, sticky=tk.W, pady=(0, 5))
    metrics_desired_text = tk.Text(main_frame, height=4, width=80, wrap=tk.WORD, state="disabled")
    metrics_desired_text.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
    metrics_desired_text.config(state="normal")
    metrics_desired_text.insert("1.0", task_metrics_goal)
    metrics_desired_text.config(state="disabled")
    
    # Metrics Achieved section
    ttk.Label(main_frame, text="Metrics Achieved:", font=("Arial", 12, "bold")).grid(row=5, column=0, sticky=tk.W, pady=(0, 5))
    metrics_achieved_text = tk.Text(main_frame, height=4, width=80, wrap=tk.WORD, state="disabled")
    metrics_achieved_text.grid(row=6, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
    metrics_achieved_text.config(state="normal")
    metrics_achieved_text.insert("1.0", "Waiting for simulation to start...")
    metrics_achieved_text.config(state="disabled")
    
    # Button frame
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=7, column=0, pady=10)
    
    # Run simulation button
    run_button = ttk.Button(button_frame, text="Run Simulation", command=run_simulation)
    run_button.pack(side=tk.LEFT, padx=(0, 10))
    
    # Close button (initially disabled)
    close_button = ttk.Button(button_frame, text="Save Datapoint", command=on_close, state="disabled")
    close_button.pack(side=tk.LEFT)
    
    # Configure grid weights for resizing
    main_frame.grid_rowconfigure(2, weight=2)  # Behavior tree text
    main_frame.grid_rowconfigure(4, weight=1)  # Desired metrics text
    main_frame.grid_rowconfigure(6, weight=1)  # Achieved metrics text
    main_frame.grid_columnconfigure(0, weight=1)
    
    # Center the window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Run the UI
    root.mainloop()
    
    return Command(
        update={
            "task_metrics_result": task_metrics_result,
        },
        goto="datapoint_saver_node",
    )

# Datapoint Saver Node ----------------------------------------------------------

def datapoint_saver_node(state: GraphState) -> Command[Literal[END, "human_input_node"]]:  # type: ignore , ignore the END warning

    behaviour_tree = state.behaviour_tree
    layman_prompt = state.task_definition
    curr_dataset_size = state.dataset_size
    dataset_path = state.dataset_path
    dataset_size_goal = state.dataset_size_goal

    save_datapoint(dataset_path=dataset_path, task_description=layman_prompt, tree_str=behaviour_tree, agent_class=RobotAgent)
    dataset_size = curr_dataset_size + 1

    # Create a UI to ask user if they want to continue generation
    continue_generation = False
    
    def on_continue():
        nonlocal continue_generation
        continue_generation = True
        root.destroy()
    
    def on_exit():
        nonlocal continue_generation
        continue_generation = False
        root.destroy()
    
    # Create the main window
    root = tk.Tk()
    root.title("Dataset Generation Progress")
    root.geometry("500x300")
    root.resizable(True, True)
    
    # Main frame
    main_frame = ttk.Frame(root, padding="20")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    
    # Success message
    success_label = ttk.Label(main_frame, text="Datapoint Saved Successfully!", 
                             font=("Arial", 16, "bold"), foreground="green")
    success_label.grid(row=0, column=0, pady=(0, 20))
    
    # Progress information
    progress_label = ttk.Label(main_frame, text="Dataset Generation Progress:", 
                              font=("Arial", 14, "bold"))
    progress_label.grid(row=1, column=0, pady=(0, 10))
    
    # Current size display
    size_label = ttk.Label(main_frame, text=f"Current Dataset Size: {dataset_size}", 
                          font=("Arial", 12))
    size_label.grid(row=2, column=0, pady=(0, 5))
    
    # Goal size display
    goal_label = ttk.Label(main_frame, text=f"Target Dataset Size: {dataset_size_goal}", 
                          font=("Arial", 12))
    goal_label.grid(row=3, column=0, pady=(0, 5))
    
    # Progress percentage
    progress_percentage = (dataset_size / dataset_size_goal) * 100 if dataset_size_goal > 0 else 0
    percentage_label = ttk.Label(main_frame, text=f"Progress: {progress_percentage:.1f}%", 
                                font=("Arial", 12, "bold"))
    percentage_label.grid(row=4, column=0, pady=(0, 20))
    
    # Button frame
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=5, column=0, pady=20)
    
    # Continue generation button
    continue_button = ttk.Button(button_frame, text="Continue Generation", command=on_continue)
    continue_button.pack(side=tk.LEFT, padx=(0, 20))
    
    # Exit generation button
    exit_button = ttk.Button(button_frame, text="Exit Generation", command=on_exit)
    exit_button.pack(side=tk.LEFT)
    
    # Configure grid weights for resizing
    main_frame.grid_rowconfigure(0, weight=1)
    main_frame.grid_rowconfigure(5, weight=1)
    main_frame.grid_columnconfigure(0, weight=1)
    
    # Center the window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Run the UI
    root.mainloop()
    
    next_node = "human_input_node" if continue_generation else END

    return Command(
        update={
            "dataset_size": dataset_size,
        },
        goto=next_node,
    )

# ------------------------------ Main Workflow ------------------------------
def main(dataset_path: str, dataset_size_goal: int) -> Tuple[str, int]:
    workflow = StateGraph(GraphState, input_schema=GraphInput, output_schema=GraphOutput)
    workflow.add_node("human_input_node", human_task_input_node)
    workflow.add_node("tree_generator_node", tree_generator_node)
    workflow.add_node("tree_validator_node", tree_validator_node)
    workflow.add_node("environment_simulator_node", environment_simulator_node)
    workflow.add_node("datapoint_saver_node", datapoint_saver_node)

    workflow.add_edge(START, "human_input_node")
    graph = workflow.compile()

    results = graph.invoke({"dataset_path": dataset_path, "dataset_size_goal": dataset_size_goal})
    path = results["dataset_path"]
    size = results["dataset_size"]
    print(f"Dataset generated at {path} with size {size}")

    return path, size

if __name__ == "__main__":
    main(dataset_path="./data_grammar/rlhf_generation/output/dataset_path.json", dataset_size_goal=10)
