"""Unified RLHF-inspired dataset generation interface using Tkinter GUI."""

import os
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk
from typing import Any, Dict, List, Optional, Type

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from vi import Agent

from agent_control import RobotAgent, RobotEnvironment
from data_grammar.rlhf_generation.utils.prompt_builder import PromptBuilder
from data_grammar.rlhf_generation.utils.run_robot_sim import run_robot_sim
from data_grammar.rlhf_generation.utils.save_data_point import save_datapoint
from tree_parser import AgentDocstringParser, BehaviorTreeGrammarValidator
from tree_parser.primitives_validator import validate_primitives

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


# ------------------------------ Define Graph Schemas ------------------------------
class GraphInput(BaseModel):
    """Input model for the graph workflow."""

    task_definition: str = Field(
        description="The natural language definition of the task to be executed"
    )
    task_metrics_goal: str = Field(
        description="The user defined metrics the tree should achieve in the simulation"
    )
    dataset_size: int = Field(description="The number of samples generated")
    dataset_path: str = Field(description="The path to the dataset")
    dataset_size_goal: int = Field(description="The goal number of samples to generate")


class GraphOutput(BaseModel):
    """Output model for the graph workflow."""

    message: str = Field(description="The final message of the graph")


class GraphState(BaseModel):
    """Represents the state of our graph."""

    task_definition: str = Field(
        description="The natural language definition of the task to be executed"
    )
    task_metrics_goal: str = Field(
        description="The user defined metrics the tree should achieve in the simulation"
    )
    behaviour_tree: Optional[str] = Field(
        default=None, description="The bahavior tree returned by the LLM"
    )
    passed_validator: bool = Field(
        default=False,
        description="Whether the tree passed both the grammar and primitive validator checks",
    )
    validator_feedback: Optional[str] = Field(
        default=None,
        description="If the tree doesnt pass, the feedback of both the grammar and primitive validator checks",
    )
    task_metrics_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The metrics returned by executing the tree on the simulator",
    )
    human_feedback: Optional[str] = Field(
        default=None,
        description="The feedback of the human on the task execution that will be given to the LLM in case of retries",
    )
    dataset_size: int = Field(description="The number of samples generated")
    dataset_path: str = Field(description="The path to the dataset")
    dataset_size_goal: int = Field(description="The goal number of samples to generate")


# ------------------------------ Unified UI Class ------------------------------
class UnifiedRLHFUI:
    """Unified RLHF dataset generation interface with persistent GUI."""

    def __init__(
        self,
        dataset_path: str,
        dataset_size_goal: int,
        agent_class: Optional[Type[Agent]] = None,
        grammar_rules: Optional[Dict[str, Any]] = None,
        environment_class: Optional[Type[Any]] = None,
        environment_kwargs: Optional[Dict[str, Any]] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the unified RLHF UI with the given parameters."""
        self.dataset_path = dataset_path
        self.dataset_size_goal = dataset_size_goal
        self.agent_class = (
            agent_class or RobotAgent
        )  # Default to RobotAgent if none provided

        # Set grammar rules (use default if not provided)
        self.grammar_rules = grammar_rules or self._get_default_grammar_rules()

        # Set environment class (use RobotEnvironment if not provided)
        self.environment_class = environment_class or RobotEnvironment

        # Store kwargs for environment and config
        # Ensure headless mode is enabled by default to prevent pygame window conflicts
        default_env_kwargs = {"headless": True}
        if environment_kwargs:
            default_env_kwargs.update(environment_kwargs)
        self.environment_kwargs = default_env_kwargs

        self.config_kwargs = config_kwargs or {}

        self.current_state: Optional[GraphState] = None
        self.workflow_running = False
        self._feedback_mode: bool = False

        # Create main window
        self.root = tk.Tk()
        agent_name = self.agent_class.__name__ if self.agent_class else "Unknown Agent"
        self.root.title(f"RLHF Dataset Generation - {agent_name}")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)

        # Initialize workflow components
        self.setup_workflow()

        # Create UI
        self.create_ui()

        # Initialize first state
        self.reset_workflow()

    def _get_default_grammar_rules(self) -> Dict[str, Any]:
        """Get default grammar rules for behavior tree validation."""
        return {
            "B": [["b", ["SEL"]], ["b", ["SEQ"]]],
            "SEL": [["sel", ["SEQn", "As"]], ["sel", ["SEQn"]]],
            "SEQn": [["SEQ", "SEQn"], ["SEQ"]],
            "SEQ": [["seq", ["Pn", "A"]], ["seq", ["As", "Pn", "A"]]],
            "b": ["BehaviorTree", ["children_nodes"]],
            "sel": ["Selector", ["children_nodes"]],
            "seq": ["Sequence", ["children_nodes"]],
            "A": [["aa", "sa"], ["aa"], ["sa"]],
            "As": [["aa"], ["sa"]],
            "aa": ["ActuatorAction"],
            "sa": ["StateAction"],
            "Pn": [["p", "Pn"], ["p"], []],
            "p": ["Condition"],
        }

    def setup_workflow(self) -> None:
        """Initialize the workflow graph and LLM components."""
        self.prompt_builder = PromptBuilder(self.agent_class)
        self.system_prompt = self.prompt_builder.build_system_prompt()

        self.tree_generator_prompt = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), ("placeholder", "{user_prompt}")]
        )

        self.tree_generator_llm = ChatOpenAI(model="gpt-4o", temperature=0)

        # Initialize agent doc parser for extracting node information
        if self.agent_class:
            self.agent_doc_parser: Optional[AgentDocstringParser] = (
                AgentDocstringParser(self.agent_class)
            )
            self.agent_config: Optional[Dict[str, Any]] = (
                self.agent_doc_parser.extract_docstring_config()
            )
        else:
            self.agent_doc_parser = None
            self.agent_config = None

    def create_ui(self) -> None:
        """Create the main UI with tabs."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self.create_main_tab()
        self.create_nodes_tab()
        self.create_dataset_tab()

    def create_main_tab(self) -> None:
        """Create the main workflow tab."""
        self.main_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.main_frame, text="Main Workflow")

        # Create main layout
        main_container = ttk.Frame(self.main_frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left side - Controls and inputs
        left_frame = ttk.Frame(main_container)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        left_frame.config(width=400)

        # Right side - Information display
        right_frame = ttk.Frame(main_container)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.create_left_panel(left_frame)
        self.create_right_panel(right_frame)

    def create_left_panel(self, parent: tk.Widget) -> tk.Widget:
        """Create the left control panel."""
        # Dataset info
        info_frame = ttk.LabelFrame(parent, text="Dataset Information", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.dataset_size_label = ttk.Label(
            info_frame, text="Current Size: 0", font=("Arial", 10)
        )
        self.dataset_size_label.pack(anchor=tk.W)

        self.dataset_goal_label = ttk.Label(
            info_frame,
            text=f"Target Size: {self.dataset_size_goal}",
            font=("Arial", 10),
        )
        self.dataset_goal_label.pack(anchor=tk.W)

        self.progress_label = ttk.Label(
            info_frame, text="Progress: 0.0%", font=("Arial", 10, "bold")
        )
        self.progress_label.pack(anchor=tk.W)

        # Task input
        task_frame = ttk.LabelFrame(parent, text="Task Definition", padding=10)
        task_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        ttk.Label(
            task_frame, text="Task Description:", font=("Arial", 10, "bold")
        ).pack(anchor=tk.W)
        self.task_entry = tk.Text(task_frame, height=6, wrap=tk.WORD)
        self.task_entry.pack(fill=tk.BOTH, expand=True, pady=(5, 10))

        ttk.Label(task_frame, text="Metrics Goal:", font=("Arial", 10, "bold")).pack(
            anchor=tk.W
        )
        self.metrics_entry = tk.Text(task_frame, height=4, wrap=tk.WORD)
        self.metrics_entry.pack(fill=tk.BOTH, expand=True, pady=(5, 10))

        # Feedback input (initially hidden)
        self.feedback_frame = ttk.LabelFrame(parent, text="Human Feedback", padding=10)
        ttk.Label(
            self.feedback_frame,
            text="Feedback for improvement:",
            font=("Arial", 10, "bold"),
        ).pack(anchor=tk.W)
        self.feedback_entry = tk.Text(self.feedback_frame, height=4, wrap=tk.WORD)
        self.feedback_entry.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        # Buttons frame
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=10)

        self.prompt_button = ttk.Button(
            button_frame, text="Generate Tree", command=self.start_generation
        )
        self.prompt_button.pack(side=tk.LEFT, padx=(0, 5))

        self.run_sim_button = ttk.Button(
            button_frame,
            text="Run Simulation",
            command=self.run_simulation,
            state=tk.DISABLED,
        )
        self.run_sim_button.pack(side=tk.LEFT, padx=5)

        self.feedback_button = ttk.Button(
            button_frame,
            text="Give Feedback",
            command=self.give_feedback,
            state=tk.DISABLED,
        )
        self.feedback_button.pack(side=tk.LEFT, padx=5)

        self.save_button = ttk.Button(
            button_frame,
            text="Save Datapoint",
            command=self.save_datapoint,
            state=tk.DISABLED,
        )
        self.save_button.pack(side=tk.LEFT, padx=5)

        return parent

    def create_right_panel(self, parent: tk.Widget) -> tk.Widget:
        """Create the right information display panel."""
        # Status display
        status_frame = ttk.LabelFrame(parent, text="Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))

        self.status_label = ttk.Label(
            status_frame,
            text="Ready to start",
            font=("Arial", 12, "bold"),
            foreground="blue",
        )
        self.status_label.pack()

        # Generated tree display
        tree_frame = ttk.LabelFrame(parent, text="Generated Behavior Tree", padding=10)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.tree_display = scrolledtext.ScrolledText(
            tree_frame, height=12, wrap=tk.WORD, state=tk.DISABLED
        )
        self.tree_display.pack(fill=tk.BOTH, expand=True)

        # Metrics/Feedback display
        metrics_frame = ttk.LabelFrame(parent, text="Validation & Metrics", padding=10)
        metrics_frame.pack(fill=tk.BOTH, expand=True)

        self.metrics_display = scrolledtext.ScrolledText(
            metrics_frame, height=8, wrap=tk.WORD, state=tk.DISABLED
        )
        self.metrics_display.pack(fill=tk.BOTH, expand=True)

        return parent

    def create_nodes_tab(self) -> None:
        """Create the nodes information tab."""
        self.nodes_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.nodes_frame, text="Available Nodes")

        # Main container
        main_container = ttk.Frame(self.nodes_frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(
            title_frame, text="Available Agent Nodes", font=("Arial", 16, "bold")
        ).pack(side=tk.LEFT)

        # Create notebook for different node types
        self.nodes_notebook = ttk.Notebook(main_container)
        self.nodes_notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs for each node type
        self.create_conditions_tab()
        self.create_actuator_actions_tab()
        self.create_state_actions_tab()

    def create_conditions_tab(self) -> None:
        """Create tab for condition nodes."""
        conditions_frame = ttk.Frame(self.nodes_notebook)
        self.nodes_notebook.add(conditions_frame, text="Conditions")

        # Scrollable text area
        text_frame = ttk.Frame(conditions_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.conditions_display = scrolledtext.ScrolledText(
            text_frame, wrap=tk.WORD, state=tk.DISABLED
        )
        self.conditions_display.pack(fill=tk.BOTH, expand=True)

        # Populate with condition nodes
        self.populate_conditions()

    def create_actuator_actions_tab(self) -> None:
        """Create tab for actuator action nodes."""
        actuator_frame = ttk.Frame(self.nodes_notebook)
        self.nodes_notebook.add(actuator_frame, text="Actuator Actions")

        # Scrollable text area
        text_frame = ttk.Frame(actuator_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.actuator_display = scrolledtext.ScrolledText(
            text_frame, wrap=tk.WORD, state=tk.DISABLED
        )
        self.actuator_display.pack(fill=tk.BOTH, expand=True)

        # Populate with actuator action nodes
        self.populate_actuator_actions()

    def create_state_actions_tab(self) -> None:
        """Create tab for state action nodes."""
        state_frame = ttk.Frame(self.nodes_notebook)
        self.nodes_notebook.add(state_frame, text="State Actions")

        # Scrollable text area
        text_frame = ttk.Frame(state_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.state_display = scrolledtext.ScrolledText(
            text_frame, wrap=tk.WORD, state=tk.DISABLED
        )
        self.state_display.pack(fill=tk.BOTH, expand=True)

        # Populate with state action nodes
        self.populate_state_actions()

    def populate_conditions(self) -> None:
        """Populate conditions tab with available condition nodes."""
        self.conditions_display.config(state=tk.NORMAL)
        self.conditions_display.delete(1.0, tk.END)

        # Configure text tags for formatting
        self.conditions_display.tag_configure("title", font=("Arial", 14, "bold"))
        self.conditions_display.tag_configure("node_name", font=("Arial", 13, "bold"))
        self.conditions_display.tag_configure("description", font=("Arial", 10))

        if self.agent_config is None:
            self.conditions_display.insert(
                tk.END, "No agent configuration available.", "title"
            )
            self.conditions_display.config(state=tk.DISABLED)
            return

        conditions = self.agent_config.get("conditions", [])

        if conditions:
            # Insert title
            self.conditions_display.insert(
                tk.END, "Available Condition Nodes:\n", "title"
            )
            self.conditions_display.insert(tk.END, "=" * 50 + "\n\n")

            for i, condition in enumerate(conditions, 1):
                # Insert node name with bold formatting
                self.conditions_display.insert(
                    tk.END, f"{i}. {condition}\n", "node_name"
                )
                self.conditions_display.insert(tk.END, "-" * 30 + "\n")

                # Try to get more info about the condition from the agent
                try:
                    # Get the method from the agent class
                    if hasattr(self.agent_class, condition):
                        method = getattr(self.agent_class, condition)
                        if hasattr(method, "__doc__") and method.__doc__:
                            self.conditions_display.insert(
                                tk.END,
                                f"Description: {method.__doc__.strip()}\n",
                                "description",
                            )
                        else:
                            self.conditions_display.insert(
                                tk.END,
                                "Description: No documentation available\n",
                                "description",
                            )
                    else:
                        self.conditions_display.insert(
                            tk.END,
                            "Description: Method not found in agent class\n",
                            "description",
                        )
                except Exception as e:
                    self.conditions_display.insert(
                        tk.END,
                        f"Description: Error retrieving info - {str(e)}\n",
                        "description",
                    )

                self.conditions_display.insert(tk.END, "\n")
        else:
            self.conditions_display.insert(tk.END, "No condition nodes found.", "title")

        self.conditions_display.config(state=tk.DISABLED)

    def populate_actuator_actions(self) -> None:
        """Populate actuator actions tab with available actuator action nodes."""
        self.actuator_display.config(state=tk.NORMAL)
        self.actuator_display.delete(1.0, tk.END)

        # Configure text tags for formatting
        self.actuator_display.tag_configure("title", font=("Arial", 14, "bold"))
        self.actuator_display.tag_configure("node_name", font=("Arial", 13, "bold"))
        self.actuator_display.tag_configure("description", font=("Arial", 10))

        if self.agent_config is None:
            self.actuator_display.insert(
                tk.END, "No agent configuration available.", "title"
            )
            self.actuator_display.config(state=tk.DISABLED)
            return

        actuator_actions = self.agent_config.get("actuator_actions", [])

        if actuator_actions:
            # Insert title
            self.actuator_display.insert(
                tk.END, "Available Actuator Action Nodes:\n", "title"
            )
            self.actuator_display.insert(tk.END, "=" * 50 + "\n\n")

            for i, action in enumerate(actuator_actions, 1):
                # Insert node name with bold formatting
                self.actuator_display.insert(tk.END, f"{i}. {action}\n", "node_name")
                self.actuator_display.insert(tk.END, "-" * 30 + "\n")

                # Try to get more info about the action from the agent
                try:
                    # Get the method from the agent class
                    if hasattr(self.agent_class, action):
                        method = getattr(self.agent_class, action)
                        if hasattr(method, "__doc__") and method.__doc__:
                            self.actuator_display.insert(
                                tk.END,
                                f"Description: {method.__doc__.strip()}\n",
                                "description",
                            )
                        else:
                            self.actuator_display.insert(
                                tk.END,
                                "Description: No documentation available\n",
                                "description",
                            )
                    else:
                        self.actuator_display.insert(
                            tk.END,
                            "Description: Method not found in agent class\n",
                            "description",
                        )
                except Exception as e:
                    self.actuator_display.insert(
                        tk.END,
                        f"Description: Error retrieving info - {str(e)}\n",
                        "description",
                    )

                self.actuator_display.insert(tk.END, "\n")
        else:
            self.actuator_display.insert(
                tk.END, "No actuator action nodes found.", "title"
            )

        self.actuator_display.config(state=tk.DISABLED)

    def populate_state_actions(self) -> None:
        """Populate state actions tab with available state action nodes."""
        self.state_display.config(state=tk.NORMAL)
        self.state_display.delete(1.0, tk.END)

        # Configure text tags for formatting
        self.state_display.tag_configure("title", font=("Arial", 14, "bold"))
        self.state_display.tag_configure("node_name", font=("Arial", 13, "bold"))
        self.state_display.tag_configure("description", font=("Arial", 10))

        if self.agent_config is None:
            self.state_display.insert(
                tk.END, "No agent configuration available.", "title"
            )
            self.state_display.config(state=tk.DISABLED)
            return

        state_actions = self.agent_config.get("state_actions", [])

        if state_actions:
            # Insert title
            self.state_display.insert(
                tk.END, "Available State Action Nodes:\n", "title"
            )
            self.state_display.insert(tk.END, "=" * 50 + "\n\n")

            for i, action in enumerate(state_actions, 1):
                # Insert node name with bold formatting
                self.state_display.insert(tk.END, f"{i}. {action}\n", "node_name")
                self.state_display.insert(tk.END, "-" * 30 + "\n")

                # Try to get more info about the action from the agent
                try:
                    # Get the method from the agent class
                    if hasattr(self.agent_class, action):
                        method = getattr(self.agent_class, action)
                        if hasattr(method, "__doc__") and method.__doc__:
                            self.state_display.insert(
                                tk.END,
                                f"Description: {method.__doc__.strip()}\n",
                                "description",
                            )
                        else:
                            self.state_display.insert(
                                tk.END,
                                "Description: No documentation available\n",
                                "description",
                            )
                    else:
                        self.state_display.insert(
                            tk.END,
                            "Description: Method not found in agent class\n",
                            "description",
                        )
                except Exception as e:
                    self.state_display.insert(
                        tk.END,
                        f"Description: Error retrieving info - {str(e)}\n",
                        "description",
                    )

                self.state_display.insert(tk.END, "\n")
        else:
            self.state_display.insert(tk.END, "No state action nodes found.", "title")

        self.state_display.config(state=tk.DISABLED)

    def create_dataset_tab(self) -> None:
        """Create the dataset exploration tab."""
        self.dataset_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.dataset_frame, text="Dataset Explorer")

        # Main container
        main_container = ttk.Frame(self.dataset_frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title and refresh button
        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(
            title_frame, text="Dataset Explorer", font=("Arial", 16, "bold")
        ).pack(side=tk.LEFT)
        self.refresh_button = ttk.Button(
            title_frame, text="Refresh", command=self.refresh_dataset_list
        )
        self.refresh_button.pack(side=tk.RIGHT)

        # Left side - Datapoint list
        left_frame = ttk.LabelFrame(main_container, text="Datapoints", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        left_frame.config(width=300)

        # Listbox with scrollbar for datapoints
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.datapoint_listbox = tk.Listbox(list_frame, width=40)
        self.datapoint_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.datapoint_listbox.bind("<<ListboxSelect>>", self.on_datapoint_select)

        list_scrollbar = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self.datapoint_listbox.yview
        )
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.datapoint_listbox.config(yscrollcommand=list_scrollbar.set)

        # Right side - Datapoint details
        right_frame = ttk.LabelFrame(
            main_container, text="Datapoint Details", padding=10
        )
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Layman task display
        ttk.Label(right_frame, text="Layman Task:", font=("Arial", 12, "bold")).pack(
            anchor=tk.W, pady=(0, 5)
        )
        self.layman_display = scrolledtext.ScrolledText(
            right_frame, height=6, wrap=tk.WORD, state=tk.DISABLED
        )
        self.layman_display.pack(fill=tk.X, pady=(0, 15))

        # Behavior tree display
        ttk.Label(right_frame, text="Behavior Tree:", font=("Arial", 12, "bold")).pack(
            anchor=tk.W, pady=(0, 5)
        )
        self.tree_details_display = scrolledtext.ScrolledText(
            right_frame, height=15, wrap=tk.WORD, state=tk.DISABLED
        )
        self.tree_details_display.pack(fill=tk.BOTH, expand=True)

        # Load initial data
        self.dataset_data: List[Dict[str, Any]] = []
        self.refresh_dataset_list()

    def reset_workflow(self) -> None:
        """Reset the workflow to initial state."""
        # Calculate current dataset size
        current_size = self.get_current_dataset_size()

        self.current_state = GraphState(
            task_definition="",
            task_metrics_goal="",
            dataset_size=current_size,
            dataset_path=self.dataset_path,
            dataset_size_goal=self.dataset_size_goal,
        )

        # Reset workflow state
        self.workflow_running = False

        # Clear input fields for fresh start
        self.task_entry.delete(1.0, tk.END)
        self.metrics_entry.delete(1.0, tk.END)
        if hasattr(self, "feedback_entry"):
            self.feedback_entry.delete(1.0, tk.END)

        self.update_dataset_info()
        self.update_status("Ready to start", "blue")
        self.clear_displays()
        self.reset_buttons()

    def get_current_dataset_size(self) -> int:
        """Get current dataset size from file."""
        import json

        if os.path.exists(self.dataset_path):
            try:
                with open(self.dataset_path, "r") as f:
                    dataset = json.load(f)
                    return len(dataset)
            except (json.JSONDecodeError, FileNotFoundError):
                return 0
        return 0

    def update_dataset_info(self) -> None:
        """Update dataset information display."""
        current_size = self.get_current_dataset_size()
        progress = (
            (current_size / self.dataset_size_goal) * 100
            if self.dataset_size_goal > 0
            else 0
        )

        self.dataset_size_label.config(text=f"Current Size: {current_size}")
        self.progress_label.config(text=f"Progress: {progress:.1f}%")

    def update_status(self, message: str, color: str = "black") -> None:
        """Update status message."""
        self.status_label.config(text=message, foreground=color)
        self.root.update()

    def clear_displays(self) -> None:
        """Clear all display areas."""
        self.tree_display.config(state=tk.NORMAL)
        self.tree_display.delete(1.0, tk.END)
        self.tree_display.config(state=tk.DISABLED)

        self.metrics_display.config(state=tk.NORMAL)
        self.metrics_display.delete(1.0, tk.END)
        self.metrics_display.config(state=tk.DISABLED)

    def reset_buttons(self) -> None:
        """Reset button states."""
        self.prompt_button.config(state=tk.NORMAL)
        self.run_sim_button.config(state=tk.DISABLED)
        self.feedback_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        self.feedback_frame.pack_forget()

    def start_generation(self) -> None:
        """Start the tree generation process."""
        if self.workflow_running:
            return

        # Get input values
        task_definition = self.task_entry.get(1.0, tk.END).strip()
        task_metrics_goal = self.metrics_entry.get(1.0, tk.END).strip()

        if not task_definition:
            self.update_status("Please fill in the task definition", "red")
            return

        # Update state
        if self.current_state:
            self.current_state.task_definition = task_definition
            self.current_state.task_metrics_goal = task_metrics_goal

        # Get feedback if available
        if (
            hasattr(self, "_feedback_mode")
            and self._feedback_mode
            and self.current_state
        ):
            human_feedback = self.feedback_entry.get(1.0, tk.END).strip()
            self.current_state.human_feedback = (
                human_feedback if human_feedback else None
            )
            self._feedback_mode = False
            self.feedback_frame.pack_forget()

        # Start generation in separate thread
        self.workflow_running = True
        self.prompt_button.config(state=tk.DISABLED)
        self.update_status("Generating behavior tree...", "orange")

        threading.Thread(target=self.generate_tree, daemon=True).start()

    def generate_tree(self) -> None:
        """Generate behavior tree using LLM."""
        try:
            if not self.current_state:
                self.root.after(
                    0, self.on_generation_error, "No current state available"
                )
                return

            # Prepare prompt
            prompt = f"Please generate a behaviour tree for the following task: {self.current_state.task_definition} \nThe task metrics goal is: {self.current_state.task_metrics_goal}"

            if self.current_state.human_feedback:
                prompt += f"\n\nHuman Feedback on Previous Attempt: {self.current_state.human_feedback}\nPlease incorporate this feedback to improve the behavior tree."

            # Generate tree
            class TreeGeneratorOutput(BaseModel):
                behaviour_tree: str = Field(
                    description="The raw behaviour tree in XML format without any quotes or markdown formatting"
                )

            tree_gen_chain = (
                self.tree_generator_prompt
                | self.tree_generator_llm.with_structured_output(TreeGeneratorOutput)
            )
            result = tree_gen_chain.invoke({"user_prompt": [("user", prompt)]})

            # Update UI in main thread
            if hasattr(result, "behaviour_tree"):
                self.root.after(0, self.on_tree_generated, result.behaviour_tree)
            elif isinstance(result, dict) and "behaviour_tree" in result:
                self.root.after(0, self.on_tree_generated, result["behaviour_tree"])
            else:
                self.root.after(
                    0,
                    self.on_generation_error,
                    f"Unexpected result format: {type(result)}",
                )

        except Exception as e:
            self.root.after(0, self.on_generation_error, str(e))

    def on_tree_generated(self, behaviour_tree: str) -> None:
        """Handle successful tree generation."""
        if self.current_state:
            self.current_state.behaviour_tree = behaviour_tree
            self.current_state.human_feedback = None  # Reset feedback

        # Display tree
        self.tree_display.config(state=tk.NORMAL)
        self.tree_display.delete(1.0, tk.END)
        self.tree_display.insert(1.0, behaviour_tree)
        self.tree_display.config(state=tk.DISABLED)

        self.update_status("Tree generated! Validating...", "orange")

        # Validate tree
        threading.Thread(target=self.validate_tree, daemon=True).start()

    def on_generation_error(self, error_msg: str) -> None:
        """Handle tree generation errors."""
        self.update_status(f"Generation failed: {error_msg}", "red")

        # Reset workflow state
        self.workflow_running = False
        self.prompt_button.config(state=tk.NORMAL)

    def validate_tree(self) -> None:
        """Validate the generated behavior tree."""
        if not self.current_state or not self.current_state.behaviour_tree:
            self.update_status("No tree to validate", "red")
            return

        self.update_status("Validating tree...", "orange")

        threading.Thread(
            target=lambda: self._validate_tree_thread(), daemon=True
        ).start()

    def _validate_tree_thread(self) -> None:
        """Run tree validation in background thread."""
        try:
            if not self.current_state or not self.current_state.behaviour_tree:
                self.root.after(
                    0, self.on_validation_error, "No tree available for validation"
                )
                return

            grammar_validator = BehaviorTreeGrammarValidator(self.grammar_rules)
            passed_grammar, grammar_feedback = grammar_validator.validate_tree(
                self.current_state.behaviour_tree
            )
            passed_primitive, primitive_feedback = validate_primitives(
                self.current_state.behaviour_tree, self.agent_class
            )

            passed = passed_grammar and passed_primitive
            feedback = ""
            if not passed_grammar:
                feedback += f"Grammar validation failed: {grammar_feedback}\n"
            if not passed_primitive:
                feedback += f"Primitive validation failed: {primitive_feedback}"

            self.root.after(0, self.on_validation_complete, passed, feedback)
        except Exception as e:
            self.root.after(0, self.on_validation_error, str(e))

    def on_validation_complete(self, passed: bool, feedback: str) -> None:
        """Handle validation completion."""
        if self.current_state:
            self.current_state.passed_validator = passed
            self.current_state.validator_feedback = feedback

        if passed:
            self.update_status("Tree validation passed!", "green")
            if self.current_state:
                self.metrics_display.config(state=tk.NORMAL)
                self.metrics_display.delete(1.0, tk.END)
                self.metrics_display.insert(
                    1.0, f"Desired Metrics:\n{self.current_state.task_metrics_goal}"
                )
                self.metrics_display.config(state=tk.DISABLED)
        else:
            self.update_status("Tree validation failed!", "red")
            self.metrics_display.config(state=tk.NORMAL)
            self.metrics_display.delete(1.0, tk.END)
            self.metrics_display.insert(
                1.0,
                f"Validation Feedback:\n{feedback}\n\nSimulation cannot run with invalid tree.",
            )
            self.metrics_display.config(state=tk.DISABLED)

        # Enable buttons based on validation result
        self.run_sim_button.config(state=tk.NORMAL if passed else tk.DISABLED)
        self.feedback_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)

        # Reset workflow state
        self.workflow_running = False
        self.prompt_button.config(state=tk.NORMAL)

    def on_validation_error(self, error_msg: str) -> None:
        """Handle validation errors."""
        self.update_status(f"Validation failed: {error_msg}", "red")

        # Reset workflow state
        self.workflow_running = False
        self.prompt_button.config(state=tk.NORMAL)

    def run_simulation(self) -> None:
        """Run simulation of the behavior tree."""
        if not self.current_state or not self.current_state.passed_validator:
            self.update_status("Cannot run simulation: tree validation failed", "red")
            return

        self.update_status("Running simulation...", "orange")
        self.run_sim_button.config(state=tk.DISABLED)

        threading.Thread(target=self.run_sim_thread, daemon=True).start()

    def run_sim_thread(self) -> None:
        """Run simulation in background thread."""
        try:
            if not self.current_state or not self.current_state.behaviour_tree:
                raise ValueError("No valid tree available for simulation")

            metrics_result = run_robot_sim(
                self.current_state.behaviour_tree,
                environment_class=self.environment_class,
                environment_kwargs=self.environment_kwargs,
                config_kwargs=self.config_kwargs,
            )

            self.root.after(0, self.on_simulation_complete, metrics_result)
        except Exception as e:
            self.root.after(0, self.on_simulation_error, str(e))

    def on_simulation_complete(self, metrics_result: Dict[str, Any]) -> None:
        """Handle simulation completion."""
        if self.current_state:
            self.current_state.task_metrics_result = metrics_result

        # Update metrics display
        self.metrics_display.config(state=tk.NORMAL)
        self.metrics_display.delete(1.0, tk.END)

        if self.current_state:
            content = f"Desired Metrics:\n{self.current_state.task_metrics_goal}\n\n"
            content += f"Achieved Metrics:\n{metrics_result}"
            self.metrics_display.insert(1.0, content)

        self.metrics_display.config(state=tk.DISABLED)
        self.update_status("Simulation completed!", "green")

        # Re-enable the simulation button
        self.run_sim_button.config(state=tk.NORMAL)

        # Force UI update to ensure responsiveness
        self.root.update_idletasks()
        self.root.update()

    def on_simulation_error(self, error_msg: str) -> None:
        """Handle simulation errors."""
        self.update_status(f"Simulation failed: {error_msg}", "red")

        # Re-enable the simulation button on error
        self.run_sim_button.config(state=tk.NORMAL)

        # Force UI update to ensure responsiveness
        self.root.update_idletasks()
        self.root.update()

    def give_feedback(self) -> None:
        """Enable feedback mode for human input."""
        if not self.current_state or not self.current_state.behaviour_tree:
            self.update_status("No tree available for feedback", "red")
            return

        self._feedback_mode = True

        # Show and setup the feedback frame
        self.feedback_frame.pack(fill=tk.X, pady=(0, 10))
        self.feedback_entry.delete(1.0, tk.END)
        self.feedback_entry.focus()

        self.update_status(
            "Provide feedback and click 'Generate Tree' to retry", "blue"
        )

    def save_datapoint(self) -> None:
        """Save current tree and metrics as a datapoint."""
        if not self.current_state or not self.current_state.behaviour_tree:
            self.update_status("No tree to save", "red")
            return

        try:
            save_datapoint(
                dataset_path=self.current_state.dataset_path,
                task_description=self.current_state.task_definition,
                tree_str=self.current_state.behaviour_tree,
                agent_class=self.agent_class,
            )

            self.update_dataset_info()
            self.update_status("Datapoint saved successfully!", "green")

            # Check if goal is reached
            if self.ask_continue_generation():
                self.reset_workflow()
            else:
                self.update_status("Dataset generation completed!", "blue")
        except Exception as e:
            self.update_status(f"Failed to save datapoint: {e}", "red")

    def ask_continue_generation(self) -> bool:
        """Ask user if they want to continue generating data."""
        if not self.current_state:
            return False

        current_size = self.get_current_dataset_size()

        if current_size >= self.dataset_size_goal:
            import tkinter.messagebox as msgbox

            result = msgbox.askyesno(
                "Goal Reached",
                f"Dataset size goal of {self.dataset_size_goal} reached! "
                f"Current size: {current_size}. Continue generating?",
            )
        else:
            import tkinter.messagebox as msgbox

            result = msgbox.askyesno(
                "Continue Generation",
                f"Datapoint saved! ({current_size}/{self.dataset_size_goal})\nWould you like to continue generating more datapoints?",
            )

        if result:
            self.reset_workflow()
        else:
            self.update_status("Generation complete", "blue")

        # Refresh dataset explorer
        self.refresh_dataset_list()

        return result

    def refresh_dataset_list(self) -> None:
        """Refresh the dataset list in the dataset tab."""
        if not hasattr(self, "datapoint_listbox"):
            return

        self.datapoint_listbox.delete(0, tk.END)

        import json

        try:
            if os.path.exists(self.dataset_path):
                with open(self.dataset_path, "r") as f:
                    dataset = json.load(f)
                    for i, datapoint in enumerate(dataset):
                        # Show preview of layman task
                        preview = datapoint.get("layman_task", "No task description")[
                            :50
                        ]
                        if len(preview) == 50:
                            preview += "..."
                        self.datapoint_listbox.insert(tk.END, f"{i+1}. {preview}")
        except (json.JSONDecodeError, FileNotFoundError):
            pass

        self.clear_dataset_details()

    def on_datapoint_select(self, event: Any) -> None:
        """Handle datapoint selection in the list."""
        try:
            selection = self.datapoint_listbox.curselection()  # type: ignore[no-untyped-call]
            if not selection:
                return

            index = selection[0]
            import json

            with open(self.dataset_path, "r") as f:
                dataset = json.load(f)
                if index < len(dataset):
                    self.display_datapoint_details(dataset[index])
        except Exception:
            pass

    def display_datapoint_details(self, datapoint: Dict[str, Any]) -> None:
        """Display details of selected datapoint."""
        # Display layman task
        self.layman_display.config(state=tk.NORMAL)
        self.layman_display.delete(1.0, tk.END)
        self.layman_display.insert(
            1.0, datapoint.get("layman_task", "No task description")
        )
        self.layman_display.config(state=tk.DISABLED)

        # Display formatted tree
        self.tree_details_display.config(state=tk.NORMAL)
        self.tree_details_display.delete(1.0, tk.END)

        tree_content = datapoint.get("tree", "No tree available")
        if tree_content:
            # Format XML for better readability
            try:
                import xml.dom.minidom

                dom = xml.dom.minidom.parseString(tree_content)
                formatted_tree = dom.toprettyxml(indent="  ")
                # Remove empty lines
                formatted_tree = "\n".join(
                    [line for line in formatted_tree.split("\n") if line.strip()]
                )
                self.tree_details_display.insert(1.0, formatted_tree)
            except Exception:
                self.tree_details_display.insert(1.0, tree_content)

        self.tree_details_display.config(state=tk.DISABLED)

    def clear_dataset_details(self) -> None:
        """Clear the dataset details display."""
        if hasattr(self, "layman_display"):
            self.layman_display.config(state=tk.NORMAL)
            self.layman_display.delete(1.0, tk.END)
            self.layman_display.config(state=tk.DISABLED)

        if hasattr(self, "tree_details_display"):
            self.tree_details_display.config(state=tk.NORMAL)
            self.tree_details_display.delete(1.0, tk.END)
            self.tree_details_display.config(state=tk.DISABLED)

    def run(self) -> None:
        """Start the application."""
        self.root.mainloop()


def main(
    agent_class: Type[Agent] = RobotAgent,
    grammar_rules: Optional[Dict[str, Any]] = None,
    environment_class: Optional[Type[Any]] = None,
    environment_kwargs: Optional[Dict[str, Any]] = None,
    config_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Run the unified RLHF UI application."""
    ui = UnifiedRLHFUI(
        dataset_path="./data_grammar/rlhf_generation/output/dataset_path.json",
        dataset_size_goal=10,
        agent_class=agent_class,
        grammar_rules=grammar_rules,
        environment_class=environment_class,
        environment_kwargs=environment_kwargs,
        config_kwargs=config_kwargs,
    )
    ui.run()


if __name__ == "__main__":

    grammar_rules = {
        "B": [["b", ["SEL"]], ["b", ["SEQ"]]],
        "SEL": [["sel", ["SEQn", "As"]], ["sel", ["SEQn"]]],
        "SEQn": [["SEQ", "SEQn"], ["SEQ"]],
        "SEQ": [["seq", ["Pn", "A"]], ["seq", ["As", "Pn", "A"]]],
        "b": ["BehaviorTree", ["children_nodes"]],
        "sel": ["Selector", ["children_nodes"]],
        "seq": ["Sequence", ["children_nodes"]],
        "A": [["aa", "sa"], ["aa"], ["sa"]],
        "As": [["aa"], ["sa"]],
        "aa": ["ActuatorAction"],
        "sa": ["StateAction"],
        "Pn": [["p", "Pn"], ["p"], []],
        "p": ["Condition"],
    }

    # Example usage with custom parameters:
    main(
        agent_class=RobotAgent,
        environment_class=RobotEnvironment,
        grammar_rules=grammar_rules,
        # Config parameters for simulation
        config_kwargs={
            "radius": 40,
            "duration": 500,
            "movement_speed": 1.0,
            "window_size": 800,
            "visualise_chunks": True,
        },
        # Environment parameters
        environment_kwargs={
            "n_agents": 10,
            "n_parts": 10,
            "task": "custom_task",
            "headless": False,
        },
    )
