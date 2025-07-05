"""API-based dataset generation using placeholder trees (Method B)."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from openai import OpenAI
from pydantic import BaseModel, Field

from agent_control.simulation.envs.robot_env import RobotEnvironment

from ..grammar_gen.compare_trees import validate_tree_structure
from ..grammar_gen.tree_to_prompt import (
    generate_spoon_prompt_from_string,
    generate_technical_prompt_from_string,
)
from .sys_prompt import system_prompt_b


class TreeResponse(BaseModel):
    """Response model for tree generation API calls."""

    task: str = Field(description="The task that the tree is solving")
    tree: str = Field(description="The raw behavior tree XML that solves the task")


# class Metrics(BaseModel):
#     good_parts_picked_up: Union[int, None] = Field(description="The number of good parts picked up")
#     bad_parts_picked_up: Union[int, None] = Field(description="The number of scrap parts picked up")
#     parts_dropped_in_base: Union[List[int], None] = Field(description="The number of good parts and scrap parts to be dropped in the base area")
#     parts_dropped_in_storage: Union[List[int], None] = Field(description="The number of good parts and scrap parts to be dropped in the storage area")
#     parts_dropped_in_construction: Union[List[int], None] = Field(description="The number of good parts and scrap parts to be dropped in the construction area")
#     parts_dropped_in_source: Union[List[int], None] = Field(description="The number of good parts and scrap parts to be dropped in the source area")
#     parts_dropped_in_waste: Union[List[int], None] = Field(description="The number of good parts and scrap parts to be dropped in the waste area")

# class TreeResponse2(BaseModel):
#     model_config = ConfigDict(extra='forbid')
#     task: str = Field(description="The task that the tree is solving")
#     tree: str = Field(description="The raw behavior tree XML that solves the task")
#     metrics: Metrics = Field(description="The metrics that the tree should achieve according to the task you came up with")

client = OpenAI(
    project="proj_g4ndQRSmeGRrVbVKaKJC88Su", api_key=os.getenv("OPENAI_API_KEY")
)


def get_tree_content(file_path: str) -> str:
    """
    Read and return the content of a tree file.

    Args:
        file_path: Path to the tree file to read

    Returns:
        str: Content of the tree file
    """
    with open(file_path, "r") as f:
        return f.read().strip()


def process_tree_with_api(
    tree_content: str,
    filter_env: Optional[RobotEnvironment] = None,
    filter_metrics: Optional[Union[List[str], Dict[str, Any]]] = None,
    conditions: Optional[List[str]] = None,
    actuator_actions: Optional[List[str]] = None,
    state_actions: Optional[List[str]] = None,
    max_retries: int = 1,
    focus_parts: str = "any",
) -> Tuple[Optional[str], Optional[str]]:
    """
    Send the placeholder tree to the API and retrieve the populated task and tree.

    Args:
        tree_content: The tree content string with placeholders
        filter_env: Environment to filter the trees
        filter_metrics: List of metrics that must be > 0 for tree to be valid
        conditions: List of valid condition nodes
        actuator_actions: List of valid actuator actions
        state_actions: List of valid state actions
        max_retries: Number of retries if validation fails
        focus_parts: The type of parts to focus on in the generated task

    Returns:
        Tuple[Optional[str], Optional[str]]: Task description and populated tree, or None if failed
    """
    validation_feedback = ""  # Initialize outside the loop

    for attempt in range(max_retries + 1):  # +1 for initial attempt
        user_prompt = f"""Here is a behaviour tree filled with placeholders:

        {tree_content}

        Please think of a task that could be solved with that behavior tree structure, then replace the placeholders with actual actions and conditions that match the 
        types of nodes in the tree so that the resulting tree has a high chance of solving the task you describe. Please focus your task on '{focus_parts}' parts."""

        if attempt > 0:  # Add feedback from previous attempt
            user_prompt += f"\n\nPrevious attempt had the following issues:\n{validation_feedback}\nPlease fix these issues and try again."

        completion = client.beta.chat.completions.parse(
            model="gpt-4.1-mini-2025-04-14",
            messages=[
                {"role": "system", "content": system_prompt_b},
                {"role": "user", "content": user_prompt},
            ],
            response_format=TreeResponse,
        )

        if completion.choices[0].message.parsed:
            response_obj = completion.choices[0].message.parsed
            task = response_obj.task
            tree = response_obj.tree
            # metrics = response_obj.metrics

            # Validate the generated tree against the original structure
            is_valid, validation_feedback = validate_tree_structure(
                original_tree=tree_content,
                generated_tree=tree,
                conditions=conditions,
                actuator_actions=actuator_actions,
                state_actions=state_actions,
            )
            # is_valid, validation_feedback = True, ""   # TODO: Remove this, this is just for testing

            # If structure validation passes, check metrics if filter_env provided
            if is_valid and filter_env is not None:
                print("Tree passed structure validation, testing metrics...")
                metrics_passed = _test_tree_metrics_b(
                    tree, filter_env, filter_metrics or []
                )
                if not metrics_passed:
                    print("Tree failed metrics test, retrying...")
                    validation_feedback += "\nThe generated tree did not achieve meaningful metrics in simulation. Please try a different approach."
                    is_valid = False

            if is_valid:
                return task, tree
            else:
                print(f"Attempt {attempt + 1}/{max_retries + 1} failed validation:")
                print(validation_feedback)
                if attempt == max_retries:
                    print("Max retries reached, skipping this tree")
                    return None, None
        else:
            print(
                f"Attempt {attempt + 1}/{max_retries + 1} failed: Could not parse structured response"
            )
            print(f"Raw response: {completion.choices[0].message.content}")
            if attempt == max_retries:
                print("Max retries reached, skipping this tree")
                return None, None

    # If we reach here, all attempts failed
    return None, None


def _test_tree_metrics_b(
    tree_content: str,
    filter_env: RobotEnvironment,
    filter_metrics: Union[List[str], Dict[str, Any]],
) -> bool:
    """
    Test a tree in the simulation environment and check if it achieves any target metrics.

    Similar to the one in dataset_generator.py but for string tree content.

    Args:
        tree_content: XML content of the behavior tree
        filter_env: Environment to use for testing
        filter_metrics: Metrics to check against

    Returns:
        bool: True if tree meets metric targets, False otherwise

    Raises:
        ValueError: If filter_metrics is not specified
    """
    import tempfile

    print(f"Filter metrics to test for: {filter_metrics}")

    if filter_metrics is None:
        raise ValueError(
            "filter_metrics must be specified when using filtering. No default metrics are provided."
        )

    # Create temporary file for the tree
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False
    ) as temp_file:
        temp_file.write(tree_content)
        temp_bt_path = temp_file.name

    try:
        # Create a fresh environment instance for each test
        test_env = RobotEnvironment(
            config=filter_env.config,
            bt_path=temp_bt_path,
            n_agents=getattr(filter_env, "n_agents", 10),
            n_parts=getattr(filter_env, "n_parts", 10),
            task=getattr(filter_env, "task", "default"),
            headless=getattr(filter_env, "headless", True),
        )

        # Setup and run the fresh environment
        test_env.setup()
        metrics = test_env.run()

        print(f"Tree metrics: {metrics}")

        # Check if target metrics are achieved using the same logic as in dataset_generator
        return _check_metrics_against_targets_b(metrics, filter_metrics)

    finally:
        # Clean up temporary file
        if os.path.exists(temp_bt_path):
            os.unlink(temp_bt_path)


def _check_metrics_against_targets_b(
    actual_metrics: Dict[str, Any], target_metrics: Union[List[str], Dict[str, Any]]
) -> bool:
    """
    Check if actual metrics meet the target criteria.

    Same logic as in dataset_generator.py but duplicated to avoid circular imports.

    Args:
        actual_metrics: Dictionary of actual metrics from simulation
        target_metrics: Either a list of metric names or dict of metric targets

    Returns:
        bool: True if metrics meet targets, False otherwise

    Raises:
        ValueError: If target_metrics format is invalid
    """
    if target_metrics is None:
        return True

    if isinstance(target_metrics, list):
        # Legacy mode: check if any metric in list is > 0
        for metric in target_metrics:
            if actual_metrics.get(metric, 0) > 0:
                return True
        return False

    elif isinstance(target_metrics, dict):
        # New mode: ALL metrics must meet their targets (AND logic)
        for metric_name, target_value in target_metrics.items():
            actual_value = actual_metrics.get(metric_name, 0)

            if isinstance(target_value, list) and isinstance(actual_value, list):
                # Both are lists - ALL elements must meet their corresponding targets
                if len(actual_value) != len(target_value):
                    return False
                for actual_elem, target_elem in zip(actual_value, target_value):
                    if target_elem == 0:
                        # For target 0, require exact equality
                        if actual_elem != 0:
                            return False
                    else:
                        # For target > 0, require >= target
                        if actual_elem < target_elem:
                            return False
            elif isinstance(target_value, (int, float)) and isinstance(
                actual_value, (int, float)
            ):
                # Both are numbers - check exact match for 0, >= for others
                if target_value == 0:
                    if actual_value != 0:
                        return False
                else:
                    if actual_value < target_value:
                        return False
            elif isinstance(target_value, (int, float)) and isinstance(
                actual_value, list
            ):
                # Target is number, actual is list
                if target_value == 0:
                    # For target 0, all elements must be exactly 0
                    if not all(elem == 0 for elem in actual_value):
                        return False
                else:
                    # For target > 0, at least one element must be >= target
                    if not any(elem >= target_value for elem in actual_value):
                        return False
            else:
                # Type mismatch - this metric fails
                return False

        # If we reach here, ALL metrics passed their criteria
        return True

    else:
        raise ValueError(
            f"target_metrics must be list or dict, got {type(target_metrics)}"
        )


def process_trees_in_folder(
    folder_path: str,
    output_json_path: str,
    max_trees: Optional[int] = None,
    filter_env: Optional[RobotEnvironment] = None,
    filter_metrics: Optional[Union[List[str], Dict[str, Any]]] = None,
    node_translations: Optional[Dict[str, str]] = None,
    node_connectors: Optional[Dict[str, str]] = None,
    spoon_node_translations: Optional[Dict[str, str]] = None,
    conditions: Optional[List[str]] = None,
    actuator_actions: Optional[List[str]] = None,
    state_actions: Optional[List[str]] = None,
) -> None:
    """
    Process trees in a folder and save results to a JSON file.

    Args:
        folder_path: Path to the folder containing XML tree files
        output_json_path: Path where the JSON dataset will be saved
        max_trees: Maximum number of trees to process
        filter_env: Environment to filter the trees
        filter_metrics: List of metrics that must be > 0 for tree to be valid
        node_translations: Dictionary mapping node names to technical translations
        node_connectors: Dictionary with connectors for nodes
        spoon_node_translations: Dictionary mapping node names to spoon translations
        conditions: List of valid condition nodes
        actuator_actions: List of valid actuator actions
        state_actions: List of valid state actions
    """
    dataset = []
    processed_count = 0

    # Determine focus_parts from filter_metrics
    focus_parts = "any"  # default
    if filter_metrics:
        # Check for presence of keys in the dictionary or elements in the list
        has_good = "good_parts_picked_up" in filter_metrics
        has_bad = "bad_parts_picked_up" in filter_metrics

        if has_good and has_bad:
            focus_parts = "both good and bad"
        elif has_good:
            focus_parts = "good"
        elif has_bad:
            focus_parts = "scrap"

    print(f"Focusing on parts: {focus_parts}")

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".xml") and (
            max_trees is None or processed_count < max_trees
        ):
            file_path = os.path.join(folder_path, file_name)
            try:
                print(f"Processing file: {file_name}")
                tree_content = get_tree_content(file_path)

                # Pass the filter parameters to process_tree_with_api
                task, tree = process_tree_with_api(
                    tree_content,
                    filter_env=filter_env,
                    filter_metrics=filter_metrics,
                    conditions=conditions,
                    actuator_actions=actuator_actions,
                    state_actions=state_actions,
                    focus_parts=focus_parts,
                )

                if task and tree:  # Only add to dataset if generation was successful
                    print(f"Task: {task}\n\nTree:\n{tree}\n")

                    try:
                        technical_task = generate_technical_prompt_from_string(
                            tree, node_translations or {}, node_connectors or {}
                        )
                    except Exception as e:
                        technical_task = f"Error generating technical task: {e}"

                    try:
                        spoon_task = generate_spoon_prompt_from_string(
                            tree, spoon_node_translations or {}, node_connectors or {}
                        )
                    except Exception as e:
                        spoon_task = f"Error generating spoon task: {e}"

                    dataset.append(
                        {
                            "layman_task": task,
                            "technical_task": technical_task,
                            "spoon_task": spoon_task,
                            "tree": tree,
                        }
                    )
                    processed_count += 1
                    max_trees_display = (
                        max_trees if max_trees is not None else "unlimited"
                    )
                    print(
                        f"------------Produced: {processed_count} of {max_trees_display} trees------------"
                    )
                else:
                    print(
                        f"Skipping {file_name} due to validation or metrics failures\n"
                    )
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    with open(output_json_path, "w") as json_file:
        json.dump(dataset, json_file, indent=4)
        print(f"Dataset saved to {output_json_path}")
