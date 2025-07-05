"""Dataset generator for behavior tree training data.

Here we will define the class DatasetGenerator that will wrap the entire dataset generation process given a grammar and other parameters.
Example parameters should in addition to the grammar should be like the size of the dataset, whether to use procedure a or b.
"""

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from vi import Agent

from control_layer.simulation.agents import RobotAgent
from control_layer.simulation.envs import RobotEnvironment, SimEnvironment
from data_grammar.dataset_generation.enrich_dataset import enrich_dataset as enrich_fn
from data_grammar.grammar_gen.grammar_config import grammar_parameters as default_params
from data_grammar.grammar_gen.grammar_config import grammar_rules as default_rules
from data_grammar.grammar_gen.int_to_list import generate_nested_list
from data_grammar.grammar_gen.list_to_trees import list_to_xml, pretty_print_xml
from data_grammar.grammar_gen.node_translations import (
    node_connectors as default_connectors,
)
from parser_layer.agent_doc_parser import AgentDocstringParser

# Add type: ignore for dotenv import since it's an optional dependency
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:
    pass  # dotenv is optional


class DatasetGenerator:
    """
    A class that handles the generation of datasets for behavior trees.

    This class provides methods to:
    1. Generate datasets using procedure A (processing existing trees with API)
    2. Generate datasets using procedure B (populating placeholder trees)
    3. Upload datasets to Hugging Face
    """

    def __init__(
        self,
        agent_class: Optional[Type[Agent]] = None,
        grammar_rules: Optional[Dict[str, List[str]]] = None,
        grammar_parameters: Optional[
            Union[
                Dict[str, Dict[str, Any]], List[Tuple[int, Dict[str, Dict[str, Any]]]]
            ]
        ] = None,
        node_connectors: Optional[Dict[str, str]] = None,
        output_dir: str = "./data_grammar/output",
        seed: Optional[int] = None,
    ):
        """
        Initialize the DatasetGenerator.

        Args:
            agent_class: Agent class to extract grammar from (optional)
            grammar_rules: Dictionary defining the grammar rules (optional)
            grammar_parameters: Dictionary with parameters to control tree generation, OR
                               List of tuples [(count, params_dict), ...] where count is the number
                               of trees with that structure to include in the final dataset (optional)
            node_connectors: Dictionary with the connectors for the nodes (optional)
            output_dir: Directory to save generated trees and datasets
            seed: Random seed for reproducibility

        Raises:
            ValueError: If grammar_parameters is not a dict or list of tuples
        """
        # If no agent_class provided, use RobotAgent as default
        if agent_class is None:
            agent_class = RobotAgent

        self.agent_class = agent_class

        # Extract docstring information from agent class
        docstring_parser = AgentDocstringParser(agent_class)
        self.extracted_config = docstring_parser.extract_docstring_config()

        # Store extracted translations for use in dataset generation
        self.node_translations = self.extracted_config["node_translations"]
        self.spoon_node_translations = self.extracted_config["spoon_node_translations"]

        # Use provided rules or defaults
        self.grammar_rules = grammar_rules or default_rules

        # Handle grammar_parameters - can be dict or list of (count, params)
        if grammar_parameters is None:
            self.grammar_parameters: Dict[str, Dict[str, Any]] = dict(default_params)  # type: ignore
            self.mixed_structures = False
        elif isinstance(grammar_parameters, dict):
            self.grammar_parameters = grammar_parameters
            self.mixed_structures = False
        elif isinstance(grammar_parameters, list):
            self.grammar_parameters_list = grammar_parameters
            self.mixed_structures = True
            # For type checking, set a default value
            self.grammar_parameters = dict(default_params)  # type: ignore
        else:
            raise ValueError(
                "grammar_parameters must be a dict or list of (count, params) tuples"
            )

        self.node_connectors = node_connectors or default_connectors
        self.output_dir = Path(output_dir)

        # Create output directories if they don't exist
        self.trees_dir = self.output_dir / "trees"
        self.trees_dir.mkdir(parents=True, exist_ok=True)
        self.populated_dir = self.trees_dir / "populated"
        self.populated_dir.mkdir(exist_ok=True)
        self.unpopulated_dir = self.trees_dir / "unpopulated"
        self.unpopulated_dir.mkdir(exist_ok=True)

        self.datasets_dir = self.output_dir / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)

        self.over_generate_trees = False  # This is used to generate more trees than needed to account for filtering in dataset b

    def _generate_trees(
        self,
        n_trees: int = 10,
        size: int = 10,
        placeholders: bool = True,
        filter_env: Optional[SimEnvironment] = None,
        output_dir: Optional[Path] = None,
        filter_metrics: Optional[Union[List[str], Dict[str, Any]]] = None,
        grammar_params: Optional[Dict[str, Dict[str, Any]]] = None,
        start_index: int = 0,
    ) -> List[str]:
        """
        Generate behavior trees based on grammar rules.

        Args:
            n_trees: Number of trees to generate (or target if filtering)
            size: Size of the integer list used for generation
            placeholders: Whether to use placeholders for node values
            filter_env: Environment to filter the trees (only works for dataset A). Providing one automatically filters the trees.
            output_dir: Directory to save the generated trees
            filter_metrics: List of metrics that must be > 0 for tree to be valid
            grammar_params: Specific grammar parameters to use (if None, uses self.grammar_parameters)
            start_index: Starting index for file naming

        Returns:
            List of paths to the generated tree files
        """
        if output_dir is None:
            output_dir = self.unpopulated_dir if placeholders else self.populated_dir

        # Use provided grammar parameters or default
        params_to_use = (
            grammar_params if grammar_params is not None else self.grammar_parameters
        )

        tree_paths: List[str] = []

        # For dataset A with filtering
        if not placeholders and filter_env is not None:
            return self._generate_filtered_trees(
                n_trees,
                size,
                filter_env,
                output_dir,
                filter_metrics,
                params_to_use,
                start_index,
            )

        # Original logic for non-filtered generation
        print(
            f"Generating {n_trees} trees with size {size} and placeholders: {placeholders}"
        )
        for i in range(n_trees):
            if i % 1000 == 0 and i > 0:
                print(f"Processing tree {i} of {n_trees}")

            # Generate random integers
            integers = [random.randint(1, 9) for _ in range(size)]

            # Convert to nested list
            nested_list = generate_nested_list(integers, self.grammar_rules, params_to_use)  # type: ignore

            # Convert to XML using the extracted node lists
            xml_tree = list_to_xml(
                nested_list,
                placeholders,
                conditions=self.extracted_config["conditions"],
                actuator_actions=self.extracted_config["actuator_actions"],
                state_actions=self.extracted_config["state_actions"],
            )
            str_tree = pretty_print_xml(xml_tree)

            # Save tree to file.
            tree_path = output_dir / f"behavior_tree_{start_index + i}.xml"

            with open(tree_path, "w") as f:
                f.write(str_tree)

            tree_paths.append(str(tree_path))

        return tree_paths

    def _generate_filtered_trees(
        self,
        target_trees: int,
        size: int,
        filter_env: SimEnvironment,
        output_dir: Path,
        filter_metrics: Optional[Union[List[str], Dict[str, Any]]] = None,
        grammar_params: Optional[Dict[str, Dict[str, Any]]] = None,
        start_index: int = 0,
    ) -> List[str]:
        """
        Generate trees with filtering based on simulation metrics.

        Args:
            target_trees: Number of valid trees to generate
            size: Size of the integer list used for generation
            filter_env: Environment to use for filtering trees
            output_dir: Directory to save the generated trees
            filter_metrics: List of metrics that must be > 0 for tree to be valid
            grammar_params: Specific grammar parameters to use
            start_index: Starting index for file naming

        Returns:
            List[str]: List of paths to the generated tree files

        Raises:
            ValueError: If filter_metrics is not specified
        """
        if filter_metrics is None:
            raise ValueError(
                "filter_metrics must be specified when using filtering. No default metrics are provided."
            )

        # Use provided grammar parameters or default
        params_to_use = (
            grammar_params if grammar_params is not None else self.grammar_parameters
        )

        tree_paths: List[str] = []
        attempts = 0

        print(f"Generating filtered trees. Target: {target_trees}")

        while len(tree_paths) < target_trees:
            attempts += 1

            if attempts % 100 == 0:
                print(
                    f"Attempt {attempts}: Found {len(tree_paths)}/{target_trees} valid trees"
                )

            # Generate tree
            integers = [random.randint(1, 9) for _ in range(size)]
            nested_list = generate_nested_list(integers, self.grammar_rules, params_to_use)  # type: ignore
            xml_tree = list_to_xml(
                nested_list,
                placeholders=False,
                conditions=self.extracted_config["conditions"],
                actuator_actions=self.extracted_config["actuator_actions"],
                state_actions=self.extracted_config["state_actions"],
            )

            # Test tree in simulator
            if self._test_tree_metrics(xml_tree, filter_env, filter_metrics):
                # Save valid tree
                tree_path = (
                    output_dir / f"behavior_tree_{start_index + len(tree_paths)}.xml"
                )
                str_tree = pretty_print_xml(xml_tree)

                with open(tree_path, "w") as f:
                    f.write(str_tree)

                tree_paths.append(str(tree_path))

        print(
            f"Filtering complete: {len(tree_paths)} valid trees found in {attempts} attempts"
        )
        return tree_paths

    def _generate_mixed_structure_trees(
        self,
        tree_size: int = 10,
        placeholders: bool = True,
        filter_env: Optional[SimEnvironment] = None,
        filter_metrics: Optional[Union[List[str], Dict[str, Any]]] = None,
        output_dir: Optional[Path] = None,
    ) -> Union[List[str], List[Tuple[Path, int, Optional[Dict[str, Any]]]]]:
        """
        Generate trees with mixed structures based on grammar_parameters_list.

        Args:
            tree_size: Size parameter for tree generation
            placeholders: Whether to use placeholders for node values
            filter_env: Environment to filter the trees
            filter_metrics: List of metrics that must be > 0 for tree to be valid
            output_dir: Directory to save the generated trees

        Returns:
            If filter_env is None: List of paths to all generated tree files
            If filter_env is not None: List of tuples (structure_folder_path, target_count, structure_metrics)

        Raises:
            ValueError: If structure config does not have 2 or 3 elements
        """
        if output_dir is None:
            output_dir = self.unpopulated_dir if placeholders else self.populated_dir

        all_tree_paths: List[str] = []
        structure_info: List[Tuple[Path, int, Optional[Dict[str, Any]]]] = (
            []
        )  # For returning folder info when using separate folders
        use_separate_folders = False

        print(
            f"Generating mixed structure dataset with {len(self.grammar_parameters_list)} different structures"
        )

        for i, structure_config in enumerate(self.grammar_parameters_list):
            # Handle both 2-element and 3-element structure configs
            if len(structure_config) == 2:
                target_count, params = structure_config
                structure_metrics = None
            elif len(structure_config) == 3:
                use_separate_folders = True  # Always use separate folders for mixed structures with metrics
                target_count, params, structure_metrics = structure_config  # type: ignore
            else:
                raise ValueError(
                    f"Structure config must have 2 or 3 elements, got {len(structure_config)}"
                )

            print(f"Using separate folders per structure: {use_separate_folders}")

            if use_separate_folders:
                # Create structure-specific subfolder
                structure_dir = output_dir / f"structure_{i}"
                structure_dir.mkdir(exist_ok=True)
                current_output_dir = structure_dir
                start_index = 0  # Reset index for each structure folder

                # Generate more trees than needed to account for filtering
                multiplier = 50 if self.over_generate_trees else 1
                trees_to_generate = max(target_count * multiplier, target_count)
                print(
                    f"Structure {i+1}/{len(self.grammar_parameters_list)}: Generating {trees_to_generate} trees in {structure_dir}"
                )

                # Store structure info for later processing
                structure_info.append((structure_dir, target_count, structure_metrics))
            else:
                # Use shared folder with continuous indexing
                current_output_dir = output_dir
                start_index = len(all_tree_paths)
                trees_to_generate = target_count
                print(
                    f"Structure {i+1}/{len(self.grammar_parameters_list)}: Generating {trees_to_generate} trees starting from index {start_index}"
                )

            # Generate trees with this specific structure
            # Use structure-specific metrics if available and filtering is enabled
            metrics_to_use = (
                structure_metrics
                if (structure_metrics is not None and filter_env is not None)
                else filter_metrics
            )

            structure_trees = self._generate_trees(
                n_trees=trees_to_generate,
                size=tree_size,
                placeholders=placeholders,
                filter_env=filter_env,
                output_dir=current_output_dir,
                filter_metrics=metrics_to_use,
                grammar_params=params,
                start_index=start_index,
            )

            # Add all generated trees to the list (only used when not using separate folders)
            if not use_separate_folders:
                all_tree_paths.extend(structure_trees)

            print(
                f"Structure {i+1} base generation complete: {len(structure_trees)} trees created"
            )

        if use_separate_folders:
            print(
                f"Mixed structure generation complete: {len(structure_info)} structure folders created"
            )
            return structure_info
        else:
            print(
                f"Mixed structure generation complete: {len(all_tree_paths)} total trees created"
            )
            return all_tree_paths

    def _test_tree_metrics(
        self,
        xml_tree: Any,
        filter_env: SimEnvironment,
        filter_metrics: Union[List[str], Dict[str, Any]],
    ) -> bool:
        """
        Test a tree in the simulation environment and check if it achieves any target metrics.

        Args:
            xml_tree: XML tree to test in simulation
            filter_env: Environment to use for testing
            filter_metrics: Metrics to check against

        Returns:
            bool: True if tree meets metric targets, False otherwise
        """
        # Create temporary file for the tree
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False
        ) as temp_file:
            temp_file.write(pretty_print_xml(xml_tree))
            temp_bt_path = temp_file.name

        try:
            # Create a fresh environment instance for each test
            # This prevents metric carryover between tests
            # Note: Using RobotEnvironment instead of SimEnvironment for proper type compatibility
            test_env = RobotEnvironment(
                config=filter_env.config,
                bt_path=temp_bt_path,
                n_agents=getattr(filter_env, "n_agents", 1),
                n_parts=getattr(filter_env, "n_parts", 5),
                task=getattr(filter_env, "task", "default"),
                headless=getattr(filter_env, "headless", True),
            )

            # Setup and run the fresh environment
            test_env.setup()
            metrics = test_env.run()

            # Check if target metrics are achieved
            if self._check_metrics_against_targets(metrics, filter_metrics):
                print(
                    f"Tree {pretty_print_xml(xml_tree)}\n passed with metrics: {metrics}"
                )
                return True

            return False

        finally:
            # Clean up temporary file
            if os.path.exists(temp_bt_path):
                os.unlink(temp_bt_path)

    def _check_metrics_against_targets(
        self,
        actual_metrics: Dict[str, Any],
        target_metrics: Union[List[str], Dict[str, Any]],
    ) -> bool:
        """
        Check if actual metrics meet the target criteria.

        Args:
            actual_metrics: Dictionary of actual metrics from simulation
            target_metrics: Either a list of metric names (check if > 0) or dict of metric targets

        Returns:
            True if metrics meet targets, False otherwise

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

    def generate_dataset_a(
        self,
        dataset_name: str = "dataset_a",
        n_trees: int = 1000,
        tree_size: int = 10,
        max_trees_to_process: Optional[int] = None,
        filter_env: Optional[SimEnvironment] = None,
        filter_metrics: Optional[Union[List[str], Dict[str, Any]]] = None,
        enrich_dataset: bool = False,
    ) -> str:
        """
        Generate dataset using method A (populated trees).

        This method:
        1. Generates populated trees (with actual node values)
        2. Processes these trees to create a dataset
        3. Optionally filters trees based on simulation metrics
        4. Optionally enriches the dataset with handcoded examples

        Args:
            dataset_name: Name of the dataset to generate
            n_trees: Number of trees to generate (or target if filtering)
            tree_size: Size parameter for tree generation
            max_trees_to_process: Maximum number of trees to process for the dataset
            filter_env: Environment to filter the trees
            filter_metrics: List of metrics that must be > 0 for tree to be valid
            enrich_dataset: Whether to enrich the dataset with handcoded examples

        Returns:
            Path to the generated dataset
        """
        from data_grammar.dataset_generation.api_gen_dataset_a import (
            process_trees_in_folder,
        )

        self.over_generate_trees = False  # In case generate_dataset_b was called before, we don't want to over-generate trees by mistake

        to_filter = False
        if filter_env is not None:
            to_filter = True

        # Generate populated trees
        print(
            f"Generating populated trees for dataset {dataset_name} and with filtering: {to_filter}..."
        )

        if self.mixed_structures:
            # Calculate max_trees_to_process based on trees_per_params (first element of each structure config)
            max_trees_to_process = sum(
                config[0] for config in self.grammar_parameters_list
            )
            print(f"Max trees to process: {max_trees_to_process}")

            # Generate mixed structure trees
            structure_result = self._generate_mixed_structure_trees(
                tree_size=tree_size,
                placeholders=False,
                filter_env=filter_env,
                filter_metrics=filter_metrics,
                output_dir=self.populated_dir,
            )
        else:
            # Generate single structure trees
            if to_filter is False:
                self._generate_trees(
                    n_trees=n_trees,
                    size=tree_size,
                    placeholders=False,
                    output_dir=self.populated_dir,
                )
            else:
                self._generate_trees(
                    n_trees=n_trees,
                    size=tree_size,
                    placeholders=False,
                    output_dir=self.populated_dir,
                    filter_env=filter_env,
                    filter_metrics=filter_metrics,
                )
            structure_result = None

        # Process trees to create dataset
        output_file = self.datasets_dir / f"{dataset_name}.json"
        print(f"Processing trees to create dataset {dataset_name}...")

        if (
            self.mixed_structures
            and isinstance(structure_result, list)
            and len(structure_result) > 0
            and isinstance(structure_result[0], tuple)
        ):
            # Process each structure folder separately (same as dataset B)
            print("Processing structures separately...")
            all_dataset_entries = []

            for i, (structure_dir, target_count, structure_metrics) in enumerate(
                structure_result
            ):
                # Use structure-specific metrics if available, otherwise fall back to global metrics
                metrics_to_use = (
                    structure_metrics
                    if structure_metrics is not None
                    else filter_metrics
                )
                print(
                    f"Processing structure {i+1}/{len(structure_result)} from {structure_dir} (target: {target_count} trees)"
                )
                print(f"Using metrics: {metrics_to_use}")

                # Create temporary file for this structure's dataset
                temp_output = self.datasets_dir / f"temp_structure_{i}.json"

                process_trees_in_folder(  # type: ignore
                    folder_path=str(structure_dir),
                    output_json_path=str(temp_output),
                    max_trees=target_count,  # type: ignore
                    node_translations=self.node_translations,
                    node_connectors=self.node_connectors,
                    spoon_node_translations=self.spoon_node_translations
                )

                # Load the results and add to combined dataset
                if temp_output.exists():
                    with open(temp_output, "r") as f:
                        structure_entries = json.load(f)
                    all_dataset_entries.extend(structure_entries)
                    print(
                        f"Structure {i+1} contributed {len(structure_entries)} entries"
                    )

                    # Clean up temporary file
                    temp_output.unlink()
                else:
                    print(f"Warning: No output file created for structure {i+1}")

            # Save combined dataset
            with open(output_file, "w") as f:
                json.dump(all_dataset_entries, f, indent=4)
            print(
                f"Combined dataset saved with {len(all_dataset_entries)} total entries"
            )

        else:
            # Standard processing (single structure or shared folder)
            process_trees_in_folder(
                folder_path=str(self.populated_dir),
                output_json_path=str(output_file),
                max_trees=max_trees_to_process,
                node_translations=self.node_translations,
                node_connectors=self.node_connectors,
                spoon_node_translations=self.spoon_node_translations,
            )

        # Enrich dataset if requested (modify in place)
        if enrich_dataset:
            print(f"Enriching dataset {dataset_name}...")
            enrich_fn(
                input_file=str(output_file),
                output_file=str(output_file),  # Same file - overwrite
                handcoded_examples=None,  # Use default examples
                node_translations=self.node_translations,
                node_connectors=self.node_connectors,
                spoon_node_translations=self.spoon_node_translations,
            )

        return str(output_file)

    def generate_dataset_b(
        self,
        dataset_name: str = "dataset_b",
        n_trees: int = 1000,
        tree_size: int = 10,
        max_trees_to_process: Optional[int] = None,
        filter_env: Optional[SimEnvironment] = None,
        filter_metrics: Optional[Union[List[str], Dict[str, Any]]] = None,
        enrich_dataset: bool = False,
    ) -> str:
        """
        Generate dataset using method B (populating placeholder trees).

        This method:
        1. Generates unpopulated trees (with placeholder node values)
        2. Processes these trees to create a dataset by populating placeholders
        3. Optionally filters trees based on simulation metrics
        4. Optionally enriches the dataset with handcoded examples

        Args:
            dataset_name: Name of the dataset to generate
            n_trees: Number of trees to generate initially
            tree_size: Size parameter for tree generation
            max_trees_to_process: Maximum number of trees to process for the dataset
            filter_env: Environment to filter the trees
            filter_metrics: List of metrics that must be > 0 for tree to be valid
            enrich_dataset: Whether to enrich the dataset with handcoded examples

        Returns:
            Path to the generated dataset
        """
        from data_grammar.dataset_generation.api_gen_dataset_b import (
            process_trees_in_folder,
        )

        self.over_generate_trees = True  # Generate more trees than needed to account for filtering and llm mistakes in dataset b

        # Generate unpopulated trees
        print(f"Generating unpopulated trees for dataset {dataset_name}...")

        if self.mixed_structures:
            # Generate mixed structure trees
            structure_result = self._generate_mixed_structure_trees(
                tree_size=tree_size, placeholders=True, output_dir=self.unpopulated_dir
            )
        else:
            # Generate single structure trees
            self._generate_trees(
                n_trees=n_trees,
                size=tree_size,
                placeholders=True,
                output_dir=self.unpopulated_dir,
            )
            structure_result = None

        # Process trees to create dataset with optional filtering
        output_file = self.datasets_dir / f"{dataset_name}.json"
        print(f"Processing trees to create dataset {dataset_name}...")

        to_filter = filter_env is not None
        print(f"Filtering enabled: {to_filter}")

        if self.mixed_structures:
            # Process each structure folder separately
            print("Processing structures separately due to filtering...")
            all_dataset_entries = []

            for i, (structure_dir, target_count, structure_metrics) in enumerate(structure_result):  # type: ignore
                # Use structure-specific metrics if available, otherwise fall back to global metrics
                metrics_to_use = (
                    structure_metrics
                    if structure_metrics is not None
                    else filter_metrics
                )
                print(f"Processing structure {i+1}/{len(structure_result)} from {structure_dir} (target: {target_count} trees)")  # type: ignore
                print(f"Using metrics: {metrics_to_use}")

                # Create temporary file for this structure's dataset
                temp_output = self.datasets_dir / f"temp_structure_{i}.json"

                process_trees_in_folder(
                    folder_path=str(structure_dir),
                    output_json_path=str(temp_output),
                    max_trees=target_count,  # type: ignore
                    filter_env=filter_env,  # type: ignore
                    filter_metrics=metrics_to_use,  # type: ignore
                    node_translations=self.node_translations,
                    node_connectors=self.node_connectors,
                    spoon_node_translations=self.spoon_node_translations,
                    conditions=self.extracted_config["conditions"],
                    actuator_actions=self.extracted_config["actuator_actions"],
                    state_actions=self.extracted_config["state_actions"],
                )

                # Load the results and add to combined dataset
                if temp_output.exists():
                    with open(temp_output, "r") as f:
                        structure_entries = json.load(f)
                    all_dataset_entries.extend(structure_entries)
                    print(
                        f"Structure {i+1} contributed {len(structure_entries)} entries"
                    )

                    # Clean up temporary file
                    temp_output.unlink()
                else:
                    print(f"Warning: No output file created for structure {i+1}")

            # Save combined dataset
            with open(output_file, "w") as f:
                json.dump(all_dataset_entries, f, indent=4)
            print(
                f"Combined dataset saved with {len(all_dataset_entries)} total entries"
            )

        else:
            # Standard processing (single structure or no filtering)
            process_trees_in_folder(
                folder_path=str(self.unpopulated_dir),
                output_json_path=str(output_file),
                max_trees=max_trees_to_process,
                filter_env=filter_env,  # type: ignore
                filter_metrics=filter_metrics,
                node_translations=self.node_translations,
                node_connectors=self.node_connectors,
                spoon_node_translations=self.spoon_node_translations,
                conditions=self.extracted_config["conditions"],
                actuator_actions=self.extracted_config["actuator_actions"],
                state_actions=self.extracted_config["state_actions"],
            )

        # Enrich dataset if requested (modify in place)
        if enrich_dataset:
            print(f"Enriching dataset {dataset_name}...")
            enrich_fn(
                input_file=str(output_file),
                output_file=str(output_file),  # Same file - overwrite
                handcoded_examples=None,  # Use default examples
                node_translations=self.node_translations,
                node_connectors=self.node_connectors,
                spoon_node_translations=self.spoon_node_translations,
            )

        return str(output_file)

    def upload_dataset(
        self,
        dataset_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        repo_id: Optional[str] = None,
    ) -> str:
        """
        Upload a dataset to Hugging Face Hub.

        Args:
            dataset_path: Path to the dataset file
            dataset_name: Name of the dataset
            repo_id: Repository ID on Hugging Face Hub

        Returns:
            URL or identifier of the uploaded dataset
        """
        # Implementation would go here
        # For now, return a placeholder
        return f"Dataset {dataset_name} uploaded to {repo_id}"
