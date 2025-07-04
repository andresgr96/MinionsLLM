# Simulation Layer

This folder contains all the script and classes pertaining the simulation layer, which executes a given behaviour tree in a environment for a specific scenario and returns wether the agent solved the scenario or not.

## Key Definitions

- **Environment**: The physical environment where the agent operates, can be used with different agents and scenarios.
- **Agent**: The being that executes the behaviour tree and its tied to a specific environment due to scenario success calculation.
- **Scenario**: The task that the agent needs to solve in the environment. For example:
    - **Scenario Name**: Forage
    - **Prompt**: "Find food and bring it back to the nest"
- **Tree**: The behaviour tree given as an xml file that the agent will attempt to execute in the environment.
- **Behaviors**: Functions available to the agent to check environmental conditions or perform actions, represented as nodes in trees.
    - **Conditions**: Boolean checks for environmental or agent information, for example "is_agent_in_source" checks if the agent is within the coordinates of the source.
    - **Actions**: Have the agent perform actions in the environment, for example "move_towards_nest" has the agent change is direction towards the nest. 

## Creating an Environment

1. **Design the environment and base scenario**:
    - The first step is to compose your environment and think of at least one scenario to solve in it. We use here the example of foraging.
    - The goal of foraging is to find food, so naturally a scenario to solve is findinf that food and bringing it to the nest.

2. **Implement your environment**:
    - For consistency and modularity, inherit SimEnvironment to implement yours.
    - Since you havent implemented the agent yet, for now have the agent class from vi as placeholder.

## Creating an Agent

The main framework architecture consists of the following layers:

1. **Implement the agent**:
    - You should ALWAYS inherit the base agent class of the Vi simulator in order to implement your custom class.
    - Make sure to abstract any global information from the agent, as can be see in the SwarmAgent class the LightSensor handles some global information that the agent shouldnt have access to.

2. **Implementing Behaviours**:
    - Action and condition nodes should ALWAYS have docstring with the same structure seen in the SwarmAgent class.
    - These behaviours are automatically extracted by the `Layer_BT_parser.py` script.
    - If you wish to implement any helper function, make sure to start thefunction name with `helper_` for example `helper_check_scenario`

2. **Solving Scenarios**:
    - At least for now, the agent is responsible for checking the success of the scenario, therefore feed it as param to its init function.
    - To make sure that the simulation will return True once a scenario has been solved, signal it to the environment. For example:
        ```python
            if self.food_delivered:
                print(f"Solved: {scenario}") 
                self.env.success = True
        ```

## Running the Simulation

- Once you have finished all of the above, to run the simulation first generate a tree with `Layer_Run_simulator.py` by specifying a prompt. Then use the `Layer_Run_simulator.py` file and import the correct environment class.

- Example usage:
    ```sh
    python Layer_run_simulator.py --bt_path "./trees/behavior_tree_FT.xml" --n_agents 1 --scenario "forage" --headless
    ```



