## Ongoing ##
1. Check Ollama prompt format, for some reason using the same model with llamacpp and ollama has different performance.
2. FIX THE RUM SIMULATION EXAMPLE (Mandatory)
3. Int list size parameter during gen is incorrect (Mandatory)
4. Add a way to add custom system prompts for dataset generation

## Done, Needs Checking ##
1. Add a way to create a custom dataset generation with custom grammar. (Done, needs testing)
2. That the nodes and its types are automatically deduced by the program from the class itself trough docstrings (Done, Needs Checking)
3. Dataset enrichment, filtering  (Filtering and Enrichment done but need checking)
4. Add example for custom dataset generation with custom grammar
5. PROPER SYNTAX CHECK (Mandatory, Done needs checking)
6. Check defult RobotAgent node translations (Mandatory)
7. Make Ollama autoimport work on linux too (Mandatory)
8. Use base env in agent definitions


## Fully Completed ##
1. Implement further control tree structure + explain a lot more in system prompt b. 
2. Add ALL metrics to environment, like parts placed in any of the areas etc
3. Decide on models and rebuild experimental loop.
4. Upload dataset to hf

## Not started ##
1. Add code quality tools like black, isort, ruff, mypy, etc. (Mandatory!)
2. Add tqdm to dataset generation processes, add a verbose option to dataset generation to print all that we currently print but that is too verbose when just using the framework and not debugging (Mandatory)
3. Final goal is to make a full on tutorial on how to use the framework. (Mandatory)
    a. The tutorial should be on how to use the framework for an entirely new problem.
    b. Therefore it should include a guide on how to start the design of the problem first to understand the requirements.
    c. Then we should go over sequentially on the steps an user would need to do to set up the problem in the framework.
4. Check properly the mixed structures datagen procedure for inneficiencies.
5. Make agent functions that are not primitives starting with _ or ubnder to be ifnored and not depending on the name helper
6. Check if its posible to make a base agent that inherits from VIs agent.
7. Update readme at the end. (Mandatory)

## Future Work ##
1. Instead of first making trees and then giving them to the LLM and filter for some metrics:
    a. Have the LLM think of a task and the metrics it needs to achieve.
    b. Then generate a tree that achieves the task and the metrics.
    c. Then filter the tree for the metrics. Until metrics achieved,
    d. If metrics not achieved, give back to the LLM with sim feedback and conv history so it can improve the tree iteratively.
    e. Set max retries for the LLM to improve the tree.
    f. Also keep track of dataset distribution and metrics achieved so LLM doesnt bias the dataset into specific metric/s.
    g. Also use the grammar validator so the LLM doesnt generate syntacticallyinvalid trees.
2. Allow for more providers other than OpenAI
3. Modularize better the simulation part so changing simulators is easier.


## Ideas ##
1. Add example for custom nodes
2. Examples of creating a custom environment, agent and using it in the simulation. (Optional, but could be very easy just reuse my current ones but make them maybe simpler)
3. For practice, set these tools to run automatically on github actions (Optional)
4. Change the llms layer function for getting the behaviors to use the new function call_behaviors() adhere to the new docstring format. (Optional)
5. Add to dataset b structured output so the LLM can also include which metric to check for when filtering 


