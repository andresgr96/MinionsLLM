""" Here we define everything regarding the grammar, its production rules and parameters to control the size of the trees generated."""

# Defining the grammar rules
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

''' 
    This dictionary allows to further customize the tree generation according to the usecase.

    Params:
        "list_max": For list expansions, it controls the list_max amount of times that recursion is allowed to list_max - 1, forcing expanding to the single version of the symbol.
        "list_always": For list expansions, it controls the the exact amount of the list expansion to be chosen, forcing always having 'list_always' amount of the list expansion.
        "only": For any rule, it forces choosing the option at the given index, regardless of the current integer in the list.
        "parent": For any rule, if the immediate parent is of this type, it forces choosing the option at the given index, regardless of the current integer 
            in the list. It is a dictionary so it can be used to force different options for different parents.

 '''

grammar_parameters = {                                                                                                              
    "SEQn": {"list_max": 11, "parent": {"SEL": 0}},
    "SEQ": {"only": 0},                                                                    
    "Pn": {"list_max": 4},

}







