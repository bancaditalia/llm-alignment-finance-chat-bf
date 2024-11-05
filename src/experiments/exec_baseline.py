### imports
import pandas as pd
import numpy as np
import itertools
from p_tqdm import p_map
from functools import partial

### config python file to start a simulation on FTX exchange scenario
import os
from experiments.runner import exec_simulation
from experiments.utils import save_simulation

###### Simulation Parameters we want to try out ######
######################################################

## how many parallel process to create
N_CPU = 8
## WHere to save the results
OUTPUT_FOLDER = "data/baseline/"
### Randomness of the simulation, for each combination of inputs
SIMULATION_SEEDS = list(range(25))

# Initial variable of money
BANK_ACCOUNT_MONEY = 300_000
# Initial Prompt for the CEO
INITIAL_LLM_VARIABLES = {
    "LOAN_DUE": np.asarray([BANK_ACCOUNT_MONEY * 0.5]),
    "NAGA_DOLLARS": np.asarray([BANK_ACCOUNT_MONEY]),
    "GOVERNANCE_ACCOUNT": [""],
    "ASSET_NAME": ["seashells"],
    "FEES": ["a flat fee paid every day by each customer"],
    "RISK_AVERSION_PROMPT": [""],
    "TRUST_PROMPT": [""],
    "REGULATION_PROMPT": [""],
    "REGRET_PROMPT": [""],
}

## Agent parameters to vary:
AGENT_PARAMETERS = {
    ("events", "market_outcome"): [None],  # , "low", "high"],
    ("CEO", "llm_model"): ["SampleLLM"],  # ["claude-3-haiku-20240307"]
}

## Action to Monitor
ACTIONS_TO_MONITOR = [("CEO", "evaluate_trader_request")]

if __name__ == "__main__":
    ## create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    ### run all combinations of the simulation
    ### Initial LLM-CEO details
    llm_items = INITIAL_LLM_VARIABLES.items()
    llm_fields = [x[0] for x in llm_items]
    llm_values = list(itertools.product(*[x[1] for x in llm_items]))

    ## Initial Agent details
    agent_items = AGENT_PARAMETERS.items()
    agent_fields = [x[0] for x in agent_items]
    agent_values = list(itertools.product(*[x[1] for x in agent_items]))

    ### try the different combinations
    run_sim = partial(
        exec_simulation,
        llm_fields=llm_fields,
        agent_fields=agent_fields,
        output_folder=OUTPUT_FOLDER,
        actions_to_monitor=ACTIONS_TO_MONITOR,
    )
    all_combinations = list(
        enumerate(itertools.product(llm_values, agent_values, SIMULATION_SEEDS))
    )
    sim_list_results = p_map(run_sim, all_combinations, num_cpus=N_CPU)

    ## save the experiment results
    save_simulation(sim_list_results, OUTPUT_FOLDER)
