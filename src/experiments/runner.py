### imports
import pandas as pd
from simulator import logging

### config python file to start a simulation
from simulator.core import SimulationBuilder
import os
from collections import defaultdict

## Change to False here when launching a long run

RETURN_DATA = True


### run a single simulation
def exec_simulation(
    input_parameter_combination,
    llm_fields,
    agent_fields,
    output_folder,
    actions_to_monitor,
):
    ## These are all the parameter for the simulation
    sim_id, (llm_comb, agent_comb, seed) = input_parameter_combination

    ## output folder and files
    sim_output_path = os.path.join(output_folder, "logs/")
    os.makedirs(sim_output_path, exist_ok=True)
    sim_output_path = os.path.join(sim_output_path, f"simulation_{sim_id}")
    sim_exp_res_file = sim_output_path + "_tmp_results.csv"

    ## check if we already executed the simulation, and re-load it
    if os.path.exists(sim_exp_res_file):
        tmp_df = pd.read_csv(sim_exp_res_file, index_col=None)
        tmp_df.index = [sim_id]
    else:
        logging.VERBOSE_LOGGING = False
        ## tmp_results of this run, to save if it works well.
        tmp_results = defaultdict(list)

        ## create the simulation here and fill the ad-hoc parameters over the default one
        simulation_builder = SimulationBuilder(seed=seed, verbose=False)

        ## UPDATE the default LLM parameters
        for k, v in zip(llm_fields, llm_comb):
            ## save current values
            tmp_results[k] += [str(v)]
            simulation_builder.update_ceo_initialisation_prompt(key=k, value=v)

        ## UPDATE the default agent parameters
        for k, v in zip(agent_fields, agent_comb):
            ## save current values
            tmp_results["_".join(k)] += [str(v)]
            agent_name, agent_attr = k
            simulation_builder.update_agent_details(agent_name, agent_attr, v)

        ## improve this
        simulation_builder.sim_config_data["agents"]["trading_branch"]["attributes"][
            "money_ask"
        ] = int(simulation_builder.pressure_variables["LOAN_DUE"])
        simulation_builder.sim_config_data["agents"]["CEO"]["attributes"][
            "initial_money"
        ] = int(simulation_builder.pressure_variables["NAGA_DOLLARS"])

        ## update seed
        simulation_builder.update_sim_details("seed", seed)

        ## actual build
        simulation = simulation_builder.build()
        simulation.run()

        ## Extract simulation results
        for agent, action in actions_to_monitor:
            outcomes = simulation.extract_informations(agent, action)
            # NOTE: We are getting only last action, this is okay for our goal, but in the future multiple actions can be monitored
            tmp_results[action] += [outcomes[-1]]

        simulation.close(sim_output_path)
        tmp_df = pd.DataFrame(tmp_results, index=[sim_id])

        tmp_df.to_csv(sim_exp_res_file, index=False)

    if RETURN_DATA:
        return tmp_df
    else:
        return None
