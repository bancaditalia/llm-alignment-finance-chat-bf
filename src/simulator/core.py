import pandas as pd
import numpy as np
import json
import os
import pickle


### config python file to start a simulation
from simulator.logging import logger, reset_logger
from simulator.utils import StaticPromptsGenerator
from simulator import utils
from simconfig import base_config
from simulator.logging import logger

from simulator.agents import (
    CEO,
    TradingBranch,
    Agent,
)


class SimModerator:

    def __init__(self, id, simulation):
        self.id = id
        self.simulation = simulation

    def handle_action_and_outcome(self, agent, action, prompt_outcome, action_outcome):
        """This method handles the outcome of a previous action, deciding if we need to perform a new action or not."""
        if isinstance(agent, CEO):
            if prompt_outcome is not None:
                self.simulation.simulation_prompts.append(
                    {"role": "assistant", "content": prompt_outcome}
                )

        elif isinstance(agent, TradingBranch):
            if action == "ask_money":
                self.simulation.agents["CEO"].act(
                    "evaluate_trader_request",
                    self,
                    self.simulation,
                    prompt_outcome,
                    action_outcome,
                )

    def handle_actions_in_the_step(self):
        """Handle and decide which actions to perform in the current step, for each agent in the sim environment"""

        ## iterate over the agents, following the given sim order (in the config)
        for agent_key in self.simulation.agent_order:

            ## agent has been removed/deactivated
            if agent_key not in self.simulation.agents:
                continue

            ## get the agent
            agent = self.simulation.agents[agent_key]
            scheduler = agent.action_scheduler

            ## available actions for the agent
            available_actions = [
                act for act in agent.ordered_functions if act in scheduler
            ]

            ## execute
            for action in available_actions:
                ## scheduler decides the basic functions which are executed regularly
                if scheduler[action].perform_action(
                    self.simulation.current_step, self.simulation.random_state
                ):
                    agent.act(action, self, self.simulation)
                    if self.simulation.is_over():
                        break


class Simulation:
    """This is an event based simulation, on multiple steps/days"""

    def __init__(self, agents, environment, prompt_generator, sim_config):
        self.agents = agents
        self.sim_config = sim_config
        self.prompt_generator = prompt_generator
        self.start_date = pd.to_datetime(environment["start_date"])
        self.end_date = pd.to_datetime(environment["end_date"])
        self.timestep = environment["timestep"]
        self.simulation_days = pd.date_range(
            start=self.start_date, end=self.end_date, freq=self.timestep
        )
        self.asset = environment["asset"]
        self.seed = environment["seed"]
        self.output_file = environment["output_file"]
        self.current_date = environment["start_date"]
        self.agent_order = environment["agent_order"]
        self.random_state = np.random.default_rng(self.seed)
        self.moderator = SimModerator("Moderator", self)
        self.state = []
        self.actions = []
        ## if a simulation if over, can't restart
        self.sim_over = False

        ## empty list of users, can be extended with additional agents
        self.user_agents = []

        self.history = []  # Fill with events based on what happens in the environment
        self.market = None
        self.current_step = -1

        ## A list of prompts and information to understand current and previous states
        self.simulation_prompts = []

    def __str__(self):
        return f"Environment with {len(self.agents)} agents, starting at {self.start_date} and ending at {self.end_date}"

    def is_over(self):
        if self.sim_over:
            return True

        ## jail
        ceo_to_jail = self.agents["CEO"].jail

        ## bankruptcy
        ceo_bankruptcy = self.agents["CEO"].bankruptcy

        # end days
        end_days = self.current_step == len(self.simulation_days) - 1

        # no users
        self.sim_over = (
            len(self.user_agents) == 0 or ceo_to_jail or ceo_bankruptcy or end_days
        )

        ## Logging
        if self.sim_over:
            if end_days:
                reason = "end of sim-days."
            elif len(self.user_agents) == 0:
                reason = "No more users"
            elif ceo_to_jail:
                reason = "CEO is in jail"
            elif ceo_bankruptcy:
                reason = "CEO is bankrupt"

            logger.log_event("Simulation", "Simulation ended due to " + reason)
            self.simulation_prompts.append(
                {"role": "user", "content": "Simulation ended due to " + reason}
            )

        return self.sim_over

    ## Simulation loop
    def begin_step(self):
        """all the actions to perform at the beginning of a new step"""
        self.current_step += 1
        self.current_day = self.simulation_days[self.current_step]
        logger.new_day(self.current_day, self.current_step)

    def step(self):
        """Advance the simulation step"""
        self.begin_step()
        self.moderator.handle_actions_in_the_step()
        return self.end_step()

    def end_step(self) -> bool:
        """all the actions to perform at the end of a new step.

        Return True if the simulation can continue, False otherwise
        """
        return not self.is_over()

    def run(self):
        """Actually run the full simulation"""
        try:
            while self.step():
                continue
        except Exception as e:
            logger.log_event("Simulation ended with an Error", str(e))
            raise e

    def extract_informations(self, agent_id, action):
        """This method extract all information (outcomes) about agent and related input action"""
        outcomes = []
        for log_entity, log_value in logger.logs:
            if log_entity == agent_id and action in log_value:
                outcomes.append(log_value[action])

        return outcomes

    def close(self, output_path: str = None, additional_info=None):
        logger.log_event("Simulation", "Simulation ended at " + str(self.current_day))
        self.simulation_prompts.append(
            {"role": "user", "content": "Simulation ended at " + str(self.current_day)}
        )

        ## add different output folder
        if output_path is None:
            output_path = self.output_file

        logger.save_json(output_path)

        ## llm inputs
        with open(output_path + "_llm_interaction.json", "w") as f:
            json.dump(self.agents["CEO"].llm.get_messages(self), f, indent=4)

        if additional_info is not None:
            with open(output_path + "_additional.pickle", "wb") as f:
                pickle.dump(additional_info, f)

        ## save the sim_config
        with open(output_path + "_config.json", "w") as f:
            json.dump(self.sim_config, f, indent=4)

        ## save the prompt
        prompt_config = self.prompt_generator.prompt_config
        with open(output_path + "_prompt_config.json", "w") as f:
            json.dump(prompt_config, f, indent=4)

        reset_logger()


############################################################
#################### Simulation Builder ##################################
############################################################


class SimulationBuilder:
    ### By default this Builder starts loading the default parameters from src.simconfig folder.

    def __init__(self, seed: int, verbose: bool = False, agent_parameters: dict = None):
        """A builder to create a simulation environment. Agent parameters contains the parameters for the agents to be used in the simulation"""
        ## Load config for Environment and basic agents
        self.seed = seed
        self.random_obj = np.random.default_rng(seed)
        self.sim_config_data = utils.load_config()
        self.sim_config_data["simulation_parameters"]["seed"] = seed
        self.agent_parameters = (
            base_config.AGENT_PARAMETERS
            if agent_parameters is None
            else agent_parameters
        )
        self.__default_prompt()
        self.verbose = verbose

    def __default_prompt(self):
        ### CEO initial prompt
        initial_prompt_gen = utils.InitialPrompt()
        self.initial_prompt_ceo = initial_prompt_gen.prompt_template
        self.pressure_variables = initial_prompt_gen.pressure_variables

        ## AGENTS initial prompts
        ## combine the initial prompt with the basic prompts for all the agents, from src/config/AGENT_PROMPTS.json
        prompts_file = base_config.AGENT_PROMPTS
        self.users_prompt_data = json.load(open(prompts_file, "r"))

        ## AGENTS attributes
        ## set attributes of the agent according the input config
        for (agent, parameter), value in self.agent_parameters.items():
            self.update_agent_details(agent, parameter, value)

    def view_sim_agents(self) -> list:
        """give you a list of the agents, sorted by their execution priority at each simulation day"""
        return self.sim_config_data["simulation_parameters"]["agent_order"]

    def view_sim_details(self) -> dict:
        """a details of the main parameters of the env/sim"""
        return self.sim_config_data["simulation_parameters"]

    def update_sim_details(self, key: str, value) -> dict:
        """a details of the main parameters of the env/sim"""
        self.sim_config_data["simulation_parameters"][key] = value

    def update_agent_details(self, agent_name: str, key: str, value) -> dict:
        """give you details (setup) of agent_name"""
        self.sim_config_data["agents"][agent_name]["attributes"][key] = value

    def view_agent_details(self, agent_name: str) -> dict:
        """give you details (setup) of agent_name"""
        return self.sim_config_data["agents"][agent_name]

    def view_agent_prompts(self, agent_name: str) -> dict:
        """The basic prompts for each agent"""
        return self.users_prompt_data["Agents"][agent_name]

    def ceo_initialisation_prompt(self) -> dict:
        """return initialization for the CEO. This can be modified by the ad-hoc function"""
        return self.pressure_variables

    def update_ceo_initialisation_prompt(self, key: str, value: str) -> dict:
        """override the initialization for the CEO"""
        self.pressure_variables[key] = value

    def __build_prompt_generator(self):
        ## intial_prompt
        initial_prompt_ceo = self.initial_prompt_ceo.format(**self.pressure_variables)
        self.users_prompt_data["Agents"]["CEO"]["initial_prompt"] = {
            "template": "{prompt}",
            "placeholders": [initial_prompt_ceo],
        }
        self.prompt_generator = StaticPromptsGenerator(self.users_prompt_data, logger)

    def __build_agents(self):
        agents = {}
        # Extract agents and environment (simulation parameters)
        agents_template = self.sim_config_data.get("agents", {})
        environment = self.sim_config_data.get("simulation_parameters", {})

        # Building agents...
        for agent_name, agent_info in agents_template.items():
            agent = Agent.agent_builder(
                agent_info["class"],
                agent_name,
                agent_info["actions"],
                agent_info.get("attributes", {}),
                self.prompt_generator,
                agent_info.get("order", []),
            )
            agents[agent_name] = agent

            if self.verbose:
                print(agent)

        if self.verbose:
            print("###" * 10)

        return agents, environment

    def __build_sim(self, agents, environment):
        env = Simulation(
            agents,
            environment,
            prompt_generator=self.prompt_generator,
            sim_config=self.sim_config_data,
        )
        return env

    def build(self):
        self.__build_prompt_generator()
        agents, environment = self.__build_agents()
        sim = self.__build_sim(agents, environment)
        return sim
