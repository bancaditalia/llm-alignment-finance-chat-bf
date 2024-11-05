from abc import ABC

import numpy as np

from simulator.utils import (
    ActionScheduler,
    PromptsGenerator,
    LLM,
)
from simulator.logging import log_agent_action
import re


class Agent(ABC):

    def __init__(
        self,
        id,
        action_info: dict,
        attributes: dict,
        prompt_generator: PromptsGenerator,
        order: list = [],
    ):
        """Build the agent based on the info about the actions and attributes type.

        Args:
            action_info (dict): The information about actions of the agent (see yaml).
            attributes (dict): The attributes for the agent.
            prompt_generator (PromptsGenerator): The prompt generator to use to generate the LLM interaction for each action.
            order (list) : The order of the actions to be executed


            The attributes are actual attributes of the Agent class, like the money, the name, etc.
            While action_info indicates probability, when, and which frequency we should use for each Agent.
        """
        super().__init__()
        self.id = id
        self.available_functions = [f for f in dir(self) if not f.startswith("__")]
        self.action_info = action_info
        self.attributes = attributes
        self.prompt_generator = prompt_generator
        self.ordered_functions = order
        self.action_scheduler = self.__initialize_action_scheduler()

    def __initialize_action_scheduler(self):
        """For each action of the agent initializes the action scheduler, which decides when to perform the action."""
        if self.action_info is None:
            return None

        return {
            k: ActionScheduler(
                v.get("probability", 1),
                v.get("modulo_interval", 1),
                v.get("first_timestep", None),
                v.get("next_actions", []),
            )
            for k, v in self.action_info.items()
        }

    @classmethod
    def agent_builder(
        cls,
        agent_type: str,
        id,
        action_info: dict,
        attributes: dict,
        prompt_generator: PromptsGenerator,
        order: list = [],
    ):
        """Build the correct agent based on the agent type.

        Args:
            agent_type (str): The type of agent to build.
            action_info (dict): The information about actions of the agent (see yaml).
            attributes (dict): The attributes for the agent.

        Raises:
            ValueError: If the agent type is unknown.

        Returns:
            Agent: The built agent.
        """
        subclasses = {cls.__name__: cls for cls in Agent.__subclasses__()}

        if agent_type in subclasses:
            return subclasses[agent_type](
                id, action_info, attributes, prompt_generator, order
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def __str__(self):
        return f"{self.id} - {self.attributes}"

    def act(self, func_name, moderator, *args):
        """Apply the func_name to the agent and handle the execution and outcome with the moderator"""
        prompt_outcome, action_outcome = self.__getattribute__(func_name)(*args)
        moderator.handle_action_and_outcome(
            self, func_name, prompt_outcome, action_outcome
        )


class RandomEvents(Agent):

    @log_agent_action
    def profit_expectation(self, env) -> str:
        """Return the possible profit expectation from the market."""
        outcome = self.attributes["market_outcome"]
        if outcome is None:
            return "", None

        prompt = self.prompt_generator.generate_env_prompt(
            self.__class__.__name__,
            self.id,
            "profit_expectation",
            outcome,
            env.random_state,
        )
        return prompt, outcome


class CEO(Agent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seashells = self.attributes["initial_seashells"]
        self.bank_amount = self.attributes["initial_money"]

        self.llm = LLM(
            model=self.attributes["llm_model"],
            temperature=self.attributes["temperature"],
            top_p=self.attributes["top_p"],
        )

        self.reputation = 0
        self.jail = False
        self.bankruptcy = False
        self.initial_prompt()

    def extract_numerical_action(self, llm_asnwer: str):
        """Extract the numerical action from the llm_answer.

        We assume the llm_answer contains the number of the action within parenthesis, like:
            (1) for action 1 or (2) for action 2
        """
        ## Remove chat-gpt * characters used in the answer
        llm_asnwer = (
            llm_asnwer.replace("*", "").lower().replace("(", "").replace(")", "")
        )
        numerical_action = re.findall(r"final decision: (\d+)", llm_asnwer)

        ## Currently, we do not accept ambiguous answers, though we could opt to keep the last one.
        if len(numerical_action) == 1:
            return int(numerical_action[0])
        else:
            return None

    def __interact_llm(self, env, prompt):
        """This function actually sends the information to the LLM to decide the action"""
        env.simulation_prompts.append({"role": "user", "content": prompt})
        prompt_result = self.llm.user_prompt(env)
        return prompt_result

    @log_agent_action
    def initial_prompt(self):
        """Setting the initial prompt of the LLM CEO"""
        prompt = self.prompt_generator.generate_agent_prompt(
            self.__class__.__name__,
            self.id,
            "initial_prompt",
            None,
            np.random.default_rng(42),  ## here the seed is not meaningful
        )
        self.llm.system_prompt(prompt)
        return None, "Done"

    @log_agent_action
    def evaluate_trader_request(self, env, trader_prompt, amount):
        """Evaluate the request of money from the trader."""

        market_exp_prompt, market_exp = env.agents["events"].profit_expectation(env)
        if market_exp is not None:
            market_exp_prompt = (
                "You check market forecasts: << " + market_exp_prompt + " >>."
            )

        prompt = self.prompt_generator.generate_agent_prompt(
            self.__class__.__name__,
            self.id,
            "evaluate_trader_request",
            None,
            env.random_state,
            trader_prompt,
            market_exp_prompt,
        )
        prompt_result = self.__interact_llm(env, prompt)
        outcome = self.extract_numerical_action(prompt_result)

        ## accepted outcomes
        if outcome not in [1, 2, 3]:
            outcome = None

        ## if loan is accepted, we update the bank amount
        if outcome == 2:
            self.bank_amount -= amount // 2
        if outcome == 3:
            self.bank_amount -= amount

        return prompt_result, outcome


## Hedge-fund manager
class TradingBranch(Agent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.money_ask = self.attributes["money_ask"]

    @log_agent_action
    def ask_money(self, env):  ## ask back money to llm
        """Here we notice we need money to pay the loan, and we send the actual request."""
        prompt = self.prompt_generator.generate_agent_prompt(
            self.__class__.__name__,
            self.id,
            "ask_money",
            None,
            env.random_state,
            self.money_ask,
        )
        return prompt, self.money_ask
