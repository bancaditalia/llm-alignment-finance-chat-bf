import re
import json
import yaml
import numpy as np

from abc import ABC, abstractmethod

from openai import OpenAI
import anthropic
from tests.sample_llm import SampleLLM

from functools import partial

from simconfig import base_config


class LLM:

    def __init__(self, model: str, temperature: float = None, top_p: float = None):
        """The LLM class to interact

        NOTE: We assume you have the API key saved in an .env file in the project folder.

        """
        self.model_type = base_config.get_model_type(model)
        if self.model_type == "Local":
            self.client = OpenAI(
                api_key=base_config.LOCAL_API_KEY,  # this should be set when initializing the VLLM api server
                base_url=base_config.LOCAL_API_IP,
            )
        elif self.model_type == "Anthropic":
            self.client = anthropic.Anthropic()
        elif self.model_type == "OpenAI":
            self.client = OpenAI()
        elif self.model_type == "SampleLLM":
            self.client = SampleLLM()
        else:
            raise ValueError(
                f"Model {model} not found in the configuration, for any of the available model types."
            )

        self.max_hist_messages = (
            np.inf
        )  ## max nr. messages to keep in the LLM API request
        self.temperature = temperature
        self.top_p = top_p
        self.model = model

    def __summarise_messages(self, messages) -> list:
        """Shorten the messages to the last max_hist_messages"""
        if len(messages) > self.max_hist_messages:
            messages = messages[-self.max_hist_messages :]
        return messages

    def __model_prompt(self, messages) -> str:
        """Call the LLM API and return the response."""
        messages = self.__summarise_messages(messages)

        ## build the correct function to execute
        if self.model_type == "SampleLLM":
            func = self.client.chat
        elif self.model_type == "Anthropic":
            func = self.client.messages.create
        else:
            func = self.client.chat.completions.create

        if self.temperature is not None:
            func = partial(func, temperature=self.temperature)
        if self.top_p is not None:
            func = partial(func, top_p=self.top_p)

        ## specific setup for the anthropic model and local model
        if self.model_type == "Anthropic":
            func = partial(func, max_tokens=1024, system=self.__system_prompt)
        if self.model_type == "Local":
            func = partial(
                func,
                extra_body={
                    "stop_token_ids": base_config.STOP_TOKENS.get(self.model, None)
                },
            )

        ## actual query
        response = func(model=self.model, messages=messages)

        ### extract anthropic response
        if self.model_type == "Anthropic":
            return str(response.content[0].text)
        elif self.model_type == "SampleLLM":
            return response
        else:
            return response.choices[0].message.content

    def user_prompt(self, env) -> str:
        """Actually retrieve the previous messages/interactions, and then use LLM API to return the response."""
        messages = self.get_messages(env)
        return self.__model_prompt(messages)

    def get_messages(self, env) -> list:
        """Extract the history messages to use in the API call"""
        messages = []

        ## we add the system prompt only if the model is not anthropic
        if self.model_type == "OpenAI" and "o1" not in self.model:
            messages += [{"role": "system", "content": self.__system_prompt}]
        elif "o1" in self.model:
            messages += [{"role": "user", "content": self.__system_prompt}]

        for message in env.simulation_prompts:
            messages.append(message)

        return messages

    def system_prompt(self, message) -> str:
        """Set the system prompt message for the LLM"""
        self.__system_prompt = message


class PromptsGenerator(ABC):
    """This class is used to create prompts from the agent actions. Used by the agents to interact with the LLMs."""

    def __init__(self, logger) -> None:
        self.logger = logger

    @abstractmethod
    def generate_agent_prompt(
        self, agent_type, agent_id, current_action, action_outcome, random_state, *args
    ) -> str:
        """Generate a prompt for the agent to interact with the LLMs, given the current action (e.g., extension_response)
            and related outcome (e.g., deny, approve, etc..)

        Args:
            agent_type : the type of agent
            agent_id (_type_): the agent id who is performing the action
            current_action (_type_): the current action performed by the agent
            action_outcome (_type_): the outcome of the action
            random_state (_type_): the random state to use for the generation
            *args : additional arguments to format the prompt

        Returns:
            str: the prompt to use
        """
        pass

    @abstractmethod
    def generate_env_prompt(
        self, event_type, event_id, current_event, event_outcome, random_state, *args
    ):
        """Generate a prompt based on current event."""
        pass


class StaticPromptsGenerator(PromptsGenerator):

    def __init__(self, prompt_config: dict, logger) -> None:
        super().__init__(logger)
        self.prompt_config = prompt_config

    def __generate_prompt_answer(
        self,
        element_type,
        agent_type,
        id,
        prompt_identifier,
        prompt_outcome,
        random_state,
        *args,
    ) -> str:
        assert (
            agent_type in self.prompt_config[element_type]
        ), f"{element_type} {agent_type} not found in the json prompt configuration"
        prompt_data = self.prompt_config[element_type][agent_type]
        assert (
            prompt_identifier in prompt_data
        ), f"Action {prompt_identifier} not found in the prompt configuration for {element_type} {agent_type}"
        prompt_data = prompt_data[prompt_identifier]

        ## template and prompts
        template = prompt_data["template"]
        prompts = (
            prompt_data["placeholders"]
            if prompt_outcome is None
            else prompt_data["placeholders"][prompt_outcome]
        )

        action_prompt = random_state.choice(prompts).format(*args)
        output = template.format(prompt=action_prompt)
        return output

    def generate_agent_prompt(
        self, agent_type, agent_id, current_action, action_outcome, random_state, *args
    ) -> str:
        return self.__generate_prompt_answer(
            "Agents",
            agent_type,
            agent_id,
            current_action,
            action_outcome,
            random_state,
            *args,
        )

    def generate_env_prompt(
        self, event_type, event_id, current_event, event_outcome, random_state, *args
    ) -> str:
        return self.__generate_prompt_answer(
            "Events",
            event_type,
            event_id,
            current_event,
            event_outcome,
            random_state,
            *args,
        )


class ActionScheduler:

    def __init__(
        self,
        probability: float,
        module_interval: int,
        first_timestep: int,
        next_actions: list,
    ):
        """The action scheduler class

        Args:
            probability (float): The probability of performing an action
            module_interval (int): The interval of timesteps to perform the action
            first_timestep (int): The first timestep to perform the action

            The module_interval is the interval of timesteps to perform the action, we execute an action if timestep % module_interval == 0
            If module_interval is None, we perform the action only at the first_timestep.
        """
        self.probability = probability
        self.modulo_interval = module_interval
        self.first_timestep = first_timestep
        self.next_actions = next_actions

    def __prob_execute(self, rng) -> bool:
        """Compute the probability of performing an action and return the outcome"""
        return rng.random() < self.probability

    def perform_action(self, current_timestep: int, rng: np.random.RandomState) -> bool:
        """whether to perform an action at the current timestep

        Args:
            current_timestep (int): The current timestep

        Returns:
            bool: whether to perform an action at the current timestep
        """
        if self.first_timestep is None:  ## actionable only when called by other agents
            return False
        elif self.modulo_interval is None:
            if current_timestep == self.first_timestep:
                return self.__prob_execute(rng)
        elif current_timestep == self.first_timestep:
            return self.__prob_execute(rng)
        elif (
            current_timestep >= self.first_timestep
            and current_timestep % self.modulo_interval == 0
        ):
            return self.__prob_execute(rng)

        return False


class InitialPrompt:
    """A utility class to handle the initial prompt of the CEO."""

    def __init__(self) -> None:
        self.prompt_dict = base_config.PRESSURE_VARIABLES
        self.initial_prompt_llm = base_config.INITIAL_PROMPT

        self.prompt_template = self.__load_base_prompt()
        self.pressure_variables = base_config.PRESSURE_VARIABLES

    def __load_base_prompt(self) -> str:
        with open(self.initial_prompt_llm, "r") as f:
            return f.read()


def compare_llm_and_exchange_numerical_results(s1, s2):
    """Compate the output from an LLM and compare it to a numerical result"""

    def extract_numbers(s):
        # Use regular expression to find all numbers in the string
        return re.findall(r"\d+", s)

    def clean_string(s):
        # Replace all punctuation and symbols with an empty string
        return re.sub(r"[^\w\s]", "", s)

    # Extract numbers from both strings
    numbers1 = extract_numbers(clean_string(s1))
    numbers2 = extract_numbers(clean_string(s2))

    # Convert the extracted numbers to lists of integers
    numbers1 = list(map(int, numbers1))
    numbers2 = list(map(int, numbers2))

    # Compare the sorted lists of numbers
    return sorted(numbers1) == sorted(numbers2)


def load_config():
    """Load sim env config"""
    config_file = base_config.ENV_SETUP
    config_data = yaml.safe_load(open(config_file, "r"))
    return config_data
