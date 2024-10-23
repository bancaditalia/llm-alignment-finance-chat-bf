from dotenv import load_dotenv

load_dotenv()

## Initial prompts from agents and env
INITIAL_PROMPT = "src/simconfig/llm_system_prompt.txt"
ENV_SETUP = "src/simconfig/sim_env.yaml"
AGENT_PROMPTS = "src/simconfig/agent_prompts.json"


### Simulation base parameters ####
PRESSURE_VARIABLES = {
    "GOVERNANCE_ACCOUNT": "",
    "ASSET_NAME": "seashells",
    "FEES": "a flat fee paid every day by each customer.",
    "RISK_AVERSION_PROMPT": "",
    "NAGA_DOLLARS": 300_000,
    "LOAN_DUE": 150_000,
    "TRUST_PROMPT": "",
    "REGULATION_PROMPT": "",
    "REGRET_PROMPT": "",
}

## Agent parameters to vary:
AGENT_PARAMETERS = {
    ("events", "market_outcome"): None,  # choices are [None, "low", "high"]
    ("CEO", "llm_model"): "gpt-4o-mini",
    #    ("CEO", "temperature"): 0,     ## not selected for now. The default is from the model itselfs.
    ("CEO", "initial_money"): PRESSURE_VARIABLES["NAGA_DOLLARS"],
    ("trading_branch", "money_ask"): PRESSURE_VARIABLES["LOAN_DUE"],
}


## Available LLM Models, we can add more.
MODELS_TYPE = {
    "Local": ["Meta-Llama-3.1-8B-Instruct", "phi-3.5-mini"],
    "Anthropic": ["claude-3-5-sonnet", "claude-3-haiku"],
    "OpenAI": [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "o1-preview",
        "o1-mini",
    ],
}
## Local models info
STOP_TOKENS = {"Meta-Llama-3.1-8B-Instruct": [128001, 128008, 128009]}


## understand the type of the model from the name. This is important as different providers have different APIs.
def get_model_type(model_name):
    ## sample test LLM
    if model_name == "SampleLLM":
        return model_name

    for key, value in MODELS_TYPE.items():
        if model_name in value:  ## we actually check the full name
            return key
        for v in value:
            if v in model_name:  ## we also check a subset (to include snapshot)
                return key
    return "Unknown"


## local apis
LOCAL_API_KEY = "local"
LOCAL_API_IP = "http://0.0.0.0:8000/v1"
