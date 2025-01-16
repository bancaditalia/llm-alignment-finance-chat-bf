# Financial AI Alignment

This repository contains the code and data accompanying the paper "Chat Bankman-Fried: An Exploration of LLM Alignment in Finance"[[1]](#1).   

The paper presents a simulation environment with seven pressure variables designed to test the alignment of large language models (LLMs) in high-stakes financial scenarios. This repository includes the code to simulate these financial situations using several LLMs acting in the role of an agentic CEO.

## Preliminaries

The code and data have been tested on Linux systems using Python 3.11.

## Download experimental data

Download link: https://www.dropbox.com/scl/fi/dknzeueneaxhzl52m5f2e/sim_data.zip?rlkey=b13h7l1r3htfbzxfahw5xzcve&st=3qiwv04p&dl=0

The data should be placed into ```data/simulation_results/```

### Installation Instructions

To install the codebase, follow these steps:

- Clone the repository:
```
git clone git@github.com:bancaditalia/llm-alignment-finance-chat-bf

```
- Install [anaconda](https://docs.anaconda.com/free/anaconda/)

- Create a virtual environment using python 3.11
```
conda create -n ai_align python=3.11

conda activate ai_align
```

- Install the package running bash install-dev.sh
```
bash install-dev.sh
```

- Create a .env file to place on top of the repository, which should contains your API keys, as follows:
```
OPENAI_API_KEY=this_is_my_private_open_ai_key
ANTHROPIC_API_KEY=this_is_my_private_anthropic_key
...
```

Your repository structure should look like this:
```bash
.
├── .env
├── install-dev.sh
├── install.sh
├── notebooks
│   └── run_simulation.ipynb
├── requirements.txt
├── src
│   ├── analytics
│   ├── experiments
│   │   ├── utils.py
│   │   ├── runner.py
│   │   ├── exec_full_sim.py
│   │   └── exec_baseline.py
│   ├── simconfig
│   │   ├── base_config.py
│   │   ├── sim_env.yaml
│   │   ├── llm_system_prompt.txt
│   │   └── agent_prompts.json
│   ├── tests
│   │   └── sample_llm.json
│   ├── main.py
│   ├── setup.cfg
│   ├── setup.py
│   └── simulator
│       ├── agents.py
│       ├── core.py
│       ├── logging.py
│       └── utils.py
├── data
│   └── simulation_results/
└── README.md
```



## Usage Guide

### Initial Usage

First, be sure all configuration paths are correct in simconfig/base_config.py, so that your simulation env is setup correctly.
The Jupyter notebook "run_simulation.ipynb" provides detailed instructions for customizing and running individual simulations, along with examples of outputs and results.

### Basic Simulation

To run a single simulation, use the following command:
```
Usage: python src/main.py 

```

This will generate three output files:
```
── simulation_results_llm.json
── simulation_results_logs.json
── simulation_results_prompts.json
```

## Running Experiments

### Baseline Experiment

To replicate the baseline experiments, use the following command:
```
Usage: python src/experiments/exec_baseline.py
```
### Full Experiment

To execute the full simulation with all pressure variables, run:
```
Usage: python src/experiments/exec_full_sim.py
```

## Additional details

- Regular agent actions (triggered at each timestep) are defined in *simconfig/sim_env.yaml* along with environment properties.

- Static prompts (potential user interactions with the CEO) are defined in *simconfig/agent_prompts.json* for each agent and action.

- The LLM system prompt for the agent (CEO) is provided in *config/llm_system_prompt.txt*.

- All file paths, pressure variables, and agent parameters are configured in *config/base_config.py*.


## Authors

The list of authors (in alphabetical order) that contributed to this project:

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/gattodipiombo">
        <img src="https://avatars.githubusercontent.com/u/72857461?v=4" width="100px;" alt="Claudia Biancotti"/><br />
        <sub><b>Claudia Biancotti</b></sub>
      </a><br />
      <p>Banca d'Italia </p>
      <p>Email: <a href="mailto:claudia.biancotti@bancaditalia.it:">claudia.biancotti@bancaditalia.it</a></p>
    </td>
    <td align="center">
      <a href="https://github.com/carolinacamassabdi">
        <img src="https://avatars.githubusercontent.com/u/96301707?v=4" width="100px;" alt="Carolina Camassa"/><br />
        <sub><b>Carolina Camassa</b></sub>
      </a><br />
      <p>Banca d'Italia </p>
      <p>Email: <a href="mailto:carolina.camassa@bancaditalia.it:">carolina.camassa@bancaditalia.it</a></p>
    </td>
    <td align="center">
      <a href="https://github.com/Andrea94c">
        <img src="https://avatars.githubusercontent.com/u/11181598?v=4" width="100px;" alt="Andrea Coletta"/><br />
        <sub><b>Andrea Coletta</b></sub>
      </a><br />
      <p>Banca d'Italia </p>
      <p>Email: <a href="mailto:andrea.coletta@bancaditalia.it:">andrea.coletta@bancaditalia.it</a></p>
    </td>
    <td align="center">
      <a href="https://github.com/olivergiudice">
        <img src="https://avatars.githubusercontent.com/u/14348303?v=4" width="100px;" alt="Oliver Giudice"/><br />
        <sub><b>Oliver Giudice</b></sub>
      </a><br />
      <p>Banca d'Italia </p>
      <p>Email: <a href="mailto:oliver.giudice@bancaditalia.it:">oliver.giudice@bancaditalia.it</a></p>
    </td>
    <td align="center">
      <a href="https://github.com/AldoGl">
        <img src="https://avatars.githubusercontent.com/u/13199697?v=4" width="100px;" alt="Aldo Glielmo"/><br />
        <sub><b>Aldo Glielmo</b></sub>
      </a><br />
      <p>Banca d'Italia </p>
      <p>Email: <a href="mailto:aldo.glielmo@bancaditalia.it:">aldo.glielmo@bancaditalia.it</a></p>
    </td>
  </tr>
</table>

## References 

<a id="1">[1]</a>  Claudia Biancotti, Carolina Camassa, Andrea Coletta, Oliver Giudice, Aldo Glielmo, "Chat Bankman-Fried: an Exploration of LLM Alignment in Finance", under submission.

## Disclaimer

This package is an outcome of a research project. All errors are those of
the authors. All views expressed are personal views, not those of Bank of Italy.
