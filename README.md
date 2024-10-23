# Financial AI Alignment

This repository contains code and data for the paper "Chat Bankman-Fried: an Exploration of LLM Alignment in Finance"[[1]](#1).   

The paper proposes a simulation environment with seven pressure variables to test the alignment of LLM models in the high-stake financial situation.
In particular, this repository contains the code to simulate the financial scenario using several LLMs acting as a agentic-CEO. 

## Preliminaries

The code and data are tested on linux system, using using python 3.11. 

Install the codebase as follows:

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

- Create a .env file to place on top of the repository, which should contains your OPENAI_API_KEY, as follows:
```
OPENAI_API_KEY=this_is_my_private_open_ai_key
ANTHROPIC_API_KEY=this_is_my_private_anthropic_key
```


You repo should look like this:
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



## Walkthrough Usage

First: check and setup the config files (simconfig/base_config.py), to ensure all the paths are correct and your simulation env is setup correctly.

The notebook "run_simulation.ipynb" contains a details guide to personalize and run a single simulation. Also it shows the output and the simulation results.

## Basic Usage

The generic usage to test a single simulation scenario:
```
Usage: python src/main.py 

```

This will create three outputs: 
```
── simulation_results_llm.json
── simulation_results_logs.json
── simulation_results_prompts.json
```

## Experiments


The **baseline** experiments can be replicated using the following command:
```
Usage: python src/experiments/exec_baseline.py
```

While the **full-test** with all the pressure variables can be executed as follows:
```
Usage: python src/experiments/exec_full_sim.py
```

## Additional details

The agent regular actions (i.e., executed at regular timesteps) should be defined in the "simconfig/sim_env.yaml", along the env properties.

The static prompts (i.e., possible messages from users to the CEO) should be defined in the "simconfig/agent_prompts.json" for each agent and action. 

The "config/llm_system_prompt.txt" provides the system prompt for the LLM Agent (CEO).

The "config/base_config.py" defines all the paths of these files, along the basic pressure variables and agent parameters.




## Authors (Alphabetical order)

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
        <img src="https://media.licdn.com/dms/image/v2/D4E03AQGQ9iX5WCcviQ/profile-displayphoto-shrink_400_400/profile-displayphoto-shrink_400_400/0/1693412211925?e=1735171200&v=beta&t=vu_uxEND67ooyRwcRgTXmBC1LSr_B3ivmBjaklu1KSE" width="100px;" alt="Andrea Coletta"/><br />
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

## Reference 

<a id="1">[1]</a>  Claudia Biancotti, Carolina Camassa, Andrea Coletta, Oliver Giudice, Aldo Glielmo, "Chat Bankman-Fried: an Exploration of LLM Alignment in Finance", under submission.

## Disclaimer

This package is an outcome of a research project. All errors are those of
the authors. All views expressed are personal views, not those of Bank of Italy.