### Example of running a simulation scenario, using the default parameters in the simconfig folder
from simulator.core import SimulationBuilder
import os

## Load the simulation configuration and build
simulation = SimulationBuilder(seed=42, verbose=False).build()

## Run the simulation
simulation.run()

## Where to save simulation
output_path = "main_run/exp"
os.makedirs(output_path, exist_ok=True)

## Extract the CEO decision (i.e., result) from the simulation. This will extract the output of a given agent and action
action_to_monitor = [("CEO", "evaluate_trader_request")]
outcomes = simulation.extract_informations("CEO", "evaluate_trader_request")
print("###" * 20)
print("CEO evaluating trader request:", outcomes)
print("###" * 20)

## Actually close the simulation and save the results
simulation.close(output_path)
