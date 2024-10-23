import json
import functools

VERBOSE_LOGGING = True


class Logger:
    """A Logger class to log all the events happening in the simulation. Which is used to debug, save, and retrieve the simulation results."""

    def __init__(self) -> None:
        self.logs = []  ## as tuple (Entity : Event)

    def new_day(self, day, step) -> None:
        """Create a new sim day"""
        self.log_event(f"Simulation step {step} - day", str(day))

    def log_event(self, entity, event) -> None:
        if VERBOSE_LOGGING:
            print(f"({entity}) \n \t {event}")
        self.logs.append((entity, event))

    def get_logs(self, n_logs: int = 0) -> list:
        """
        Get the last n_logs logs. Default we get all the logs
        """
        return self.logs[-n_logs:]

    def save_json(self, filepath) -> None:
        with open(filepath + "_logs.json", "w") as f:
            json.dump(self.logs, f, indent=4)


logger = Logger()


def reset_logger():
    """Reset all logs in the current run.
    This is used to execute multiple runs within the same execution.
    """
    logger.logs = []


def log_agent_action(func):
    """a wrapper to log the agent action"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        obj = args[0]
        res = func(*args, **kwargs)
        if isinstance(res, tuple) and len(res) == 2:
            logger.log_event(obj.id, {func.__name__: res[1]})
        else:
            logger.log_event(obj.id, {func.__name__: res})
        return res

    return wrapper


############################################################
#################### TEST ##################################
############################################################

if __name__ == "__main__":

    class Agent:

        def __init__(self):
            self.id = "Agent1"

        @log_agent_action
        def salutation(self):
            return "Hello there, I'm agent 1!"

    agent = Agent()
    agent.salutation()
    print("Logs", logger.get_logs())
