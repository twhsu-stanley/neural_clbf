"""A mock experiment for use in testing"""
from copy import copy
from typing import Optional, TYPE_CHECKING
from unicodedata import name

from matplotlib.pyplot import figure
import pandas as pd
import torch
from torch.nn.functional import linear
import tqdm

from neural_clbf.experiments import Experiment
from neural_clbf.systems.utils import ScenarioList

# turtlebot script imports
from integration.integration.turtlebot_scripts import odometry_status
from integration.integration.turtlebot_scripts.send_command import execute_command
import os


if TYPE_CHECKING:
    from neural_clbf.controllers import Controller  # noqa


class RealTimeSeriesExperiment(Experiment):
    """
    
    An experiment for running a controller 
    a on real turtlebot.

    """

    def __init__(
            self,

            # turtlebot interface parameters
            command_publisher,
            rate,
            listener,
            move_command,
            odom_frame,
            base_frame,

            # clbf parameters
            name: str,
            start_x: torch.Tensor,
            scenarios: Optional[ScenarioList] = None,

            # Note on t_sim: actual time taken does not seem
            # to correspond to this. Raise this value as needed to
            # make script run long enough for turtlebot
            # to reach the destination
            t_sim: float = 300.0,
    ):
        """Initialize an experiment for controller performance on turtlebot.
        args:
            scenarios: a list of parameter scenarios to sample from. If None, use the
                       nominal parameters of the controller's dynamical system
            t_sim: the amount of time to simulate for. Note on t_sim: the actual
                        amount of time taken to run the experiment (when it's not a simulation) does
                        not seem to correspond to this value. Raise t_sim as needed to 
                        make the script run long enough for the turtlebot to reach the destination.
        """
        super(RealTimeSeriesExperiment, self).__init__(name)

        # clbf parameters
        self.start_x = start_x
        self.scenarios = scenarios
        self.t_sim = t_sim

        # turtlebot interface parameters
        self.rate = rate
        self.command_publisher = command_publisher
        self.listener = listener
        self.move_command = move_command
        self.odom_frame = odom_frame
        self.base_frame = base_frame

    @torch.no_grad()
    def run(self, controller_under_test: "Controller") -> pd.DataFrame:
        """
        Run the experiment, likely by evaluating the controller, but the experiment
        has freedom to call other functions of the controller as necessary (if these
        functions are not supported by all controllers, then experiments will be
        responsible for checking compatibility with the provided controller)
        args:
            controller_under_test: the controller with which to run the experiment
        returns:
            a pandas DataFrame containing the results of the experiment, in tidy data
            format (i.e. each row should correspond to a single observation from the
            experiment).
        """
        # reset turtlebot odometry
        os.system("timeout 3 rostopic pub /reset std_msgs/Empty '{}'")

        # Deal with optional parameters
        if self.scenarios is None:
            scenarios = [controller_under_test.dynamics_model.nominal_params]
        else:
            scenarios = self.scenarios

        # Set up a dataframe to store the results
        results_df = pd.DataFrame()

        # Determine the parameter range to sample from
        # this pulls the min and max values from the turtlebot model script
        # so no need to modify this
        parameter_ranges = {}
        for param_name in scenarios[0].keys():
            param_max = max([s[param_name] for s in scenarios])
            param_min = min([s[param_name] for s in scenarios])
            parameter_ranges[param_name] = (param_min, param_max)

        # Generate a tensor of start states
        n_dims = controller_under_test.dynamics_model.n_dims
        n_controls = controller_under_test.dynamics_model.n_controls

        # get intial state from odometry
        (self.position, self.rotation) = odometry_status.get_odom(self.listener, self.odom_frame, self.base_frame)

        # Execute!
        delta_t = controller_under_test.dynamics_model.dt
        num_timesteps = int(self.t_sim // delta_t)
        x_current = torch.zeros(1, n_dims).type_as(self.start_x)
        u_current = torch.zeros(1, n_dims).type_as(self.start_x)
        controller_update_freq = int(controller_under_test.controller_period / delta_t)
        prog_bar_range = tqdm.trange(
            0, num_timesteps, desc="Controller Rollout", leave=True
        )

        for tstep in prog_bar_range:
            
            (self.position, self.rotation) = odometry_status.get_odom(self.listener, self.odom_frame, self.base_frame)
            x_current[0, :] = torch.tensor([self.position.x, self.position.y, self.rotation]) + self.start_x

            # Get the control input at the current state if it's time
            if tstep % controller_update_freq == 0:

                # TODO This line currently has issues, and will not work as-is.
                # Currently only works with u_nominal specifically:
                # u_current = controller_under_test.dynamics_model.u_nominal(x_current)
                u_current = controller_under_test.u(x_current)

                # set the output command to the command obtained from the
                # dynamics model
                linear_command = u_current[0][0].item()
                print(linear_command)
                angular_command = u_current[0][1].item()

                # pull the control limits from the turtlebot system file
                (upper_command_limit, _)= controller_under_test.dynamics_model.control_limits

                # call the function that sends the commands to the turtlebot
                execute_command(self.command_publisher, self.move_command, linear_command, angular_command, self.position, self.rotation, upper_command_limit)

                # Log the current state and control
                base_log_packet = {"t": tstep * delta_t}

                # Include the parameters
                param_string = ""
                for param_name, param_value in scenarios[0].items():
                    param_value_string = "{:.3g}".format(param_value)
                    param_string += f"{param_name} = {param_value_string}, "
                    base_log_packet[param_name] = param_value
                base_log_packet["Parameters"] = param_string[:-2]

                # Log the goal/safe/unsafe/out of bounds status
                x = x_current[0, :].unsqueeze(0)
                is_goal = controller_under_test.dynamics_model.goal_mask(x).all()
                is_safe = controller_under_test.dynamics_model.safe_mask(x).all()
                is_unsafe = controller_under_test.dynamics_model.unsafe_mask(x).all()

                for measurement_label, value in zip(
                        ["goal", "safe", "unsafe"], [is_goal, is_safe, is_unsafe]
                ):
                    base_log_packet[measurement_label] = value.cpu().numpy().item()

                # If this controller supports querying the Lyapunov function, save that
                if hasattr(controller_under_test, "V"):
                    V = controller_under_test.V(x).cpu().numpy().item()  # type: ignore

                    log_packet = copy(base_log_packet)
                    log_packet["measurement"] = "V"
                    log_packet["value"] = V
                    results_df = results_df.append(log_packet, ignore_index=True)

        results_df = results_df.set_index("t")
        return results_df

    def plot():
        """
        
        plot function here is left empty. It is required
        because of the Experiment class, but has no
        purpose for this script.
        
        """
        pass