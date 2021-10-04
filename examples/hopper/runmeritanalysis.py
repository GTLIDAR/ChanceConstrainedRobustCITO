"""


"""
from trajopt.analysistools import batch_run_merit_analysis, batch_run_merit_analysis_min, batch_run_merit_analysis_fisher
from trajopt.contactimplicit import ContactImplicitDirectTranscription as ContactOpt
from systems.hopper.hopper import Hopper
import os

def run_hopper_merit(directory):
    # directory = os.path.join("examples","robust_block","success")
    # Recreate the trajopt
    hopper = Hopper()
    hopper.Finalize()
    trajopt = ContactOpt(hopper, hopper.multibody.CreateDefaultContext(),
                        num_time_samples = 101,
                        minimum_timestep = 3./100,
                        maximum_timestep = 3./100)
    batch_run_merit_analysis(trajopt, directory, filename='trajoptresults.pkl')

def main_hopper_nominal():
    # Specify the directory
    directory = os.path.join('examples','hopper','reference_linear','strict_nocost')
    run_hopper_merit(directory)

def main_hopper_erm():
    # Specify the directory
    directory = os.path.join('examples','hopper','robust_nonlinear','erm_1e5','success')
    run_hopper_merit(directory)

def main_hopper_chance_erm():
    # Specify the directory
    directory = os.path.join('examples','hopper','robust_nonlinear','cc_erm_1e5_mod','success')
    run_hopper_merit(directory)

if __name__ == "__main__":
    main_hopper_nominal()
    main_hopper_erm()
    main_hopper_chance_erm()