"""


"""
from trajopt.analysistools import batch_run_merit_analysis, batch_run_merit_analysis_min, batch_run_merit_analysis_fisher
from trajopt.contactimplicit import ContactImplicitDirectTranscription as ContactOpt
from systems.block.block import Block
import os

def run_block_merit(directory):
    # directory = os.path.join("examples","robust_block","success")
    # Recreate the trajopt
    block = Block()
    block.Finalize()
    trajopt = ContactOpt(block, block.multibody.CreateDefaultContext(),
                        num_time_samples = 101,
                        minimum_timestep = 1./100,
                        maximum_timestep = 1./100)
    batch_run_merit_analysis(trajopt, directory, filename='trajoptresults.pkl')

def main_block_nominal():
    # Specify the directory
    directory = os.path.join("data",'IEEE_Access','sliding_block','PaperResults','warm_start')
    run_block_merit(directory)

def main_block_erm():
    # Specify the directory
    directory = os.path.join("data",'IEEE_Access','sliding_block','ERM')
    run_block_merit(directory)

def main_block_chance_erm():
    # Specify the directory
    directory = os.path.join("data",'IEEE_Access','sliding_block','ERM+CC')
    run_block_merit(directory)

if __name__ == "__main__":
    main_block_nominal()
    main_block_erm()
    main_block_chance_erm()