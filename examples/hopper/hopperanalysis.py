"""
POST PROCESSING PIPELINE FOR HOPPER DATA

Luke Drnach
September 15, 2021
"""
import os
from examples.hopper.organizehopperdata import organize_solutions_by_success, split_config_from_data
from examples.hopper.runmeritanalysis import run_hopper_merit
from examples.hopper.generatemeritfigures import compare_hopper_merit
from examples.hopper.hopperfootheight import compare_chance_footheights, compare_ERM_footheights
from examples.hopper.robusthoppingopt import RobustOptimizationOptions

# File Directories
REF_DIR = os.path.join("examples","hopper","reference_highfriction","strict")
ERM_DIR = os.path.join("examples","hopper","robust_highfriction","erm_1e5","success")
CCS_DIR = os.path.join("examples","hopper","robust_highfriction","cc_erm_1e5")

# Anaylsis Switches
MERIT_REF = True

ORGANIZE_ERM = False
HEIGHT_ERM = True
MERIT_ERM = True

ORGANIZE_CC = True
HEIGHT_CC = True
MERIT_CC = True

# Organize the data
if ORGANIZE_ERM and ERM_DIR is not None:
    split_config_from_data(ERM_DIR)
    ERM_DIR = organize_solutions_by_success(ERM_DIR)
if ORGANIZE_CC and CCS_DIR is not None:
    split_config_from_data(CCS_DIR)
    CCS_DIR = organize_solutions_by_success(CCS_DIR)

# Run the foot height comparisons
if HEIGHT_ERM and ERM_DIR is not None:
    compare_ERM_footheights(ERM_DIR, reffile = os.path.join(REF_DIR, 'trajoptresults.pkl'))

if HEIGHT_CC and ERM_DIR is not None and CCS_DIR is not None:
    compare_chance_footheights(ERM_DIR, CCS_DIR, REF_DIR)

# Run merit analysis
if MERIT_REF and REF_DIR is not None:
    run_hopper_merit(REF_DIR)
if MERIT_ERM and ERM_DIR is not None:
    run_hopper_merit(ERM_DIR)
if MERIT_CC and CCS_DIR is not None:
    run_hopper_merit(CCS_DIR)

# Do the merit comparison
if REF_DIR is not None and ERM_DIR is not None and CCS_DIR is not None:
    compare_hopper_merit(REF_DIR, ERM_DIR, CCS_DIR, CCS_DIR)