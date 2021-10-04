"""
POST PROCESSING PIPELINE FOR HOPPER DATA

Luke Drnach
September 15, 2021
"""
import os
from examples.hopper.organizehopperdata import organize_solutions_by_success, split_config_from_data
from examples.hopper.runmeritanalysis import run_hopper_merit
from examples.hopper.generatemeritfigures import compare_hopper_merit, save_merit_to_table
from examples.hopper.hopperfootheight import compare_chance_footheights, compare_ERM_footheights, calculate_mean_footheight, calculate_peak_footheight
from examples.hopper.robusthoppingopt import RobustOptimizationOptions

# File Directories
REF_DIR = os.path.join("examples","hopper","reference_linear","strict_nocost")
ERM_DIR = os.path.join("examples","hopper","robust_nonlinear","decimeters_1e3","erm","success")
CCS_DIR = os.path.join("examples","hopper","robust_nonlinear","decimeters_1e3","erm_cc","success")

# Anaylsis Switches
MERIT_REF = False

ORGANIZE_ERM = False
HEIGHT_ERM = False
MERIT_ERM = False

ORGANIZE_CC = False
HEIGHT_CC = True
MERIT_CC = False

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

# Do the merit comparison, footheight comparison
if REF_DIR is not None and ERM_DIR is not None and CCS_DIR is not None:
    compare_hopper_merit(REF_DIR, ERM_DIR, CCS_DIR, CCS_DIR)
    save_merit_to_table(REF_DIR, ERM_DIR, CCS_DIR, CCS_DIR)
    calculate_mean_footheight(ERM_DIR, CCS_DIR, REF_DIR)
    calculate_peak_footheight(ERM_DIR, CCS_DIR, REF_DIR)