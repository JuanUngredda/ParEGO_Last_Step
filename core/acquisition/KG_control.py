import numpy as np
import sys
import subprocess as sp
import os
import argparse
# from mistery_experiment import function_caller_mistery
# from KG_DEB_experiment import DEB_function_caller_test
# from KG_SRN_experiment import SRN_function_caller_test
#from KG_BNH_experiment import BNH_function_caller_test
# from KG_TNK_BNH_experiment import TNK_BNH_function_caller_test

from ParEGO_HOLE_experiment_Ushape import HOLE_function_caller_test
from ParEGO_POL_experiment_Ushape import POL_function_caller_test
from ParEGO_NO_HOLE_experiment_Ushape import NO_HOLE_function_caller_test
#from HVI_TNK_BNH_experiment import TNK_BNH_function_caller_test
#from HVI_BNH_experiment import BNH_function_caller_test
# from test_func_2_experiment import function_caller_test_func_2

# from new_branin_TS import function_caller_new_brannin_TS
# from test_func_2_TS import function_caller_test_func_2_TS
# from mistery_experiment_TS import function_caller_mistery_TS
# from RMITD_TS import function_caller_RMITD_TS
# from RMITD_EI import function_caller_RMITD_EI
# from RMITD_experiment import function_caller_RMITD
#
# from NN_TS import function_caller_NN_TS
# from NN_EI import function_caller_NN_EI
# from NN_experiment import function_caller_NN


# This is a bare script that receives args, prints something, wastes some time,function_caller_test_func_2_TS
# and saves something. Use this as a blank template to run experiments.
# The sys.argv = [demo_infra_usage.py (time_stamped_folder) (integer)]
# so use the (time_stamped_folder)/res/ to save outputs
# and use the (integer) to define experiment settings from a lookup table.
#
# To run this script 10 times distributed over CSC machines, on local computer type:
#  $ python fork0_to_csc.py \$HOME/cond_bayes_opt/scripts/demo_infra_usage.py 10 -v
#
# see fork0_to_csc.py for further help.



def run(args):
    """
    This is a stupid function just for demonstration purposes.
    It takes the args and prints something and saves something.
    In general any python experiment running code can go here.
    As long as the conda env and git branch are correctly set.
    """

    # define a list of all job settings (here is some random crap)
    np.random.seed(1)
    all_job_settings = np.random.uniform(size=(1000,))

    # use the args.k as a lookup into all_job_settings
    this_job_setting = all_job_settings[args.k]
    this_job_savefile = args.dirname+"/res/"+str(args.k)

    # Now to run some code!
    # Let's print something, say the conda env, args, computer and job setting?
    # get current conda environment that called this script
    conda_env = os.environ['CONDA_DEFAULT_ENV']

    # get current computer name
    hostname = sp.check_output(['hostname'], shell=True).decode()[:-1]

    # IMPORT AND RUN MODULES
    #functions = [function_caller_new_brannin_TS, function_caller_test_func_2_TS, function_caller_mistery_TS, function_caller_RMITD_TS, function_caller_RMITD_EI, function_caller_RMITD]
    #functions = [function_caller_RMITD ]
    functions = [HOLE_function_caller_test, NO_HOLE_function_caller_test, POL_function_caller_test]
    for func in functions:
        func(args.k)

    # save something to hard drive in /res/ subfolder
    with open(this_job_savefile, 'w') as f:
        f.write(output + "\n\n")

    # end of demo
    print("\nOutput saved to file: ", this_job_savefile, "\n\n\n\n")


if __name__=="__main__":
    ####################################### WARNING ####################################
    # ALL EXPERIMENT RUNNERS MUST HAVE THE FOLLOWING ARGS!!!! DO NOT CHANGE THIS!!!!
    ####################################### WARNING ####################################

    parser = argparse.ArgumentParser(description='Run k-th experiment from the look-up table')
    parser.add_argument('dirname', type=str, help='Experiment directory')
    parser.add_argument('k', type=int, help='Row in look-up table corresponding to specific experiment')

    # These arguments are assumed by the forking files. Use args.dirname+"/res/" as a results output directory.
    # In this file, define a list of aaallll the experiments you want to run and use args.k as a lookup index 
    # within the list. Save the output as args.dirname+"/res/" + str(args.k) (the /res/ folder has been made already)

    args = parser.parse_args()
    run(args)
