#!/home/6/uc02086/local/bin/zsh

# to debug the submission script uncomment the next line
# set -x

# Root directory of FEP-suite
FEPSUITE_ROOT=$HOME/work/fepsuite

# Root GROMACS directory
GROMACS_DIR=$HOME/opt/gromacs-2020-hrex-gpu

# Specify jobtype: rbfe for ligand relative binding free energy
JOBTYPE=rbfe

# Specify jobsystem
JOBSYSTEM=none
# If jobsystem need additional information fill this variable
#SUBSYSTEM=

# Run actual controller
source $FEPSUITE_ROOT/controller.zsh $0 $@
