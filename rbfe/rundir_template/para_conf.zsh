# Put this file at either the parent directory or its child.
# The file on the parent directory is sourced by the pipeline, then the file on the child directory is sourced, so the child one will overwrite the variables in the parent.

# initial structure and topology
BASECONF=conf_ionized.pdb
BASETOP=topol_ionized.top

# Allowed values are "auto", "no", "posonly"
CHARGE=auto

# Force field types to use.
FF=amber

# Number of replicas to use. Recommended: 32-36 for ligand FEP
NREP=32

# Number of MPI processes per replica. For GPU, recommended = 1.
PARA=1

# Number of threads per process. Recommended: for CPU: 1, for GPU: 2-6.
TPP=4

# Number of tuning cycles for fep parameters
NTUNE=5

# Simulation time of one chunk in ps, recommended total time is 4 ns
SIMLENGTH=4000

# replica exchange interval, recommended: 1000
REPLICA_INTERVAL=1000

# Energy sampling interval, recommended: 100
SAMPLING_INTERVAL=100

# Increase if you are using strange topology file
BASEWARN=1

# reference coordinate upon initialization
REFINIT=$BASECONF
# Set if you want extra restraints during FEP
REFCRD=

# Domain decomposition shrink factor for dummy atoms
DOMAIN_SHRINK=0.6

# REST2 configurations
# For RBFE: distance from perturbed ligand atoms to include in hot region
REST2_REGION_DISTANCE=0.4

# REST2 hot-region temperature
REST2_TEMP=1200
