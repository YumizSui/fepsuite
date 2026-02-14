if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <FEP_ID> [NREP]"
    exit 1
fi
ID=$1
NREP=${2:-32}
FEPREST_ROOT=/gs/bs/tga-furui/workspace/fep/fepsuite/feprest
if [ ! -f mdp/run.mdp ]; then
    echo "Error: $ID/mdp/run.mdp not found!"
    exit 1
fi
TEMP=$(grep '^\s*ref[_|-]t' mdp/run.mdp | cut -d '=' -f2 | cut -d ';' -f1)

python $FEPREST_ROOT/bar_deltae.py --xvgs $ID/prodrun/rep%sim/deltae.xvg  --nsim $NREP --temp $TEMP --save-dir $ID/bar | tee $ID/bar1.log
