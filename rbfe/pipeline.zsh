#!/bin/zsh
# RBFE pipeline: Ligand relative binding free energy calculations
# Based on feprest/pipeline.zsh, adapted for ligand FEP
# Key difference: REST2 hot region = ligand atoms (not protein mutation site)

local reqstate
local stateno
reqstate=$1
stateno=$2
if [[ -z $stateno ]]; then
    echo "This file should be called via controller.zsh" 1>&2
    exit 1
fi
shift
shift

#-------- subroutines for production run (shared with feprest)
mdrun_find_possible_np() {
    least_unit=$1
    shift
    args=($@)
    log_basename="" || true
    is_log=0
    multidir=()
    for arg in $args; do
        if [[ $is_log = 1 ]]; then
            log_basename=$arg
        elif [[ $is_multidir = 1 ]]; then
            multidir+=$arg
        fi
        case $arg in
            -deffnm|-l)
                is_log=1
                ;;
            -multidir)
                is_multidir=1
                ;;
            -*)
                is_multidir=0
                is_log=0
                ;;
            *)
                ;;
        esac
    done
    case $log_basename in
        *.log)
            log_basename=${log_basename%%.log}
            ;;
    esac
    if [[ -z $multidir ]]; then
        multidir=(.)
    fi
    if [[ -n $PROCS_SAVED ]]; then
        NP=$PROCS_SAVED
    else
        NP=$PROCS
    fi
    ntomp=()
    if [[ -z $OMP_NUM_THREADS ]] || (( OMP_NUM_THREADS == 1 )); then
        ntomp=(-ntomp 1)
    fi
    while true; do
        echo "Trying with NP=$NP"
        job_mpirun $NP $GMX_MPI mdrun $args $NSTLIST_CMD $ntomp
        if [[ $? != 0 ]]; then
            domain_error=0
            for d in $multidir; do
                if tail -20 "$d/$log_basename.log" | grep -qi -e domain -e prime; then
                    domain_error=1
                fi
            done
            if (( domain_error == 0 )); then
                echo "Error: domain-unrelated error"
                exit $ERRORCODE
            fi
            PREVNP=$NP
            (( NP = (NP - 1) / least_unit * least_unit ))
            if (( NP == 0 )) || (( PREVNP == NP )); then
                echo "Error: unable to find proper parallel processes"
                exit $ERRORCODE
            fi
        else
            break
        fi
    done
    PROCS_SAVED=$NP
    echo "Normal termination of mdrun"
}

do_bar ()
{
    ID=$1
    TEMP=$2
    mkdir $ID/bar || true
    $PYTHON3 $RBFE_ROOT/../feprest/bar_deltae.py --xvgs $ID/prodrun/rep%sim/deltae.xvg  --nsim $NREP --temp $TEMP --save-dir $ID/bar | tee $ID/bar1.log || echo "BAR failed due to bad convergence, please continue the run to get it fixed"
}

parallelizable_singlerun ()
{
    logfile=$1
    shift

    if [[ $RUN_TPR_PARALLEL = yes ]]; then
        ( job_singlerun $@ >& $logfile && echo "Exit_success" >> $logfile )&
    else
        job_singlerun $@
    fi
}

wait_if_needed ()
{
    workdir=$1
    logfile=$2
    errmsg=$3

    if [[ $RUN_TPR_PARALLEL = yes ]]; then
        wait
        for i in {0..$((NREP - 1))}; do
            mrundir=$workdir/rep$i
            if [[ ! -e $mrundir/$logfile ]] || [[ $(tail -1 $mrundir/$logfile) != Exit_success ]]; then
                echo "$errmsg" 1>&2
                false
            fi
        done
    fi
}

initref() {
    REFCMDINIT=()
    if [[ -n $REFINIT ]]; then
        REFCMDINIT=(-r $ID/$REFINIT)
    fi
    REFCMD=()
    if [[ -n $REFCRD ]]; then
        REFCMD=(-r $ID/$REFCRD)
    fi
}

main() {
    initref

    # Use FEPREST_ROOT for shared tools (bar_deltae.py, rest2py, etc.)
    FEPREST=${FEPREST:-$RBFE_ROOT/../feprest}

    case $reqstate,$stateno in
        query,all)
            echo {1..8}
            ;;
        query,1)
            echo "DEPENDS=(); (( PPM = PARA )); MULTI=1"
            ;;
        run,1)
            # EM for state A
            sed -e "/%LAMBDA%/d;/%VDWLAMBDA%/d;/%STATE%/d;/free-energy/c free-energy = no" mdp/cginit.mdp > $ID/minA.mdp
            sed -e "/integrator/c integrator = steep" $ID/minA.mdp > $ID/steepA.mdp
            if ! grep -q POSRES $ID/$BASETOP ; then
                sed -e "s/-DPOSRES/ /" -i $ID/minA.mdp
                sed -e "s/-DPOSRES/ /" -i $ID/steepA.mdp
            fi
            job_singlerun $GMX grompp -f $ID/steepA.mdp -c $ID/$BASECONF -p $ID/$BASETOP -o $ID/steepA -po $ID/steepA.mdout -maxwarn $((BASEWARN+0)) $REFCMDINIT
            mdrun_find_possible_np 1 -deffnm $ID/steepA
            job_singlerun $GMX grompp -f $ID/minA.mdp -c $ID/steepA.gro -p $ID/$BASETOP -o $ID/minA -po $ID/minA.mdout -maxwarn $((BASEWARN+0)) $REFCMDINIT
            mdrun_find_possible_np 1 -deffnm $ID/minA
            ;;
        query,2)
            echo "DEPENDS=(1); (( PPM = PARA )); MULTI=1"
            ;;
        run,2)
            # NVT run
            sed -e "/%LAMBDA%/d;/%VDWLAMBDA%/d;/%STATE%/d;/free-energy/c free-energy = no" mdp/nvtinit.mdp > $ID/nvtA.mdp
            if ! grep -q POSRES $ID/$BASETOP ; then
                sed -e "s/-DPOSRES/ /" -i $ID/nvtA.mdp
            fi
            $PYTHON3 $FEPREST/turn-heavy.py -p $ID/$BASETOP -o $ID/heavy.top
            job_singlerun $GMX grompp -f $ID/nvtA.mdp -c $ID/minA.gro -p $ID/heavy.top -o $ID/nvtA -po $ID/nvtA.mdout -maxwarn $((BASEWARN+1)) $REFCMDINIT
            mdrun_find_possible_np 1 -deffnm $ID/nvtA
            ;;
        query,3)
            echo "DEPENDS=(2); (( PPM = PARA )); MULTI=1"
            ;;
        run,3)
            # NPT run
            cp mdp/nptinit.mdp $ID/nptA.mdp
            job_singlerun $GMX grompp -f $ID/nptA.mdp -c $ID/nvtA.gro -p $ID/heavy.top -o $ID/nptA -po $ID/nptA.mdout -maxwarn $((BASEWARN+1)) -pp $ID/fep_pp.top $REFCMD
            mdrun_find_possible_np 1 -deffnm $ID/nptA
            ;;
        query,4)
            echo "DEPENDS=(3); (( PPM = PARA )); MULTI=1"
            ;;
        run,4)
            # REST2 setup
            # For RBFE: hot region is the ligand atoms (not protein mutation site)
            # Use add_underline.py with --target-molecule to select ligand atoms,
            # or directly mark ligand atoms as hot.
            # The merged ligand ITP already has state A/B annotations (perturbed atoms),
            # so add_underline.py should detect them correctly via charge/type differences.
            $PYTHON3 $FEPREST/add_underline.py -c $ID/$BASECONF -t $ID/fep_pp.top -o $ID/fep_underlined.top --distance $REST2_REGION_DISTANCE --ignore-perturbing-multiple-molecules
            prev=$ID/nptA
            top=$ID/fep_underlined.top
            if [[ $CHARGE != no ]]; then
                $PYTHON3 $FEPREST/neutralize.py --topology $ID/fep_underlined.top --gro $ID/nptA.gro --output-topology $ID/fep_underlined_neut.top --output-gro $ID/nptA_neut.gro --mode $CHARGE --ff $FF
                prev=$ID/nptA_neut
                top=$ID/fep_underlined_neut.top
            fi
            $PYTHON3 $FEPREST/underlined_group.py -t $top -o $ID/for_rest.ndx
            $PYTHON3 $FEPREST/rest2py/replica_optimizer.py init $NREP feprest --basedir $ID --temp $REST2_TEMP
            mkdir $ID/genmdps || true
            $PYTHON3 $FEPREST/rest2py/replica_optimizer.py update-mdp mdp/cg.mdp $ID/genmdps/cg%d.mdp --basedir $ID --temp $REST2_TEMP
            mkdir $ID/gentops || true
            $PYTHON3 $FEPREST/rest2py/replica_optimizer.py update-topology $top $ID/gentops/fep_%d.top --basedir $ID --temp $REST2_TEMP
            ln -s $FEPREST/itp_addenda/*.itp $ID || true
            ln -s $PWD/$ID/*.itp gentops || true
            ln -s $PWD/*.itp $ID || true
            for i in {0..$((NREP - 1))}; do
                work=$ID/min$i
                mkdir $work || true
                sed -e "s/cg/steep/;/nsteps/s/5000/500/;" $ID/genmdps/cg$i.mdp > $work/steep$i.mdp
                $PYTHON3 $FEPREST/recover-water.py -p $ID/gentops/fep_$i.top -o $ID/gentops/fep_tip3p_${i}_light.top --ff $FF
                $PYTHON3 $FEPREST/turn-heavy.py -p $ID/gentops/fep_tip3p_${i}_light.top -o $ID/gentops/fep_tip3p_$i.top
                job_singlerun $GMX grompp -f $work/steep$i.mdp -c $prev -p $ID/gentops/fep_tip3p_$i.top -o $work/steep$i -po $work/steep.mdout.$i -maxwarn $((BASEWARN+1)) $REFCMD
                mdrun_find_possible_np 1 -deffnm $work/steep$i -rdd $DOMAIN_SHRINK
                job_singlerun $GMX grompp -f $ID/genmdps/cg$i.mdp -c $work/steep$i.gro -p $ID/gentops/fep_tip3p_$i.top -o $work/min$i -po $work/min.mdout.$i -maxwarn $((BASEWARN+1)) $REFCMD
                mdrun_find_possible_np 1 -deffnm $work/min$i -rdd $DOMAIN_SHRINK
                prev=$work/min$i
            done
            ;;
        query,5)
            echo "DEPENDS=(4); (( PPM = PARA )); (( MULTI = NREP ))"
            ;;
        run,5)
            # Tune replex (same as feprest)
            top=$ID/fep_underlined.top
            if [[ $CHARGE != no ]]; then
                top=$ID/fep_underlined_neut.top
            fi
            prevgro=()
            for i in {0..$((NREP-1))}; do
                prevgro+=$ID/min$i/min$i.gro
            done
            NINITIALTUNE=${NTUNE:-0} || true
            for p in {1..$NTUNE}; do
                work=$ID/nvt$p
                mkdir $work || true
                $PYTHON3 $FEPREST/rest2py/replica_optimizer.py update-mdp mdp/nvt.mdp $work/nvt${p}_%d.mdp --basedir $ID --temp $REST2_TEMP
                $PYTHON3 $FEPREST/rest2py/replica_optimizer.py update-topology $top $work/fep_%d.top --basedir $ID --temp $REST2_TEMP
                reps=()
                for i in {0..$((NREP - 1))}; do
                    mrundir=$work/rep$i
                    mkdir $mrundir || true
                    reps+=$mrundir
                    $PYTHON3 $FEPREST/recover-water.py -p $work/fep_$i.top -o $work/fep_tip3p_${i}_light.top --ff $FF
                    $PYTHON3 $FEPREST/turn-heavy.py -p $work/fep_tip3p_${i}_light.top -o $work/fep_tip3p_$i.top
                    { echo "energygrps = hot"; echo "userint1 = 1" } >> $work/nvt${p}_$i.mdp
                    parallelizable_singlerun $mrundir/grompp.log $GMX grompp -f $work/nvt${p}_$i.mdp -c ${prevgro[$((i+1))]} -p $work/fep_tip3p_$i.top -o $mrundir/nvt -po $mrundir/nvt.mdout -maxwarn $((BASEWARN+1)) $REFCMD -n $ID/for_rest.ndx
                done
                wait_if_needed $work grompp.log "Error: failed to grompp on tuning cycle $p replica $i"
                mdrun_find_possible_np $NREP -deffnm nvt -multidir $reps -rdd $DOMAIN_SHRINK
                REPOPT_EXTEND_RUN_LENGTH=${EXTEND_RUN_LENGTH:-50}
                for i in {0..$((NREP - 1))}; do
                    mrundir=$work/rep$i
                    parallelizable_singlerun $mrundir/extend_run.log $GMX convert-tpr -s $mrundir/nvt -o $mrundir/nvt_c -extend $REPOPT_EXTEND_RUN_LENGTH
                done
                wait_if_needed $work extend_run.log "Error: failed to run convert-tpr on tuning cycle $p replica $i" 1>&2
                REPOPT_REPLEX_INTERVAL=${REPOPT_REPLEX_INTERVAL:-100}
                local -a nstlista
                nstlista=()
                if [[ -n $REPOPT_REPLEX_INTERVAL ]] && (( REPOPT_REPLEX_INTERVAL <= 50 )); then
                    nstlista=(-nstlist $REPOPT_REPLEX_INTERVAL)
                fi
                mdrun_find_possible_np $NREP -deffnm nvt -multidir $reps -s nvt_c -cpi nvt -hrex -replex $REPOPT_REPLEX_INTERVAL $nstlista -rdd $DOMAIN_SHRINK -bonded cpu
                (( STEPCOUNT = p - NINITIALTUNE )) || true
                if (( STEPCOUNT < 1 )); then
                    (( STEPCOUNT = 1 ))
                fi
                $PYTHON3 $FEPREST/rest2py/replica_optimizer.py optimize $work/rep0/nvt.log --basedir $ID --step $STEPCOUNT --temp $REST2_TEMP
                cat $ID/replica_states
                prev=$work
                prevgro=()
                for i in {0..$((NREP-1))}; do
                    prevgro+=${reps[$((i+1))]}/nvt.gro
                done
            done
            cp $prev/fep_tip3p_{0..$((NREP-1))}.top $ID/gentops/
            ;;
        query,6)
            echo "DEPENDS=(5); (( PPM = PARA )); (( MULTI = NREP ))"
            ;;
        run,6)
            # NPT run
            work=$ID/npt
            mkdir $work || true
            $PYTHON3 $FEPREST/rest2py/replica_optimizer.py update-mdp mdp/npt.mdp $work/npt%d.mdp --basedir $ID --temp $REST2_TEMP
            reps=()
            for i in {0..$((NREP - 1))}; do
                mrundir=$work/rep$i
                mkdir $mrundir || true
                reps+=$mrundir
                parallelizable_singlerun $mrundir/npt_grompp.log $GMX grompp -f $work/npt$i.mdp -c $ID/nvt${NTUNE}/rep$i/nvt.gro -t $ID/nvt${NTUNE}/rep$i/nvt.cpt -p $ID/gentops/fep_tip3p_$i.top -o $mrundir/npt -po $mrundir/npt.mdout -maxwarn $((BASEWARN+1)) $REFCMD
            done
            wait_if_needed $work npt_grompp.log "Error: failed to grompp on final npt run"
            mdrun_find_possible_np $NREP -deffnm npt -multidir $reps -rdd $DOMAIN_SHRINK
            ;;
        query,7)
            echo "DEPENDS=(6); (( PPM = PARA )); (( MULTI = NREP ))"
            ;;
        run,7)
            # Production run initialization
            mkdir $ID/run.mdout || true
            mkdir $ID/runmdps || true
            $PYTHON3 $FEPREST/rest2py/replica_optimizer.py update-mdp mdp/run.mdp $ID/runmdps/run%d.mdp --basedir $ID --temp $REST2_TEMP
            work=$ID/prodrun
            mkdir $work || true
            reps=()
            for i in {0..$((NREP - 1))}; do
                mrundir=$work/rep$i
                mkdir $mrundir || true
                reps+=$mrundir
                { echo "energygrps = hot"; echo "userint1 = 1" } >> $ID/runmdps/run$i.mdp
                parallelizable_singlerun $mrundir/prodrun_grompp.log $GMX grompp -f $ID/runmdps/run$i.mdp -c $ID/npt/rep$i/npt.gro -t $ID/npt/rep$i/npt.cpt -p $ID/gentops/fep_tip3p_$i.top -o $mrundir/prodrun -po $mrundir/run.mdout -maxwarn $((BASEWARN+1)) $REFCMD -n $ID/for_rest.ndx
                [[ -e $mrundir/deltae.xvg ]] && mv $mrundir/deltae.xvg $mrundir/deltae.xvg.bak
            done
            wait_if_needed $work prodrun_grompp.log "Error: failed to grompp on final production run"
            cpi_args=()
            if [[ -e ${reps[1]}/prodrun.cpt ]]; then
                cpi_args=(-cpi prodrun -cpt 60)
                echo "Resuming from checkpoint files"
            fi
            mdrun_find_possible_np $NREP -deffnm prodrun -multidir $reps -rdd $DOMAIN_SHRINK $cpi_args
            mkdir $ID/checkpoint_7 || true
            for d in $reps; do
                mkdir -p $ID/checkpoint_7/$d || true
                cp $d/prodrun.cpt $ID/checkpoint_7/$d
                cp $d/prodrun.tpr $d/prodrun_ph0.tpr
            done
            ;;
        query,999)
            echo "DEPENDS=(); PPM=1; MULTI=1"
            ;;
        run,999)
            for state in A B; do
                case $state in
                A)
                    REPNO=0 || true
                    ;;
                B)
                    REPNO=$((NREP - 1))
                    ;;
                esac
                ndxfile=$ID/prodrun/fepbase_$state.ndx
                if [[ ! -e $ndxfile ]]; then
                    $PYTHON3 $FEPREST/make_ndx_trjconv_analysis.py -i $ID/fepbase_$state.pdb -o $ndxfile
                fi
                sourcefile=$ID/prodrun/rep$REPNO/prodrun.trr
                destfile=$ID/prodrun/state$state.xtc
                if [[ ! -e $sourcefile ]]; then
                    break
                fi
                if [[ -e $destfile ]] && [[ $destfile -nt $sourcefile ]]; then
                    continue
                fi
                echo "centering\noutput" | job_singlerun $GMX trjconv -s $ID/prodrun/rep$REPNO/prodrun.tpr -f $sourcefile -o $destfile -pbc atom -ur compact -center -n $ndxfile
            done
            ;;
        query,*)
            (( PREV = stateno - 1 ))
            echo "DEPENDS=($PREV); (( PPM = PARA )); (( MULTI = NREP ))"
            ;;
        run,*)
            # Extend production run
            (( PHASE = stateno - 7 )) || true
            TEMP=$(grep '^\s*ref[_|-]t' mdp/run.mdp | cut -d '=' -f2 | cut -d ';' -f1)
            work=$ID/prodrun
            reps=()
            for i in {0..$((NREP - 1))}; do
                mrundir=$work/rep$i
                mkdir $mrundir || true
                reps+=$mrundir
                parallelizable_singlerun $mrundir/extend$PHASE.log $GMX convert-tpr -s $mrundir/prodrun_ph$((PHASE-1)) -o $mrundir/prodrun_ph$PHASE -extend $SIMLENGTH
            done
            wait_if_needed $work extend$PHASE.log "Error: failed to extend on final production run"
            mdrun_find_possible_np $NREP -deffnm prodrun -s prodrun_ph$PHASE -cpi prodrun -cpt 60 -hrex -othersim deltae -othersiminterval $SAMPLING_INTERVAL -multidir $reps -replex $REPLICA_INTERVAL -rdd $DOMAIN_SHRINK -bonded cpu

            mkdir $ID/checkpoint_$stateno || true
            for d in $reps; do
                mkdir -p $ID/checkpoint_$stateno/$d || true
                cp $d/prodrun.cpt $ID/checkpoint_$stateno/$d
            done
            do_bar $ID $TEMP
        ;;
    esac
}

main
