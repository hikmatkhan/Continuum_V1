GPUID=0
OUTDIR=outputs/split_MNIST_incremental_domain
REPEAT=10
mkdir -p $OUTDIR

IBATCHLEARNPATH=/home/hikmat/Desktop/JWorkspace/CL/Continuum/ContinuumBenchmarks/MNIST/Continual-Learning-Benchmark

python -u ${IBATCHLEARNPATH}/iBatchLearn.py --outdir $OUTDIR --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400                                              --lr 0.001 --offline_training  | tee ${OUTDIR}/Offline.log
python -u ${IBATCHLEARNPATH}/iBatchLearn.py --outdir $OUTDIR --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400                                              --lr 0.001                     | tee ${OUTDIR}/Adam.log
python -u ${IBATCHLEARNPATH}/iBatchLearn.py --outdir $OUTDIR --gpuid $GPUID --repeat $REPEAT --optimizer SGD     --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400                                              --lr 0.01                      | tee ${OUTDIR}/SGD.log
python -u ${IBATCHLEARNPATH}/iBatchLearn.py --outdir $OUTDIR --gpuid $GPUID --repeat $REPEAT --optimizer Adagrad --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400                                              --lr 0.01                      | tee ${OUTDIR}/Adagrad.log
python -u ${IBATCHLEARNPATH}/iBatchLearn.py --outdir $OUTDIR --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type customization  --agent_name EWC_online_mnist --lr 0.001 --reg_coef 700    | tee ${OUTDIR}/EWC_online.log
python -u ${IBATCHLEARNPATH}/iBatchLearn.py --outdir $OUTDIR --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type customization  --agent_name EWC_mnist        --lr 0.001 --reg_coef 100    | tee ${OUTDIR}/EWC.log
python -u ${IBATCHLEARNPATH}/iBatchLearn.py --outdir $OUTDIR --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type regularization --agent_name SI  --lr 0.001 --reg_coef 3000     | tee ${OUTDIR}/SI.log
python -u ${IBATCHLEARNPATH}/iBatchLearn.py --outdir $OUTDIR --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type regularization --agent_name L2  --lr 0.001 --reg_coef 0.5      | tee ${OUTDIR}/L2.log
python -u ${IBATCHLEARNPATH}/iBatchLearn.py --outdir $OUTDIR --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type customization  --agent_name Naive_Rehearsal_1100  --lr 0.001   | tee ${OUTDIR}/Naive_Rehearsal_1100.log
python -u ${IBATCHLEARNPATH}/iBatchLearn.py --outdir $OUTDIR --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type customization  --agent_name Naive_Rehearsal_4400  --lr 0.001   | tee ${OUTDIR}/Naive_Rehearsal_4400.log
python -u ${IBATCHLEARNPATH}/iBatchLearn.py --outdir $OUTDIR --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type regularization --agent_name MAS --lr 0.001 --reg_coef 10000    | tee ${OUTDIR}/MAS.log
python -u ${IBATCHLEARNPATH}/iBatchLearn.py --outdir $OUTDIR --gpuid $GPUID --repeat $REPEAT --optimizer SGD     --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type customization  --agent_name GEM_1100 --lr 0.01  --reg_coef 0.5 | tee ${OUTDIR}/GEM_1100.log
python -u ${IBATCHLEARNPATH}/iBatchLearn.py --outdir $OUTDIR --gpuid $GPUID --repeat $REPEAT --optimizer SGD     --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type customization  --agent_name GEM_4400 --lr 0.01  --reg_coef 0.5 | tee ${OUTDIR}/GEM_4400.log