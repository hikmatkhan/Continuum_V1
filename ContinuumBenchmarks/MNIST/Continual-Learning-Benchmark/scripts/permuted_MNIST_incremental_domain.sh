GPUID=0
OUTDIR=outputs/permuted_MNIST_incremental_domain
REPEAT=3
N_PERMUTATION=10
mkdir -p $OUTDIR
IBATCH_LEARN=/home/hikmat/Desktop/GlobXAI/ContinuumBench/MNIST/Continual-Learning-Benchmark
# IBATCH_LEARN_LAMBDA = /data/hikmat/GitHub/ContinuumBenchmarks/CIFAR10/Continual-Learning-Benchmark
EPOCHS=10
BATCH_SIZE=128 #128

python -u $IBATCH_LEARN/iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation $N_PERMUTATION --no_class_remap --force_out_dim 10 --schedule $EPOCHS --batch_size $BATCH_SIZE --model_name MLP1000                                                     --lr 0.0001  --offline_training  | tee ${OUTDIR}/Offline.log
python -u $IBATCH_LEARN/iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation $N_PERMUTATION --no_class_remap --force_out_dim 10 --schedule $EPOCHS --batch_size $BATCH_SIZE --model_name MLP1000                                                     --lr 0.0001                      | tee ${OUTDIR}/Adam.log
python -u $IBATCH_LEARN/iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer SGD     --n_permutation $N_PERMUTATION --no_class_remap --force_out_dim 10 --schedule $EPOCHS --batch_size $BATCH_SIZE --model_name MLP1000                                                     --lr 0.001                       | tee ${OUTDIR}/SGD.log
python -u $IBATCH_LEARN/iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adagrad --n_permutation $N_PERMUTATION --no_class_remap --force_out_dim 10 --schedule $EPOCHS --batch_size $BATCH_SIZE --model_name MLP1000                                                     --lr 0.001                       | tee ${OUTDIR}/Adagrad.log
python -u $IBATCH_LEARN/iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation $N_PERMUTATION --no_class_remap --force_out_dim 10 --schedule $EPOCHS --batch_size $BATCH_SIZE --model_name MLP1000 --agent_type customization  --agent_name EWC_online --lr 0.0001 --reg_coef 250       | tee ${OUTDIR}/EWC_online.log
python -u $IBATCH_LEARN/iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation $N_PERMUTATION --no_class_remap --force_out_dim 10 --schedule $EPOCHS --batch_size $BATCH_SIZE --model_name MLP1000 --agent_type customization  --agent_name EWC        --lr 0.0001 --reg_coef 150       | tee ${OUTDIR}/EWC.log
python -u $IBATCH_LEARN/iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation $N_PERMUTATION --no_class_remap --force_out_dim 10 --schedule $EPOCHS --batch_size $BATCH_SIZE --model_name MLP1000 --agent_type regularization --agent_name SI         --lr 0.0001 --reg_coef 10        | tee ${OUTDIR}/SI.log
python -u $IBATCH_LEARN/iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation $N_PERMUTATION --no_class_remap --force_out_dim 10 --schedule $EPOCHS --batch_size $BATCH_SIZE --model_name MLP1000 --agent_type regularization --agent_name L2         --lr 0.0001 --reg_coef 0.02      | tee ${OUTDIR}/L2.log
python -u $IBATCH_LEARN/iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation $N_PERMUTATION --no_class_remap --force_out_dim 10 --schedule $EPOCHS --batch_size $BATCH_SIZE --model_name MLP1000 --agent_type customization  --agent_name Naive_Rehearsal_4000   --lr 0.0001          | tee ${OUTDIR}/Naive_Rehearsal_4000.log
python -u $IBATCH_LEARN/iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation $N_PERMUTATION --no_class_remap --force_out_dim 10 --schedule $EPOCHS --batch_size $BATCH_SIZE --model_name MLP1000 --agent_type customization  --agent_name Naive_Rehearsal_16000  --lr 0.0001          | tee ${OUTDIR}/Naive_Rehearsal_16000.log
python -u $IBATCH_LEARN/iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation $N_PERMUTATION --no_class_remap --force_out_dim 10 --schedule $EPOCHS --batch_size $BATCH_SIZE --model_name MLP1000 --agent_type regularization --agent_name MAS        --lr 0.0001 --reg_coef 0.1       | tee ${OUTDIR}/MAS.log
python -u $IBATCH_LEARN/iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer SGD     --n_permutation $N_PERMUTATION --no_class_remap --force_out_dim 10 --schedule $EPOCHS --batch_size $BATCH_SIZE --model_name MLP1000 --agent_type customization  --agent_name GEM_4000   --lr 0.1 --reg_coef 0.5          | tee ${OUTDIR}/GEM_4000.log
python -u $IBATCH_LEARN/iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer SGD     --n_permutation $N_PERMUTATION --no_class_remap --force_out_dim 10 --schedule $EPOCHS --batch_size $BATCH_SIZE --model_name MLP1000 --agent_type customization  --agent_name GEM_16000  --lr 0.1 --reg_coef 0.5          | tee ${OUTDIR}/GEM_16000.log