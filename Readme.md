# Documentation for running the code

```
<MODEL-DIR>: directory to save model checkpoints
<FEATURES-DIR>: CLEVR images features (Resnet-101) stored in h5py format
<CLEVR-DIR>: directory of CLEVR dataset
```

## Prepare data for training
Step 1: Download the CLEVR dataset using the link https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip and copy it in `<CLEVR-DIR>`

Step-2: Run `python image_feature_resnet.py <CLEVR-DIR>` to create Resnet-101 features for CLEVR images in h5py format

## Train a model on CLEVR dataset
```
python train.py --clevr_dir <CLEVR-DIR> --model_dir <MODEL-DIR> --map_dim 384 --max_time_stamps 5 --max_stack_len 5 --batch_size 64 --features_dir <FEATURES-DIR> --reg_coeff_op_loss 1e0
```

## Train a model on CLEVR Humans dataset (fine-tuning from saved checkpoint of model trained on CLEVR)
```
python train_clevr_humans.py --clevr_dir <CLEVR-DIR> --model_dir <MODEL-DIR> --map_dim 384 --max_time_stamps 5 --max_stack_len 5 --batch_size 64 --features_dir <FEATURES-DIR> --ckpt <PATH-TO-CHECKPOINT-FILE> --reg_coeff_op_loss 1e0
```

## Evaluate on validation set

### soft operation evaluation
```
python test.py --max_time_stamps 5 --max_stack_len 5 --map_dim 384 --batch_size 64 --model_dir <MODEL-DIR> --ckpt <PATH-TO-CHECKPOINT-FILE> --clevr_dir <CLEVR-DIR>
```

### hard operation (argmax) evaluation
```
python test.py --max_time_stamps 5 --max_stack_len 5 --map_dim 384 --batch_size 64 --model_dir <MODEL-DIR> --ckpt <PATH-TO-CHECKPOINT-FILE> --clevr_dir <CLEVR-DIR> --use_argmax
```

## Evaluate Integrated Gradient (IG) scores
```
python compute_module_ig.py --max_time_stamps 5 --max_stack_len 5 --map_dim 384 --batch_size 64 --model_dir <MODEL-DIR> --ckpt <PATH-TO-CHECKPOINT-FILE> --clevr_dir <CLEVR-DIR> --use_argmax
```

## visualize operation weights (path to checkpoint file is hard-coded inside the script)
```
python visualize_op_weights.py
```

## dump test predictions for CLEVR test set
```
python dump_test_ans.py --max_time_stamps 5 --max_stack_len 5 --map_dim 384 --batch_size 64 --model_dir <MODEL-DIR> --ckpt <PATH-TO-CHECKPOINT-FILE> --clevr_dir <CLEVR-DIR>
```

## dump test predictions for CLEVR Humans test set
```
python dump_test_ans_clevr_humans.py --max_time_stamps 5 --max_stack_len 5 --map_dim 384 --batch_size 64 --model_dir <MODEL-DIR> --ckpt <PATH-TO-CHECKPOINT-FILE> --clevr_dir <CLEVR-DIR>
```
