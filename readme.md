<div align="center">

# Cue RAG: Dynamic multi-output cue memory under H framework for Retrieval-augmented Generation

[![lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![transformers](https://img.shields.io/badge/Transformers-orange)](https://github.com/huggingface/transformers)
 <a href="https://github.com/pytorch/fairseq/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
</div>

How to find high-quality knowledge related to the input is crucial for retrieval-enhanced generation of RAG. The most relevant knowledge matched directly from the vector library is often the shallow knowledge that the model has already mastered. This paper introduces the concept of cue memory to help the model pay more attention to the underlying knowledge that matches the input. And through rigorous theoretical derivation, it is proved that cue memory is conducive to improving the answer accuracy of the model. At the same time, this paper proposes a new retrieval-enhanced generation framework-H framework. This framework incorporates the concept of cue memory, uses a global dependency encoder to obtain relevant contextual knowledge and feeds it into the information filter. The filter only retains the underlying knowledge that has not yet been learned to obtain cue memory. The cue memory is then output as a high-quality memory through a sequential decoder. Finally, an infinite memory pool is built, and all enhanced memories will be fed back to the pool. The quality of the memory pool continues to improve with continuous feedback. This paper has achieved good results on the public dialogue dataset DailyDialog, the translation dataset JRC-Acquis and private data. In particular, in the dialogue task, our H model achieved the best B-1/2 result, and the large model with cue memory also achieved SOTA results in the noise test. The relevant project code of this paper is publicly published at: https://github.com/Futr1/H-frame-work

<div align=center>
<img src=model.svg width=75% height=75% />
</div>

---

## Setup
Our code is mainly based on ⚡ [PyTorch Lightning]() and 🤗 [Transformers](https://github.com/huggingface/transformers). 

## Quick links

* [Environment](#Environment)
* [Cue memory](#Cue-memory)
* [H framework](#H-framework)
* [Licence](#Licence)

### Environment
PyTorch version >= 1.10.0
Python version >= 3.8
For training new models, we did  training on 2*A100+2*A6000.

```bash
conda create -n cue memory  python=3.10.0
conda activate cue memory
bash env.sh
```

### Cue memory
This repository contains the source code for this paper.The core of this paper is **cue memory**, which is used to extract information from the context that is not directly related but crucial. It primarily involves removing information already mastered by the model and retaining indirectly relevant knowledge. This project will provide a detailed explanation of its implementation steps, using a translation task as an example.

The script uses Fairseq and other tools to complete data preparation, model training, and evaluation for machine translation (MT) tasks. To proceed, navigate to the directory /mnt/7t/fys/fu_dir/pytorch-CycleGAN-and-pix2pix-master/selfmemory/SelfMemory/fairseq-main/fairseq-main/examples/discriminative_reranking_nmt. Below is a detailed step-by-step explanation of the script:


#### 1.Initial Definition

```
**SOURCE_FILE**: Source language text file (`source.txt`)  
**TARGET_FILE**: Target language text file (`test.txt`)  
**HYPO_FILE**: File storing model-generated hypotheses (`output.txt`)  
**XLMR_DIR**: Directory path containing the SentencePiece model and other resources  
**OUTPUT_DIR**: Directory to save processed output files and training results  
**SPLIT**: Specifies the dataset split to process (`train`, `test`, `valid`)  
**N**: Number of beam search hypotheses to generate, which must match the multiple of the target language text and hypothesis files  
**NUM_SHARDS**: Number of data shards for parallel processing  
**METRIC**: Evaluation metric, default is set to `"bleu"`
```

#### 2. Data preparation
The data will be saved in /mnt/7t/fys/fu_dir/pytorch-CycleGAN-and-pix2pix-master/selfmemory/SelfMemory/fairseq-main/fairseq-main/examples/discriminative_reranking_nmt/out2
Preprocess the data for each split by running the scripts/prep_data.py script. This script receives the source, target, and hypothesis files and generates the inputs required for training and evaluation.
```
python scripts/prep_data.py \
    --input-source ${SOURCE_FILE} \
    --input-target ${TARGET_FILE} \
    --input-hypo ${HYPO_FILE} \
    --output-dir ${OUTPUT_DIR} \
    --split $SPLIT \
    --beam $N \
    --sentencepiece-model ${XLMR_DIR}/sentencepiece.bpe.model \
    --num-shards ${NUM_SHARDS}
```
Run for each data split (train, test, valid) separately. That is, run three times, the training, validation and test data sets in the out2/blue/splite{N} path

#### 3.  Data preprocessing
Convert the data to Fairseq's binary format for fast loading, using the dict.txt file as a shared dictionary for both the source and target languages.
```
for suffix in src tgt ; do
    fairseq-preprocess --only-source \
        --trainpref ${OUTPUT_DIR}/$METRIC/split1/input_${suffix}/train.bpe \
        --validpref ${OUTPUT_DIR}/$METRIC/split1/input_${suffix}/valid.bpe \
        --destdir ${OUTPUT_DIR}/$METRIC/split1/input_${suffix} \
        --workers 60 \
        --srcdict ${XLMR_DIR}/dict.txt
```

Create a symbolic link for the shard data, that is, link your creation folder to the program's folder, so you don't have to modify the path. For other shards (starting from split2), create a symbolic link pointing to the split1 verification file to save space and time:
```
for i in `seq 2 ${NUM_SHARDS}`; do
    for suffix in src tgt ; do
        fairseq-preprocess --only-source \
            --trainpref ${OUTPUT_DIR}/$METRIC/split${i}/input_${suffix}/train.bpe \
            --destdir ${OUTPUT_DIR}/$METRIC/split${i}/input_${suffix} \
            --workers 60 \
            --srcdict ${XLMR_DIR}/dict.txt

        ln -s ${OUTPUT_DIR}/$METRIC/split1/input_${suffix}/valid* ${OUTPUT_DIR}/$METRIC/split${i}/input_${suffix}/.
    done

    ln -s ${OUTPUT_DIR}/$METRIC/split1/$METRIC/valid* ${OUTPUT_DIR}/$METRIC/split${i}/$METRIC/.
done
```

#### 4. Training the model using Fairseq
The fairseq-hydra-train command is used to start model training and configure distributed training on multiple GPUs
This configuration points to the model data directory, pre-trained model files, and other distributed training options. The training data is saved in /mnt/7t/fys/fu_dir/pytorch-CycleGAN-and-pix2pix-master/selfmemory/SelfMemory/fairseq-main/fairseq-main/examples/discriminative_reranking_nmt/multirun/2024-11-10, 2024-11-10 is your running time, and it will be saved according to the date
```
fairseq-hydra-train -m \
    --config-dir config/ --config-name deen \
    task.data=${OUTPUT_DIR}/$METRIC/split1/ \
    task.num_data_splits=${NUM_SHARDS} \
    model.pretrained_model=${XLMR_DIR}/model.pt \
    common.user_dir=${FAIRSEQ_ROOT}/examples/discriminative_reranking_nmt \
    checkpoint.save_dir=${EXP_DIR} \
    distributed_training.distributed_world_size=4
```

### H framework

Process the data
```
python create_features_data.py
python build_table.py
```
Train the model
```
python train_mslr.py
```



