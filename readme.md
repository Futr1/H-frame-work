<div align="center">

# Cue RAG: Dynamic multi-output cue memory under H framework for Retrieval-augmented Generation

[![lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![transformers](https://img.shields.io/badge/Transformers-orange)](https://github.com/huggingface/transformers)
 <a href="https://github.com/pytorch/fairseq/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
</div>


cue memory  This repository contains the source code for this paper
该脚本使用 Fairseq 及其他工具完成机器翻译（MT）任务中的数据准备、模型训练和评估。要进入目录是/mnt/7t/fys/fu_dir/pytorch-CycleGAN-and-pix2pix-master/selfmemory/SelfMemory/fairseq-main/fairseq-main/examples/discriminative_reranking_nmt，以下是脚本的详细分步说明
1.定义文件路径和变量
SOURCE_FILE：源语言文本文件（source.txt）。
TARGET_FILE：目标语言文本文件（test.txt）。
HYPO_FILE：存储模型预测的假设文件（output.txt）。
XLMR_DIR：包含 SentencePiece 模型和其他资源的目录路径。
OUTPUT_DIR：保存处理后的输出文件和训练结果的目录。
SPLIT：指定要处理的数据集分割（train、test、valid）。
N：生成的 beam search 假设数量，也就是目标语言文本和预测假设文件对应的倍数，一定要一致。
NUM_SHARDS：数据分片的数量，用于并行处理。
METRIC：评估指标，默认设置为 "bleu"。

2. 使用 prep_data.py 进行数据准备，数据会保存在/mnt/7t/fys/fu_dir/pytorch-CycleGAN-and-pix2pix-master/selfmemory/SelfMemory/fairseq-main/fairseq-main/examples/discriminative_reranking_nmt/out2中
通过运行 scripts/prep_data.py 脚本来预处理每个分割的数据。该脚本接收源、目标和假设文件，并生成训练和评估所需的输入。
python scripts/prep_data.py \
    --input-source ${SOURCE_FILE} \
    --input-target ${TARGET_FILE} \
    --input-hypo ${HYPO_FILE} \
    --output-dir ${OUTPUT_DIR} \
    --split $SPLIT \
    --beam $N \
    --sentencepiece-model ${XLMR_DIR}/sentencepiece.bpe.model \
    --num-shards ${NUM_SHARDS}
该脚本分别针对每个数据分割（train、test、valid）运行。也就是运行了三次，out2/blue/splite{N}路径下的训练、验证和测试数据集

3. Fairseq 数据预处理
针对 src（源语言）和 tgt（目标语言）使用 fairseq-preprocess 命令预处理数据，以便进行训练：
for suffix in src tgt ; do
    fairseq-preprocess --only-source \
        --trainpref ${OUTPUT_DIR}/$METRIC/split1/input_${suffix}/train.bpe \
        --validpref ${OUTPUT_DIR}/$METRIC/split1/input_${suffix}/valid.bpe \
        --destdir ${OUTPUT_DIR}/$METRIC/split1/input_${suffix} \
        --workers 60 \
        --srcdict ${XLMR_DIR}/dict.txt
done
此步骤将数据转换为 Fairseq 的二进制格式，便于快速加载，同时使用 dict.txt 文件作为源和目标语言的共享字典。

4. 为分片数据创建符号链接，也就是你的创建文件夹链接到程序的文件夹，就不用修改路径了
对于其他分片（从 split2 开始），创建指向 split1 验证文件的符号链接以节省空间和时间：
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

5. 使用 Fairseq 训练模型
fairseq-hydra-train 命令用于启动模型训练，并配置了多个 GPU 的分布式训练。
fairseq-hydra-train -m \
    --config-dir config/ --config-name deen \
    task.data=${OUTPUT_DIR}/$METRIC/split1/ \
    task.num_data_splits=${NUM_SHARDS} \
    model.pretrained_model=${XLMR_DIR}/model.pt \
    common.user_dir=${FAIRSEQ_ROOT}/examples/discriminative_reranking_nmt \
    checkpoint.save_dir=${EXP_DIR} \
    distributed_training.distributed_world_size=4
此配置指向模型数据目录、预训练模型文件以及其他分布式训练选项。训练数据保存在/mnt/7t/fys/fu_dir/pytorch-CycleGAN-and-pix2pix-master/selfmemory/SelfMemory/fairseq-main/fairseq-main/examples/discriminative_reranking_nmt/multirun/2024-11-10中，2024-11-10是你的运行时间，会根据日期进行保存

