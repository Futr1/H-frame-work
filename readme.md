<div align="center">

# Cue RAG: Dynamic multi-output cue memory under H framework for Retrieval-augmented Generation

[![lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![transformers](https://img.shields.io/badge/Transformers-orange)](https://github.com/huggingface/transformers)
 <a href="https://github.com/pytorch/fairseq/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
</div>

Nowadays, LLMs still face hallucinations problem. To this problem, Retrieval-augmented generation (RAG) transfers
non-parametric knowledge to LLMs and has been proven to be an effective means to enhance the accuracy of model
answers. Traditional RAG is effective to support LLMs to find the most relevant surface answer from the vector
library, but often does not work well on deeper reasons related to the input. This paper defines the concept of 
**cued memory**, which removes text with high confidence (even if it is highly relevant) after finding the answer related to the
LLMs input. This helps the LLMs to explore the underlying relevant knowledge that match the input. In this paper,
through the exploration of cued memory, we also propose a new **H-framework** RAG: obtain contextual relationships
in the form of a global dependency encoder, use an information filter to delete the direct information that the model
has already obtained and retain the underlying knowledge that has not been learned, and obtain out into cued memory.
Then the cued memory is sent to the sequential dependency encoder, which can achieve dynamic truncation and
dynamic output of multiple high-quality memories. To the end, by creating an infinite memory pool, all the enhanced
text can be fed back to the memory pool, so as to improves the efficiency of memory pool. Our method out performs
on the public dialogue dataset DailyDialog and the translation dataset JRC-Acquis as well as the locally built private
question-answering dataset. In particular, in the dialogue task, our H framework achieved the best B-1/2 result, and
LLMs with cue memory also achieved **SOTA results** in the noise test. The project code and data related to this article
are publicly published at: https://github.com/Futr1/H-frame-work

<div align=center>
<img src=model.svg width=75% height=75% />
</div>

---

## Setup
Our code is mainly based on ⚡ [PyTorch Lightning]() and 🤗 [Transformers](https://github.com/huggingface/transformers). 

## Quick links

* [Environment](#Environment)
* [Retrieval-Augmented Generation Benchmark](#Retrieval-Augmented)
* [Evaluation](#Evaluation)
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

### cue memory
This repository contains the source code for this paper
本文的核心为cue memory，用于提取上下文中无直接关联但至关重要的信息，主要涉及到去除原本已经模型掌握的信息和保留间接相关的知识，本项目将对其实现步骤进行详细说明（以翻译任务举例）
脚本使用 Fairseq 及其他工具完成机器翻译（MT）任务中的数据准备、模型训练和评估。要进入目录是/mnt/7t/fys/fu_dir/pytorch-CycleGAN-and-pix2pix-master/selfmemory/SelfMemory/fairseq-main/fairseq-main/examples/discriminative_reranking_nmt，以下是脚本的详细分步说明

#### 1.定义文件路径和变量
SOURCE_FILE：源语言文本文件（source.txt）。
TARGET_FILE：目标语言文本文件（test.txt）。
HYPO_FILE：存储模型预测的假设文件（output.txt）。
XLMR_DIR：包含 SentencePiece 模型和其他资源的目录路径。
OUTPUT_DIR：保存处理后的输出文件和训练结果的目录。
SPLIT：指定要处理的数据集分割（train、test、valid）。
N：生成的 beam search 假设数量，也就是目标语言文本和预测假设文件对应的倍数，一定要一致。
NUM_SHARDS：数据分片的数量，用于并行处理。
METRIC：评估指标，默认设置为 "bleu"。

#### 2. 使用 prep_data.py 进行数据准备，数据会保存在/mnt/7t/fys/fu_dir/pytorch-CycleGAN-and-pix2pix-master/selfmemory/SelfMemory/fairseq-main/fairseq-main/examples/discriminative_reranking_nmt/out2中
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

####3. Fairseq 数据预处理
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

####4. 为分片数据创建符号链接，也就是你的创建文件夹链接到程序的文件夹，就不用修改路径了
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

####5. 使用 Fairseq 训练模型
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

The data is putted in `data/`

```text
data/
├── en.json
├── en_refine.json
├── en_int.json
├── en_fact.json
├── zh.json
├── zh_refine.json
├── zh_int.json
└── zh_fact.json
```

To evalute the Information Integration, you should use `zh_int` or `en_int` for Chinese questions or English questions. 

To evalute the Counterfactual Robustness, you should use `zh_fact` or `en_fact` for Chinese questions or English questions. 

#### The refined data

We refine the retrieved documents and some answers of `en.json` and `zh.json`, and name the new data files as `en_refine.json` and `zh_refine.json`:

+ Removing incorrect positive and negative documents

+ Adding some positive documents.

+ Correcting some inaccurate answers.

### Evaluation

For evaluating ChatGPT, you can run as:

```bash
python evalue.py \
--dataset en \
--modelname chatgpt \
--temp 0.2 \
--noise_rate 0.6 \
--api_key YourAPIKEY \
--passage_num 5
```

For evaluating other models, you can run as:

```bash
python evalue.py \
--dataset en \
--modelname chatglm2-6b \
--temp 0.2 \
--noise_rate 0.6 \
--plm THUDM/chatglm-6b \
--passage_num 5
```

You should change `modelname` and `plm` for different models, where `plm` is the path of model.

`temp` is the temperature of model.

`noise_rate` is rate of noisy documents in inputs.

`passage_num` is number of provided documents for LLM (default is 5).

The outputs are:

+ all_rate: The accuracy (noise_rate<1) or rejection rate (noise_rate=1)
+ fact_check_rate: the error detection rates (ED)

---

To evaluate rejection using ChatGPT, you should first run the `evalue.py` in noise_rate=1 to obtain the generation result, and then run:

```bash
python reject_evalue.py \
--dataset en \
--modelname chatglm2-6b \
--api_key YourAPIKEY
```

The "reject_rate" in the outputs are the reject rate (Rej\*).

---

To evaluate counterfactual robustness using ChatGPT, you should first run the `evalue.py` in dataset=en_fact/zh_fact to obtain the generation result, and then run:

```bash
python fact_evalue.py \
--dataset en_fact \
--modelname chatglm2-6b \
--api_key YourAPIKEY
```

The "reject_rate" in the outputs are the error detection rates (ED\*). The `correct_rate` in the outputs are the error correction rate (CR)

## License

The code and data are released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License for Noncommercial use only. Any commercial use should get formal permission first.

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg


