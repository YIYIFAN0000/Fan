# ColAMP-WGAN：胶原来源潜在抗菌肽生成模型

这个项目已按你提供的结构整理为完整流程。该结构可以用于完整模型，因为它覆盖了候选肽生成任务所需的关键环节：

1. 原始 AMP 数据与胶原 FASTA 输入。
2. AMP 正样本清洗、羊胶原窗口构建、负样本构建。
3. AMP 活性预测器与 toxicity 预测器。
4. 带 AMP 奖励、毒性惩罚、胶原 motif 约束的 WGAN-GP。
5. 生成候选肽、规则过滤、预测器过滤、相似性过滤和综合排序。

注意：当前自带的 `data/raw/*.csv` 和 `data/raw/sheep_collagen.fasta` 是为了让代码在 PyCharm 中直接跑通的示例数据。正式研究时请替换为 DRAMP、DBAASP、APD、CAMP 与真实羊皮胶原序列。

## 目录结构

```text
data/
  raw/                         # dramp.csv, dbaasp.csv, apd3.csv, camp4.csv, sheep_collagen.fasta
  processed/                   # 自动生成的训练数据、中间文件
configs/
  colamp_gan.yaml              # 主配置文件
src/
  data/                        # 数据清洗、AMP 数据集、胶原窗口、编码
  features/                    # 肽性质、胶原 motif、相似性
  models/                      # generator, critic, AMP/toxicity predictor
  training/                    # predictor 与 GAN 训练
  inference/                   # 生成、过滤、排序
  utils/                       # FASTA、序列、配置工具
outputs/
  generated/
  filtered/
  ranked/
  checkpoints/
  predictors/
train_predictors.py
train_gan.py
generate.py
rank.py
```

## 环境安装

建议在 PyCharm 中新建 Python 3.10 或 3.11 虚拟环境，然后在项目根目录执行：

```bash
pip install -r requirements.txt
```

## ESM-2 特征输入

当前主配置已经启用 ESM-2：

```yaml
esm:
  enabled: true
  model_name: facebook/esm2_t6_8M_UR50D
  local_files_only: true
  embedding_mode: token
```

训练 AMP/toxicity 预测器时，序列会先被转换为 ESM-2 特征并缓存到
`data/processed/esm2_cache/`。GAN 训练时，生成器输出的 soft amino-acid
分布会被投影到同一套 ESM-2 token embedding 空间，因此 AMP 奖励和毒性惩罚
可以对生成器保持可微。若想用完整上下文 ESM 表征做离线筛选实验，可将
`embedding_mode` 改为 `contextual` 后重新运行预测器训练。

如果你的 Windows/conda 环境出现 OpenMP 冲突，可以在 PyCharm Run Configuration 的 Environment variables 中加入：

```text
KMP_DUPLICATE_LIB_OK=TRUE
```

## 一键式运行顺序

在项目根目录依次运行：

```bash
python train_predictors.py --config configs/colamp_gan.yaml
python train_gan.py --config configs/colamp_gan.yaml
python generate.py --config configs/colamp_gan.yaml
python rank.py
```

快速验证可以使用 smoke 配置：

```bash
python train_gan.py --config configs/colamp_smoke.yaml
python generate.py --config configs/colamp_smoke.yaml
python rank.py --config configs/colamp_smoke.yaml
```

输出结果：

- `outputs/predictors/amp_predictor.pt`
- `outputs/predictors/toxicity_predictor.pt`
- `outputs/checkpoints/generator.pt`
- `outputs/generated/generated_peptides.csv`
- `outputs/filtered/filtered_candidates.csv`
- `outputs/ranked/ranked_candidates.csv`

## PyCharm 配置

新建 4 个 Python Run Configuration：

- Script: `train_predictors.py`; Parameters: `--config configs/colamp_gan.yaml`
- Script: `train_gan.py`; Parameters: `--config configs/colamp_gan.yaml`
- Script: `generate.py`; Parameters: `--config configs/colamp_gan.yaml`
- Script: `rank.py`; Parameters 留空

Working directory 必须设为当前项目根目录。

## 数据格式

AMP 数据 CSV 至少需要：

```csv
sequence,source,label
GLFDIVKKVVGALGSL,DRAMP,1
```

羊皮胶原数据使用 FASTA：

```text
>sheep_collagen
GPPGPPGPP...
```

## 模型架构

Generator:

- 输入随机噪声 `z`
- MLP 输出 `seq_len x 20` 氨基酸概率分布

Critic:

- 输入真实/生成 one-hot 或 soft one-hot 序列
- 输出 Wasserstein critic score

Predictors:

- AMP predictor: CNN 二分类器，输出 AMP 概率
- Toxicity predictor: CNN 二分类器，当前用启发式毒性标签训练，后续可替换为真实毒性/溶血数据

GAN 目标：

```text
generator_loss =
  - critic(fake)
  - amp_reward_weight * AMP_probability(fake)
  + toxicity_penalty_weight * toxicity_probability(fake)
  - collagen_reward_weight * collagen_motif_score(fake)
```

## 科研注意事项

生成结果只是潜在候选抗菌肽，不代表真实抗菌活性。正式使用时建议继续加入：

- CD-HIT/MMseqs2 去冗余。
- BLAST 同源性排查。
- Hemolysis/toxicity 真实数据训练。
- MIC 实验验证。
- 稳定性、溶解性和合成可行性评估。
