<!-- README_SYNC: source=984ce2dbea6d8ce12f83532881a4ab4c90dd595e; updated=2026-11-18 -->

[English](README.md) · 简体中文

# Active Preference Acquisition

这是一项只用 CPU 的 Harbor 研究任务：每轮从 22,500 个候选偏好对里挑出 500 个来标注，看能否提升最终奖励模型对回答的排序质量。你只实现选择函数；数据、标注预算、模型和训练流程固定。

[![GitHub Stars](https://img.shields.io/github/stars/LikC1606/active-preference-acquisition)](https://github.com/LikC1606/active-preference-acquisition/stargazers)
[![Complete release](https://img.shields.io/badge/offline_package-complete_submission-2563eb)](https://github.com/LikC1606/active-preference-acquisition/releases/tag/complete-submission)

可以给仓库点个 Star，方便回查选择器接口和对比结果。

任务使用原始 Active Reward Modeling quickstart 数据。随机 Starter 是 **Baseline**，归一化得分为 **0**。论文的 **ARM-FI D-optimal 方法**作为 **Reference**，用于展示该协议下可达到的改进幅度；它的分数不设为上限或通过线。

## 修改范围

编辑 [workspace/harbor_task/solution/select.py](workspace/harbor_task/solution/select.py)：

```python
def select_pairs(features, logits, k, seed, round_index):
    # Return k distinct candidate indices as a list of Python integers.
    ...
```

每次调用提供 22,500 个候选对，包括 64 维特征差和预测奖励差，需返回 500 个不同的索引。每次调用都在新的隔离进程中运行；不能跨轮保存状态。候选行已经打乱，连续的行不代表同一个提示词。

每个种子有 40 轮采集、20,000 条标注和 40,000 次 Adam 更新。第 0 轮固定随机采样，第 1 至 39 轮使用你写的选择器。评测种子为 `101`、`202`、`303`、`404`、`505`。完整接口、允许导入的库及提交约束见[任务说明](workspace/harbor_task/instruction.md)。

## 评分和已记录结果

回答排序误差为 1 减去模型排序与 oracle 排序的平均 Spearman 相关系数，越低越好。最终分数用五个种子的平均误差与随机 Baseline 比较：

```text
score = (B - mean_error) / B
Public Dev B = 0.6799700626090104
```

正分表示优于随机，负分表示更差，分数连续且不截断。隐藏评测使用同一公式，不过基线在独立 Hidden 划分上计算。

| 方法 | 选择规则 | 已记录 Dev 分数 |
| --- | --- | ---: |
| Baseline | 原始随机 Starter | 0.000000 |
| Reference | 论文的 ARM-FI D-optimal 规则 | 0.267016 |
| 最优已记录 Agent 选择器 | Fisher 加权协方差杠杆值 | 0.249275 |

这些是本任务的实验结果，与论文基准分开统计。该最终选择器在本地 Hidden 冷跑中还得到 `0.244336`。[分种子的结果与对比](optimization_evidence/comparison_summary.json)保存在仓库中。

## 下载完整任务

运行任务前，先取得[完整离线发布包](https://github.com/LikC1606/active-preference-acquisition/releases/tag/complete-submission)。Git 仓库包含代码、说明和证据索引；大型数据、离线依赖包和模型检查点在发布附件中。将两个文件下载到同一目录：

- [active-preference-acquisition.tar.gz.001](https://github.com/LikC1606/active-preference-acquisition/releases/download/complete-submission/active-preference-acquisition.tar.gz.001)
- [active-preference-acquisition.tar.gz.002](https://github.com/LikC1606/active-preference-acquisition/releases/download/complete-submission/active-preference-acquisition.tar.gz.002)

压缩包约 **3.69 GB**，为了满足 GitHub 单附件 2 GiB 的限制拆成两份。先合并，再核对 SHA-256 后解包：

```sh
cat active-preference-acquisition.tar.gz.001 active-preference-acquisition.tar.gz.002 > active-preference-acquisition.tar.gz
sha256sum active-preference-acquisition.tar.gz
mkdir delivery-restored
tar -xzf active-preference-acquisition.tar.gz -C delivery-restored
cd delivery-restored
```

预期 SHA-256：

```text
db2f9d65ab8722eb29f11a9f29a549e947157399e893b6a00914aa43c41c7531
```

压缩包恢复 4,008 个文件，包括数据、离线依赖、模型和实验记录；发布时已验证文件哈希与权限。各附件的独立校验值见 [DISTRIBUTION.md](DISTRIBUTION.md)。

## 本地运行

需要 **Docker Compose v2**、**8 个 CPU** 和 **36 GiB 容器内存**，不需要 GPU。一次完整的五种子评估限时两小时，只允许同时跑一次。

依赖由随包的 wheels 安装。离线构建前，确保本地已有公共基础镜像 `python:3.11.11-slim-bookworm`。在解包后的根目录执行：

```sh
docker compose -p arm-acquisition \
  -f workspace/harbor_task/environment/docker-compose.yaml build --no-cache

docker compose -p arm-acquisition \
  -f workspace/harbor_task/environment/docker-compose.yaml up -d --no-build
```

修改解包目录中的 `workspace/harbor_task/solution/select.py`。容器初始运行的是镜像内的 Starter；评分前要把改好的文件复制到运行中的 `main` 服务：

```sh
docker compose -p arm-acquisition \
  -f workspace/harbor_task/environment/docker-compose.yaml \
  cp workspace/harbor_task/solution/select.py main:/harbor_task/solution/select.py

docker compose -p arm-acquisition \
  -f workspace/harbor_task/environment/docker-compose.yaml \
  exec -T main sh -c 'cd /harbor_task && ./solution/solve.sh --output solution/artifacts/current'
```

在公共容器中，`solution/artifacts/current/reward.json` 保存分数，`observations.json` 保存逐种子指标。失败时查 `failure.json` 的原因与建议。超时属于失败，不按零分计。

[本地运行手册](optimization_evidence/runtime/LOCAL_PACKAGE.md)给出了接口检查、已有结果复核、容器清理和 Harbor 集成步骤。Harbor 0.22.0 的本地验证使用手册中的离线 provider 适配器。

## 任务与证据目录

```text
workspace/
  harbor_task/          Task instructions, Starter, environments and tests
  reference/            Paper ARM-FI implementation
expert_evidence/        Expert notes, independent GPT/Seed trajectories and final selector
optimization_evidence/  Per-seed runs, checkpoints, comparisons and runtime evidence
```

Agent 只使用 `workspace/harbor_task/`。Reference、专家证据和私有评测材料不得放进 Agent 工作区。公共 Dev 环境和独立的 Hidden 验证器分别构建；验证器收到提交的选择器源码后独立计算分数。

## 验证记录和限制

- 本地质检通过 21 项检查，包括单独要求补查的资源限制；三项内容门槛和六项 Harbor 检查也通过。
- 最终选择器完成了五种子 Dev 和 Hidden 冷跑，分别耗时 **5,196.04 秒**和 **4,769.36 秒**，均在 7,200 秒预算内。
- 公共 Dev 双容器连续监测 **12 小时**，Baseline 和最终选择器各 681 次检查，无报错、内存耗尽或重启。监测复用了已验证模型，未重新训练，也未覆盖 Hidden。

任务使用原始约 10 GB quickstart 数据的预计算嵌入，每个提示词评估十个回答。实验没有复跑论文独立的 500 回答测试；协议与限制详见[实验计划](optimization_evidence/experiment_plan.json)。

**正式平台验收仍未完成。** 本地检查和 Hidden 冷跑属于本地记录。

[质检报告](optimization_evidence/runtime/qa-review/report.txt) · [专家说明](expert_evidence/专家作业说明文档.md) · [证据索引](optimization_evidence/README.md)

仓库根目录没有项目级许可证。复用前请查阅原始数据与随包依赖的来源和授权。
