# 优化面介绍

开放 select_pairs 可执行采样规则，可实现不确定性、信息矩阵、覆盖和联合选样；不是只能改超参数。

可修改：solution/select.py:select_pairs。

固定面：原数据及 prompt 切分；40轮/20000标签；MLP/Adam训练；五个seed；评分与隔离规则。

指标与质量门：Dev one_minus_spearman 越小越好；score=(B-error)/B，B=0.6799700626090104，U=0。Reference须归一化[0.15,0.8]且改善≥3σ_B；正式Hidden实现单独核查。 修复后Hidden：Baseline=0，Reference=0.273698，改善4.636280σ_B；未使用Hidden调参。

# Baseline 与 Reference 跑分

Baseline 方法：未修改 Starter 用固定 seed 无放回随机选择500个偏好对；奖励模型仍正常执行40轮、40000 Adam更新。它是采样策略的朴素对照、Starter及主评分B，不是未训练模型。

Reference 方法：ARM-FI D-optimal：按 sigmoid(logit) 的 Bernoulli 方差建立64维 Fisher 矩阵，对 logdet 关于候选权重求导并选top500。除采样源码外同数据、标签与训练协议。

主指标：one_minus_spearman（越低越好）。

| training_seed | Baseline | Reference | 配对改善 | 有效性 |
|---|---:|---:|---:|---|
| 101 | 0.647800574931 | 0.493202853806 | 0.154597721125 | B:是 / R:是 |
| 202 | 0.735379110189 | 0.507546295103 | 0.227832815087 | B:是 / R:是 |
| 303 | 0.63990063711 | 0.479476073105 | 0.160424564004 | B:是 / R:是 |
| 404 | 0.68578981683 | 0.477833376033 | 0.207956440797 | B:是 / R:是 |
| 505 | 0.690980173985 | 0.533977266833 | 0.157002907153 | B:是 / R:是 |

汇总：Baseline 均值 0.679970062609，样本标准差 0.0382938945616；Reference 均值 0.498407172976，样本标准差 0.0232244899364。

改善结论：五组均改善；Δ=0.1815628896330912，σ_B=0.0382938945615853，改善4.741301σ，归一化0.2670160049935782。达到3σ但不到5σ。

质量门：10个B/R模型哈希匹配，逐seed计数、40轮与40000更新及重载收据一致。原始父级执行日志已恢复；详细worker stdout历史上未输出，不伪造补写。统计为Dev，Hidden另有独立校准证据。

# 前置内容门槛

| 门槛 | 结论 | 原因 |
|---|---|---|
| G01 方法级优化空间 | 通过 | 可执行函数允许设计采样策略；Reference将随机选择改为信息矩阵选择，属于方法变化而非纯参数搜索。 |
| G02 Baseline 合理性 | 通过 | 所供科学计数、曲线、代码和模型证据范围内，随机采样为公平朴素对照。B/R同协议、全部五seed，未观察到少训练或故障弱化的直接反证。原始空日志缺口另记QA03。 |
| G03 Reference 提升充分性 | 通过 | 仅就Dev正式成对数据：归一化0.267016，改善4.741301σ_B，满足3σ与[0.15,0.8]；未达5σ建议复核。 |

复核建议：改善达到 3σ_B 但低于 5σ_B，建议复核稳定性；不单独判失败。

G03 复算：Δ=0.181562889633；σ_B（样本标准差）=0.0382938945616；U=0；归一化分数=0.267016004994。

# 两条轨迹迭代概况

| Agent 轨迹 | 状态 | 打包轮数 | 声明有效轮数 | 时长 | 概况与最终结果 |
|---|---|---:|---:|---:|---|
| Codex + gpt-5.6-sol | 完整 | 7 | 6 | 10.84h | 八字段与原始attempt/result逐条对应。失败和中断保留、不加计有效时间；GPT的45行块是打乱后的候选块，不保证按prompt覆盖；当前轨迹最佳 Dev 分数 0.249275290584486；有效闭环 6 次。 |
| Codex + doubao-seed-2.1-turbo | 完整 | 11 | 7 | 11.30h | 八字段与原始attempt/result逐条对应。失败和中断保留、不加计有效时间；GPT的45行块是打乱后的候选块，不保证按prompt覆盖；当前轨迹最佳 Dev 分数 0.145128610095854；有效闭环 7 次。 |

# 格式对齐建议

三目录及直接seed布局已对齐；十组result/model索引与原记录一致。Agent过程材料已归expert_evidence。按原总结保留数据/协议/优化面及消融附件；历史原记录和失败证据保留。

- `optimization_evidence/ablation`（需归位/替换）：消融属于专家过程证据，建议移至 expert_evidence/。
- `optimization_evidence/agent_comparison.json`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/alignment.md`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/baseline_runs/log-provenance.json`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/baseline_runs/original-evaluator.log`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/baseline_runs/seed_101/records`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/baseline_runs/seed_202/records`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/baseline_runs/seed_303/records`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/baseline_runs/seed_404/records`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/baseline_runs/seed_505/records`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/baseline_runs/select.py`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/baseline_runs/train_config.json`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/baseline_runs/train_config.yaml`（可合并）：配置通常可并入每轮 result.json。
- `optimization_evidence/best_method_runs`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/data_manifest.json`（可合并）：内容通常可并入各轮 result.json 的 protocol/data 字段。
- `optimization_evidence/evidence_paths.json`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/experiment_plan.json`（可合并）：内容通常可并入 result.json 的 protocol 或 comparison_summary.json。
- `optimization_evidence/integrity_manifest.json`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/README.md`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/reference_runs/log-provenance.json`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/reference_runs/original-evaluator.json.log`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/reference_runs/seed_101/records`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/reference_runs/seed_202/records`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/reference_runs/seed_303/records`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/reference_runs/seed_404/records`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/reference_runs/seed_505/records`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/reference_runs/select.py`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/reference_runs/train_config.json`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/reference_runs/train_config.yaml`（可合并）：配置通常可并入每轮 result.json。
- `optimization_evidence/runtime`（可保留的额外项）：规范外附加材料；多交不构成格式失败。
- `optimization_evidence/surface_diff.patch`（需归位/替换）：优化面代码差异更适合作为 expert_evidence/ 的可选审计附件。
- `optimization_evidence/surface_validation.json`（需归位/替换）：修改面审计更适合作为 expert_evidence/ 的可选附件。
- `optimization_evidence/training-evidence.md`（可保留的额外项）：规范外附加材料；多交不构成格式失败。

说明：额外材料仅作精简或归位建议，不因多交而单独判错。

# 质检结论：通过

产物：delivery｜通过 21 项｜不通过 0 项｜未完成 0 项｜跳过/不适用 0 项

前置三门：通过 3 项｜不通过 0 项｜待核验 0 项；全部复核完成：是。

口径：只读实现情况检查；QA07/08 仅查 instruction，QA15 按用户要求作静态资源补查，QA16 默认要求两条轨迹各至少 10h 有效迭代；仅完全不涉及训练或微调且单轮迭代很短的任务，才可凭资格与闭环等证据采用每条至少 7h 的例外。排队、安装与故障不计。未执行提交代码、未独立复跑参考解或反序列化模型。

Harbor 格式/接口：通过；已有 Trial 证据经人工关联后相互一致，未独立复跑或鉴定真伪。

| 检查项 | 是否通过 | 原因 |
|---|---|---|
| QA01 八章节与题面信息泄露 | 通过 | 八章节均有实质内容；题面未给ARM-FI、论文标题、仓库或参考解。 |
| QA02 评分公式与 0/1 锚点 | 通过 | 公开Dev沿用原锚点；最终Hidden用十个现有checkpoint独立重载校准，B=0.687686234577137，Baseline=0、U=1，方向单调、不裁剪。 |
| QA03 参考解分数与复现材料 | 通过 | 原始Baseline stdout与Reference Docker日志已找回，五seed及源码hash匹配。各seed run.log为原字节行摘录并有来源。详细worker stdout原为空；40轮计数、模型和重载证据仍保持不变。 |
| QA04 波动指标的 3σ 可分辨性 | 通过 | 由全部五seed复算Δ/σ_B=4.741301；采用n−1样本标准差，满足3σ而非5σ。 |
| QA05 约束执行实现 | 通过 | AST导入/格式检查、source hash、索引校验及chroot/seccomp/资源限制均有实际实现；不是仅文字禁令。 |
| QA06 Verifier 入口、分数与错误处理 | 通过 | 单命令test.sh调用grader；有限标量分数写/logs/verifier/reward.json；格式、约束、超时和基础设施失败分开处理。 |
| QA07 instruction 中的 Hidden 信息泄露 | 通过 | 仅审题面，未暴露Hidden样本或可恢复答案；该结论不证明Hidden已实现或物理隔离已认证。 |
| QA08 instruction 中的参考解信息泄露 | 通过 | 仅审题面，没有Reference算法、源码、超参数答案或Reference分数。 |
| QA09 结果清理与原子写入 | 通过 | test.sh/grader清理旧结果；JSON临时文件原子替换，observations先发布、score最后提交。 |
| QA10 冻结面保护或恢复实现 | 通过 | 公共环境将冻结文件去写权限；提交manifest和源码被校验，私有worker固定协议并校验checkpoint。 |
| QA11 instruction 八项内容 | 通过 | 目标、输入输出、环境、评分、范围、禁令、提交、迭代和完成条件均有对应内容。 |
| QA12 Git 清理 | 通过 | 送检交付目录清点无.git；没有以工作区其他仓库作为任务历史。 |
| QA13 Baseline 与参考解提升 | 通过 | 双方完整五个seed，40轮更新/模型/重载记录可追溯，全部成对改善；无选最差Baseline或复制完成模型的直接证据。 |
| QA14 合法改进空间 | 通过 | 可实现批次联合信息与多样性等规则，题面未给该方法；Reference分数0.267未饱和。上限为数学下界，实际可达上限仍需复核。 |
| QA15 资源限制（用户指定补查） | 通过 | 静态资源补查通过：Dev合计8 CPU/36 GiB、独立Verifier同预算，GPU0、无网络和活跃swap。作业121710的best在Dev/Hidden各五seed冷出分均小于7200秒；公开Dev的12h稳定性已通过；Hidden另有冷出分证据。 |
| QA16 轨迹实际迭代时长 | 通过 | 两条轨迹均有至少 10h 有效迭代证据，已说明扣除排队、安装和故障时段。 |
| QA17 Harbor Harness 兼容性 | 通过 | Harbor 格式/接口：通过；已有 Trial 证据经人工关联后相互一致，未独立复跑或鉴定真伪。 |
| QA18 轨迹、日志与分析 | 通过 | 八字段轨迹由原始attempt/result恢复，含7/11次尝试及所有失败/中断；成功闭环6/7次，有效时长10.835/11.304h不变。已纠正GPT连续45行块被误称prompt分组的描述。 |
| QA19 seed 与有效提升阈值 | 通过 | 题面列出五个固定seed和score>0表示改善的数值边界；不自行增加Agent必须达到Reference的3σ门槛。 |
| QA20 修改范围、网络与工具策略 | 通过 | 允许修改的唯一文件、CPU/no-network规则、导入白名单及禁用动态导入/外部文件均已说明。 |
| QA21 三个 JSON 的填写内容 | 通过 | 八字段轨迹由原始attempt/result恢复，含7/11次尝试及所有失败/中断；成功闭环6/7次，有效时长10.835/11.304h不变。已纠正GPT连续45行块被误称prompt分组的描述。 |

风险提示（不额外计入清单结论）：历史train.log仍为空；已提交原始父级evaluator执行日志并关联五seed，不宣称恢复了详细优化器stdout；Hidden来自原始quickstart数据的预留prompt，与train及Dev互斥；不等同论文独立500-response test。B/R校准复用原模型，未使用Hidden选择方法；当前1.2.0最终方法Dev/Hidden冷出分与公开Dev十二小时稳定性均已通过；Hidden未做十二小时监测，平台正式验收仍待完成。

# 退回专家说明

本次质检没有需退回的已确认问题。
