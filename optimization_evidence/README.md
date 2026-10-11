# 证据目录

`baseline_runs/seed_*` 和 `reference_runs/seed_*` 各包含五个正式种子。
每组的 `result.json` 汇总方法、协议、训练、执行、指标和产物；`run.log`
保留原始执行记录，`model/` 保存 checkpoint、产物清单和历史重载收据。
原始配置、逐轮指标、计数和日志放在各组的 `records/`，内容未经改写。

`comparison_summary.json` 列出全部配对值、统计量和源结果路径。
`训练证据说明.md` 说明训练及日志限制。历史未记录的命令、时间和退出码
在结果索引中保留为 null；重载差值也不补造。`model/reload.log` 是原始
结构化重载收据的副本，不表示这次进行了重载。

Agent 过程材料位于 `../expert_evidence/agent-research/`，两条正式轨迹
仍在 `expert_evidence/` 根目录。按原总结保留的数据清单、实验计划、优化面
验证和单优化点消融说明，是 Baseline/Reference 的附加证据。

`runtime/` 保存构建、评分、稳定性和失败记录。历史记录使用的旧路径见
`evidence_paths.json`；模型和原始记录的内容哈希保持不变。当前质检报告
位于 `runtime/qa-review/`。任务代码、评分协议及训练结果未因整理而改变。
