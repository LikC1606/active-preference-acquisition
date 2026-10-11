# Baseline 合理性检查

总评：在所供科学证据范围内合理；原始父级evaluator日志已回收，详细worker stdout未输出。静态审查不等于独立复现。

角色：Starter=随机采样；正式Baseline=该采样方法经过完整训练的结果；主评分B=其五seed平均误差。

| 项目 | 状态 | 事实 |
| --- | --- | --- |
| B01 | 通过 | 随机采样是本题选择策略的可解释朴素对照；未要求奖励预测随机。 |
| B02 | 通过 | 10组accounting/曲线到40轮与40000更新；模型哈希/重载匹配，空日志单记QA03。 |
| B03 | 通过 | 共享协议和数据、相同标签/训练预算，仅采样策略不同。 |
| B04 | 通过 | 训练超参数来自上游train_BT默认；B/R固定共用，不是单独给B异常参数。 |
| B05 | 通过 | 101/202/303/404/505完整成对，checkpoint互异，没有直接挑seed反证。 |
| B06 | 通过 | 正式B就是随机采样Starter同协议训练结果；包内未见可比健康对照反证。 |
| B07 | 通过 | 差异为D-optimal采样；固定模型、原始reward概率和评价，未观察额外数据收益。 |
| B08 | 通过 | 作业说明、alignment、协议记录数据/默认参数来源和论文复现限制。 |

直接不公平红旗：未发现。原始train.log为空，但已回收非空父级evaluator日志及来源；不宣称有详细优化器stdout。

证据：
- optimization_evidence/comparison_summary.json
- optimization_evidence/experiment_plan.json
- optimization_evidence/runtime/native/worker.py
- optimization_evidence/runtime/native/paper/bradleyterry.py
- optimization_evidence/runtime/native/paper/rules_pre_paired.py
- workspace/harbor_task/solution/select.py
- workspace/reference/select.py
- optimization_evidence/alignment.md
- optimization_evidence/baseline_runs/seed_101/records/evaluation.json
- optimization_evidence/baseline_runs/seed_202/records/evaluation.json
- optimization_evidence/baseline_runs/seed_303/records/evaluation.json
- optimization_evidence/baseline_runs/seed_404/records/evaluation.json
- optimization_evidence/baseline_runs/seed_505/records/evaluation.json
- optimization_evidence/reference_runs/seed_101/records/evaluation.json
- optimization_evidence/reference_runs/seed_202/records/evaluation.json
- optimization_evidence/reference_runs/seed_303/records/evaluation.json
- optimization_evidence/reference_runs/seed_404/records/evaluation.json
- optimization_evidence/reference_runs/seed_505/records/evaluation.json
