# 离线分卷包

`task-and-evidence.zip` 是代码和证据主包，包含正式模型、日志和质检报告。
大体积原始数据、离线依赖和历史运行中间产物按哈希去重后放在 `assets/`。
每个分卷小于 256 MiB。数据只做可逆的字节重排和 gzip 压缩，不量化、采样或删减。

保留本目录全部文件，在本目录运行：

```sh
python3 -B restore.py --output ../delivery-restored
```

还原仅需 Python 标准库，不联网、不训练。目标目录必须尚不存在。
程序检查分卷和全部还原文件的 SHA-256、大小与权限，生成完整的
`workspace/`、`expert_evidence/`、`optimization_evidence/` 三目录。
重复资产还原为独立文件，保留各 Docker 构建上下文。

主包单独不是完整 HarborTask。上传或构建时使用还原后的完整目录；
平台是否接受分卷传输须按其入口规则办理。本方案不改变完整任务所需的数据量。
