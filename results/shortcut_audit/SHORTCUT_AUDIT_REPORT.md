# UWF-ZeekData24 Shortcut Audit Report

运行命令：

```powershell
python scripts\audit_uwf_shortcuts.py configs\local_uwf_smoke.yaml
```

## 1. 目的

本报告用于诊断当前 UWF-ZeekData24 本机子集中的 feature-label shortcut。该诊断借鉴 BiasSeeker 的思想：先用模型无关的统计相关性发现数据集特异性捷径，再决定哪些特征需要单独报告、消融或作为环境干预维度。

## 2. 主要发现

| feature | NMI(label) | Cramer's V(label) |
|---|---:|---:|
| proto | 0.8573 | 0.9513 |
| service | 0.5458 | 0.9491 |
| dest_port_zeek | 0.4751 | 0.9640 |
| history | 0.4367 | 0.9542 |
| total_bytes | 0.3691 | 0.8922 |
| total_pkts | 0.3637 | 0.9348 |
| orig_bytes | 0.3618 | 0.8885 |
| resp_pkts | 0.3405 | 0.8288 |
| src_ip_zeek | 0.3220 | 0.9953 |

## 3. 解释

当前子集里协议、服务、端口、流量体积和 IP 与标签强相关。这说明随机 split 下的高 F1 不能直接解释为 detector 理解了攻击语义。IntervenDB 应把这些字段视为 potential shortcut / environment-entangled feature。

## 4. 后续处理

1. 报告 feature profile：`full`、`env_sensitive`、`protocol_core`。
2. 增加 source-IP split 和 time split。
3. 在论文里把 shortcut audit 作为 IntervenDB 的诊断模块，而不是隐藏数据问题。
4. 后续接入 Gotham/DataSense 验证该现象是否跨数据集存在。


