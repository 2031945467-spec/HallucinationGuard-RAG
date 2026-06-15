# Contributing

欢迎提交问题、改进评测器、增加检索方法或扩展测试集。

## 开发流程

1. Fork 本仓库并创建功能分支。
2. 保持修改范围清晰，不提交 API 密钥、虚拟环境或模型文件。
3. 运行以下检查：

```bash
python -m py_compile src/config.py src/retriever.py scripts/run_and_report.py
python scripts/run_and_report.py --evaluate-only
```

4. 在 Pull Request 中说明修改目的、实验设置和指标变化。

## 实验结果要求

- 不使用虚构数据。
- 记录模型、阈值、数据集版本与运行环境。
- 同时报告幻觉率、正确率和覆盖率。
- 保留逐题输出，便于复核汇总指标。
