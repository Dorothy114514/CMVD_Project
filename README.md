# CMVD PET/MR 研究项目

当前完成数据准备，心功能与CMVD Demo尚待开发。

先阅读 [项目状态与交接](docs/HANDOFF.md)；AI协作规则见 [AGENTS.md](AGENTS.md)。

| 位置 | 用途 |
|---|---|
| data/ | 原始受保护影像，保持只读 |
| deidentified/ | 79例已处理研究影像，仍需像素及完整隐私审查 |
| metadata/full_case_manifest.csv | 病例索引与处理计数 |
| audit/ | 原有规则复核和排除记录 |
| audit/restricted/ | 含真实患者文件名的完整性审计 |
| scripts/data_preparation/ | 数据处理与复核脚本 |
| tools/7zr.exe | 本地解压工具 |
| docs/HANDOFF.md | 当前进展、局限、后续工作及精简日志 |

查看影像可用3D Slicer导入单病例目录，如 `deidentified/阳性/POS-001`。

使用已有Conda执行全量头字段复核（约25分钟，每10,000文件输出进度，完成后更新audit中的报告）：

```powershell
& 'D:\anaconda3\python.exe' .\scripts\data_preparation\validate_deidentification.py
```

路径以脚本位置确定，不依赖启动目录。去标识脚本对现有输出设有拒绝覆盖保护；重建需先制定独立输出方案。

后续按实际功能创建src/、app/、models/、configs/、outputs/，目前不建立空目录或重复病例表。
