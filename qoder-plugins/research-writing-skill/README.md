# 科研写作助手（精选模块）

从 [Norman-bury/research-writing-skill](https://github.com/Norman-bury/research-writing-skill)
(v3.1.0) 精选四个与本论文项目（稀疏观测卫星 TE，投稿 TNSE）直接相关的技能模块，
打包为 Qoder 插件。

## 包含的技能

| 技能 | 用途 | 本项目对应场景 |
|---|---|---|
| `figures-diagram` | 生成流程图/架构图的生图 AI 提示词 | 论文 §IV 架构图（Fig. 1） |
| `figures-python` | 生成可复现的 Python 数据图脚本 | GPU 实验数据图（obs-ratio 扫描、消融、训练曲线） |
| `peer-review` | 投稿前自审检查 | TNSE 投稿前终稿检查 |
| `writing-core` | 通用写作规范与去 AI 化边界 | 英文正文润色（避免机械连接词/空壳句式，不压缩事实） |

## 有意省略的部分

源仓库共 18 个技能模块。以下未打包，原因是本项目已完成对应阶段或不适用：
头脑风暴、章节编排、文献综述、LaTeX 生成（论文工程已就位于 `paper/`）、
学科分流（社科/医学/法学）、环境安装、统计分析、旧版 modules 与多平台
配置目录（.claude-plugin/.cursor-plugin/.codex 等）。

`writing-core` 引用的 `style_check.ps1`（Windows）未打包，仅保留
`style_check.sh`，且 SKILL.md 中的脚本路径已改写为插件内相对路径。

## 来源与验证

- 来源：GitHub 仓库 clone（2026-07-28，main 分支）
- 验证：`validate_qoder_plugin.py` 通过（见提交说明）
