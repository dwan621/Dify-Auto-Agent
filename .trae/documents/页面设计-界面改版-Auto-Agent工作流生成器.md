# 页面设计文档（桌面优先）

## 全局设计（Global Styles）
- 视觉目标：简洁高级（低饱和、弱阴影、强层级）、强可读性（信息密度可控、结构清晰）、强指引（步骤化+空状态）。
- Design Tokens（建议在 :root 统一管理）
  - 背景：--bg=#F6F8FB；卡片：--surface=#FFFFFF；次级面：--surface-2=#F9FBFD
  - 文本：--text=#0F172A；次级：--muted=#64748B；弱提示：--hint=#94A3B8
  - 边框：--border=#E2E8F0；分割线：--divider=#EEF2F7
  - 品牌强调：--primary=#0F766E；--primary-2=#155E75
  - 告警：--warn-bg=#FFF7ED；--warn-border=#FDBA74；--warn-text=#9A3412
  - 圆角：--r-sm=8px；--r-md=12px；阴影：--shadow=0 6px 24px rgba(15,23,42,.08)
  - 字体：标题 28/20/16；正文 14/13；代码等宽字体（YAML/JSON）
- 交互态
  - Button：默认渐变或纯色（更高级建议纯色+轻渐变边）；hover 微提亮；disabled 降饱和+不透明。
  - Focus：所有可交互元素提供清晰 focus ring（2px）。

## 页面：工作流生成器页（/）

### Meta Information
- Title: Auto-Agent 工作流生成器
- Description: 输入需求，自动规划并生成可导入的 Dify YAML 工作流。
- Open Graph: title/description 与上同；type=website。

### Layout
- 桌面优先：12 栅格（CSS Grid）或“左侧 5 / 右侧 7”的双栏布局（支持 1280px 宽度舒适阅读）。
- 响应式：
  - ≥ 980px：双栏固定；右栏可独立滚动（保持左侧输入随时可见或反之）。
  - < 980px：上下堆叠；把“步骤条/关键按钮”吸顶。
- 间距：页面外边距 24；卡片间距 12-16；模块标题与内容间距 8-10。

### Page Structure（信息架构与分区）
1. 顶部 Header（全宽）
2. 主体双栏
   - 左栏：任务输入与配置（“做什么”）
   - 右栏：结果与编辑（“系统给了什么/你能怎么改”）

### Sections & Components

#### 1) Header（标题 + 状态）
- 左侧：H1「Auto-Agent 工作流生成器」+ Subtitle（版本与能力描述）。
- 右侧（或标题下方）：Status Pill
  - 未分析 / 分析中 / 可编辑 / 校验未通过 / 生成中 / 已生成
  - 状态文案与按钮禁用状态一致（避免“按钮能点但状态不对”）。

#### 2) 左栏：任务输入（Guided Input）
- 卡片：步骤条 Stepper（强指引）
  - Step 1 需求描述 → Step 2 分析 → Step 3 编辑 → Step 4 校验 → Step 5 生成
  - 当前步骤高亮；每步给一句“下一步要做什么”。
- 卡片：聊天区（Chat Box）
  - system/user 两种气泡；系统示例收敛到 2-3 行；超过长度可折叠“展开示例”。
- 卡片：需求输入
  - 组件：Select「快速填入示例」+ Button「加载复杂模板」同一行。
  - Textarea：placeholder 强约束（提示输入结构/示例）。
  - 主按钮：分析需求（primary）。
- 卡片：补充配置（Advanced Config）
  - 基础：应用名称、应用描述
  - 进阶：补充问题回答（多行列表）
  - 约束：budget_level、max_exec_nodes
  - 开关：自动导入 Dify（旁边解释：失败不影响下载）
  - 主按钮：生成 YAML（primary）

#### 3) 全局 Health Warn（阻断态提示）
- 置顶 Alert（不抢主视觉但足够明显）：
  - “当前无可用模型，分析/生成已禁用” + 关键配置指引（来自 /health 渲染）。
  - provider 状态用小型列表展示。

#### 4) 右栏：结果总览（Preview Overview）
- 卡片：当前场景
  - sceneLabel 大字展示；无数据时显示“尚未分析，先完成 Step 2”。
- 卡片：规划信息
  - planningMeta 用等宽/标签化（mode/selected/repair_rounds/trace_id）。
- 卡片：候选评分
  - compact list；“已选”用 Tag；为空时展示“分析后出现”。
- 卡片：校验与修复
  - warnings 分级：Error/Warning（若后端不区分，前端可按关键词轻分级）；为空时展示“无告警”。

#### 5) 右栏：节点编辑器（核心工作区）
- 卡片：节点流程（可编辑）
  - 默认折叠，支持“一键展开/收起所有”（仅前端 UI 行为，不改变后端能力）。
  - 每个节点：
    - summary：类型 + id +（如有）title
    - 字段：title input、prompt textarea、config JSON textarea
    - 错误提示：config JSON 解析失败就地提示（替代 alert，提升高级感）。

#### 6) 右栏：生成结果与 YAML 预览
- 卡片：生成结果
  - resultMessage（状态文本）+ artifactMeta（链接）+ downloadLink
  - 次按钮：复制 YAML / 仅校验 Spec（并列，视觉次级）
- 卡片：YAML 内容预览
  - 等宽字体、行高 1.6、深色背景可保留但降低对比刺眼（更高级建议接近 #0B1220）。
  - 支持长内容滚动；预览区顶部提供“复制”快捷入口（可复用现有按钮）。
