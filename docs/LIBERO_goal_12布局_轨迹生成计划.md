# LIBERO-Goal 12 布局轨迹生成计划

日期：2026-08-22。决策来源：graphslot-vla 项目 goal 套件设计讨论（六轮收敛）。
本文档指导在 demogen_libero 中为 libero_goal 生成布局多样化的演示轨迹与评估 init state，并定义 seen / unseen / counterfactual 划分。

---

## 1. 已核实的机制事实（生成设计的前提）

以下事实 2026-08-22 从 bender `~/project/LIBERO/libero/libero/bddl_files/libero_goal/` 的 BDDL 原文与 `envs/objects/articulated_objects.py` 读出：

1. **10 个 goal 任务共用同一场景**：fixtures（wooden_cabinet_1、flat_stove_1、wine_rack_1）、4 个可动物体（akita_black_bowl_1、cream_cheese_1、wine_bottle_1、plate_1）、init 区域逐字相同，只有 `:goal` 谓词和语言指令不同。
2. **原版初始随机化区域极小**（均为 2cm×2cm 框，坐标相对 main_table）：

   | 实体 | 名义位置 (x, y) | yaw |
   |---|---|---|
   | plate_1 | (0.05, −0.02) | — |
   | akita_black_bowl_1 | (−0.09, 0.00) | — |
   | wine_bottle_1 | (−0.20, −0.05) | — |
   | cream_cheese_1 | (−0.05, 0.13) | — |
   | wooden_cabinet_1 | (0.03, −0.24) | π（固定） |
   | flat_stove_1 | (−0.41, 0.21) | 未指定（默认 0） |
   | wine_rack_1 | (−0.26, −0.26) | π（固定） |

3. **三个 fixture 都带 free joint**（`WoodenCabinet`/`FlatStove` 是 `ArticulatedObject`，`joints=[dict(type="free")]`；WineRack 为 TurbosquidObjects）→ fixture 位姿在 sim qpos / init state 里，可用"改 qpos + settle"路径移动，不必重写 BDDL。
4. **goal 谓词挂在 fixture 本体上**（`flat_stove_1_cook_region`、`wooden_cabinet_1_top_region` 等是以 fixture 为 target 的附着区域），fixture 移动后判据自动跟随，无需改谓词。

> 注意：本仓库现有 oc_obs / run_grid_oc_demo 管线是 **libero_object** 接线（seg id 表 OBJECT_ORDER、states 维度 qpos=58、相机参数均为 object 套件的）。goal 是不同场景，以上均需重新接线，见 §7。

## 2. 决策汇总

| # | 决策 | 说明 |
|---|---|---|
| D1 | 方案 A：布局间 fixture **只平移**，yaw 不跨布局变化 | cabinet/rack 保持 yaw=π、stove 保持默认，仅布局内小抖动。避免建旋转变换机械 |
| D1a | **（2026-08-22 修订）fixture 平移范围 = 实测可行走廊，不是名义位 ±12cm** | 9 任务中 6 个锚在 fixture 上，±12cm 框内 12 个布局的 fixture 位置挤在 24cm 见方里，unseen 布局在承重维度上与 seen 几乎重合。角点重放探针（scripts/probe_goal_fixture_extremes.py）实测走廊见 §4 |
| D2 | 12 个布局：**8 seen + 4 unseen** | 布局 = 3 个 fixture 位姿 + 4 个可动物体名义位置的完整摆放 |
| D3 | 每布局内 target 与 goal 端轻微扰动，每个 任务×布局 格生成 **5 条 demo** | 扰动量见 §4 |
| D4 | seen 布局内任务对半分：一半格子作训练、另一半作 **counterfactual** | 分配规则见 §5 |
| D5 | **排除** `open_the_top_drawer_and_put_the_bowl_inside` | 双段复合任务，本次模型不考虑复杂任务 → 剩 9 个任务 |
| D6 | **全部 12×9 格都生成轨迹**作为可达性证明 | 非训练格的轨迹隔离存放，不进训练集（见 §6.4） |
| D7 | 机器人初始状态在 demo 初始附近**轻微扰动**；参考轨迹（源 demo）**随机选取**，不固定 | 见 §6.2、§6.3 |

## 3. 任务清单与段锚定表

排除 D5 后共 9 个任务。分段变换重放的原则：**每个轨迹段在其锚定体的坐标系里做刚性变换**（本仓库 `docs/方法_从demo生成新位置轨迹.md` 的方法，扩展"锚定体可以是 fixture"）。方案 A 下 fixture 只平移，所以 fixture 锚定段只需 delta 平移——与物体端是同一种变换。

| 任务 | 抓取/操作段锚定 | 放置段锚定 | 备注 |
|---|---|---|---|
| open_the_middle_drawer_of_the_cabinet | cabinet | — | 全轨迹随 cabinet 平移；接触段（把手抓握）对误差敏感，产率需实测 |
| turn_on_the_stove | stove | — | 全轨迹随 stove 平移 |
| put_the_bowl_on_the_plate | bowl | plate | 双端可动，变换最复杂 → 试点任务之一 |
| put_the_bowl_on_the_stove | bowl | stove | 单可动端 → 试点任务之一 |
| put_the_bowl_on_top_of_the_cabinet | bowl | cabinet | 柜顶接近臂展极限，布局采样时对该任务单独做可达门 |
| put_the_cream_cheese_in_the_bowl | cheese | bowl | 双端可动 |
| put_the_wine_bottle_on_the_rack | bottle | rack | 放置姿态敏感（插槽朝向），rack 布局内 yaw 抖动保持最小 |
| put_the_wine_bottle_on_top_of_the_cabinet | bottle | cabinet | 同柜顶可达问题 |
| push_the_plate_to_the_front_of_the_stove | ⚠ plate+stove 双锚连续接触 | — | **无法分段变换**。保留在全格生成里试跑，产率低于门槛（建议 <40%，试点后定）则从增广中剔除并如实记录，任务只在原版布局评估 |

## 4. 布局设计规范

**布局间（12 个布局之间的差异）：**
- 3 个 fixture：在各自**实测可行走廊内均匀采样**（D1a 修订，2026-08-22 角点重放探针实测，`goal_scene.GoalSpec.fixture_corridor` + `_corridor_cut`），yaw 不变（D1）。走廊（base body 桌面系坐标）与探针证据：

  | fixture | 走廊 x | 走廊 y | 排除角（探到 0/2 的死角） | 角点证据 |
  |---|---|---|---|---|
  | wooden_cabinet | (−0.17, 0.13) | (−0.38, −0.10) | x<−0.05 且 y<−0.30（后左） | (−0.17,−0.08) 抽屉+柜顶 ✅；(0.13,−0.38) 抽屉 1/2 + 柜顶 ✅；(−0.17,−0.38) 双任务 0/2；(0.13,−0.08) 物体摆不下（由采样门自动拒） |
  | flat_stove | (−0.50, −0.15) | (0.16, 0.33) | x<−0.45 且 y<0.26（后中） | (−0.15,0.33)、(−0.50,0.33) 旋钮+放碗 ✅；(−0.50,0.08) 0/2；burner y≤0.14 挤占放置区（base (−0.30,0.14) 物体摆不下）|
  | wine_rack | (−0.42, −0.10) | (−0.38, −0.14) | x>−0.16 且 y>−0.20（中前） | 4 角 3 过（含对角 32cm 两极端）；(−0.10,−0.14) 0/2 |

  三个走廊均为 ±12cm 框的约 2 倍宽；走廊的"内边界"由物体放置可行性顶住（fixture 侵入中央放置区时 sample_objects 摆不下 → 布局自动被拒），"外边界"由臂可达顶住。
- 4 个可动物体：名义位置在桌面自由区重新采样，约束：物体两两最小间距（半径感知）、不压 fixture 足迹、agentview 内不互遮/不被 fixture 遮挡；**接收放置的物体（plate、bowl）限制在中央可达带**（plate x(−0.10,0.13) y(−0.18,0.18)，bowl x≤0.13——smoke v1 中 plate 采到远角 (0.137,0.259) 使 bowl→plate 产率塌到 1/8，收紧后 v2 全任务首条成功）。
- L01 建议取原版布局（fixture 与物体都在 §1 名义位），作为与官方套件的锚点。

**布局内（同一布局 5 条 demo 之间的抖动，即 D3 的"target 和 goal 轻微扰动"）：**
- 可动物体：名义位置 ±2.5cm（待试点校准，上限受重放容差约束）。
- fixture：±1cm 平移 + yaw ±5°。
- 机器人初始状态：见 §6.2。
- **抖动泄漏护栏（2026-08-23 第 2 轮审查 B5 + 用户决策）**：±2.5cm 抖动会把 manifest 的 8cm 中心距下限侵蚀到最坏 ~1.4cm。生成时每条训练 demo 的抖动后实体位置必须与每个 unseen 布局同实体保持 **≥6cm**（`goal_scene.jitter_ok`，重采抖动直到满足）；unseen 评估 init 的抖动对 seen 侧同规则镜像。生成完成后实测报告"抖动后的真实最小分离"，不再引用中心距 0.0809。

**布局采样门（每个布局进入套件前必须全过）：**
1. settle 后所有实体静止（`probe_settle_convergence.py` 可作起点）；
2. agentview 可见性：所有任务相关部件（把手、旋钮、cook region、rack 插槽、柜顶面、全部可动物体）不被遮挡——cabinet 是高大遮挡体，此门要严格；
3. 臂可达：柜顶任务与抽屉把手逐布局做 IK 可达检查；最终背书是 D6 的全格轨迹生成成功；
4. 查重：布局间两两距离下限（此前套件出过 unseen 布局重复 5 次的事故）；
5. 泄漏守卫（2026-08-23 第二次修订，对抗审查第 1 轮阻塞级发现后）：**逐实体（全部 7 个）、对全体 8 个 seen 布局、unseen-first 采样强制**。审查确立的事实：训练的是一个多任务策略，它通过其他任务的训练格看过全部 8 个 seen 布局的画面，因此"任务感知的训练格距离"（中间版本，报 0.074m）没有区分力——其背后 unseen↔全体 seen 的真实最小距离低至 1.64cm。现行方案：先采 4 个 unseen（逐实体互距 ≥0.06m、与 L00 逐实体 ≥0.08m、盘子与 push 可行盘中心 ≥0.11m），再采 7 个 seen（逐实体与每个 unseen ≥0.08m）——下限由构造保证。0.08m 是用户在可行性权衡后定的值（0.10m 对全体 seen 在走廊几何上不可行：4 个 r=0.10 排除圆面积超过柜子走廊本身）。manifest 报每个 unseen 布局的逐实体最小距离表 + 全局下限，并记录各门的拒绝计数。
6. push 处置（用户决策 2026-08-23）：push 的目标区固定在桌面（§1.4 对该任务不成立），整轨平移只容忍 ~3-4cm 盘子位移 → **push 只在"可行盘"布局排训练格**（盘子在名义位 r=0.03 圆盘内的 L00 + 3 个受控采样的 seen 布局，每格用真实 push 重放验证），其余全部格在 manifest 标记 `push_infeasible_cells`，push 无 cf 格。
7. 重放可达门（替代伺服可达图）：每个候选布局必须让 **8 个非 push 任务**各在 ≤2 条源 demo 内末帧成功（伺服可达图两轮与已知成功重放交叉验证不一致，弃用，弃用理由与复现记录见 goal_scene.py / 审查报告）。settle 门 = 收敛 + 位移 <1cm + 抽屉/旋钮关节零自滑。

## 5. 划分定义与任务×布局分配矩阵

- **seen（训练）**：8 个 seen 布局中被指派为"训练格"的 任务×布局 格。
- **counterfactual**：8 个 seen 布局中的其余格——**布局在训练中见过（以其他任务），指令在训练中见过（在其他布局），但该 指令×布局 组合从未训练**。磁带策略在熟悉布局里会执行该布局训过的某个任务而不是指令要求的任务，因此 cf 评估必须**每条 rollout 结束时同时检查全部 9 个任务谓词**，输出混淆矩阵，不只报目标任务 SR。
- **unseen**：4 个 unseen 布局 × 9 个任务的全部格。评估只用 init state。

**分配规则（约束随机，不用纯随机）：** 每个任务恰在 8 个 seen 布局中的 **4 个**被指派为训练格、其余 4 个为 cf 格。由此每个布局的训练任务数为 4~5 个（9×4=36 个训练格摊到 8 个布局）。随机种子写进 manifest（§8）。

**数量核算：**

| 集合 | 格数 | 轨迹数（×5） | 用途 |
|---|---|---|---|
| 训练格 | 9 任务 × 4 布局 = 36 | 180 | 训练集 |
| cf 格 | 9 × 4 = 36 | 180 | 仅可行性证明 + 评估 init |
| unseen 格 | 9 × 4 = 36 | 180 | 仅可行性证明 + 评估 init |
| 合计 | 108 | **540** | |

已知限制（记录，不是待办）：每任务训练 demo 仅 4×5=20 条（原版 50 条）；unseen 布局 n=4，评估必须报**逐布局 SR**而非只报均值，结论对单布局特异性敏感。

## 6. 生成协议

### 6.1 源 demo 选择——随机化，不固定（D7）
每条生成轨迹的参考源 demo 从该任务的 50 条原版 demo 中选取：先按源摆放与目标摆放的距离取 top-k（建议 k=5，待试点校准），再在 top-k 内**随机抽一条**。目的：同一格的 5 条 demo 不是同一盘磁带的 5 份拷贝，训练分布里的轨迹形态有随机性。源 demo id 逐条写进 manifest。

### 6.2 机器人初始状态扰动（D7）
- 在源 demo 初始 qpos 附近加小关节噪声（建议 σ=0.02 rad，待试点校准），settle 后作为 episode 初始状态。
- 门：扰动后末端执行器位移 ≤ 2~3cm，且**训练 demo 与评估 init 用同一套扰动协议**——历史教训：评估端 init 重生成曾把手臂姿态漂出训练分布 6.6cm，直接压低 unseen 读数。生成完成后抽样对比训练集与评估集的初始 ee 位姿分布。
- 重放对初始扰动的耐受性是试点必测项：若开环回放动作序列在扰动初始下漂移过大，改为对变换后参考路径做闭环跟踪（ee 空间伺服到路径），让初始扰动在前几步内洗掉。

### 6.3 分段变换重放
1. 相位切分：按夹爪开合信号切 reach / grasp / transport / place（或 approach / operate，对 fixture 任务）；
2. 各段按 §3 锚定表做 delta 平移（锚定体从源摆放到目标摆放的位移）；transport 段在两锚定之间插值过渡；
3. 布局内 fixture yaw ±5° 抖动不做旋转变换，靠闭环容差吸收（试点验证此假设，不成立则收紧抖动到 0°）；
4. settle 协议贯穿：初始摆放 settle 后才记录 init state；
5. 逐条用**该任务自己的 goal 谓词**验收，失败即弃，弃用计入产率。

### 6.4 全格生成与隔离（D6）
cf 格与 unseen 格的轨迹同样生成（可达性证明，且与训练格用同一把尺子，避免"可重放偏置"只偏向训练侧），但**物理上隔离存放**（独立目录 + manifest 标记 `quarantine: true`），打包训练 hdf5 时以 manifest 为准只取训练格。这些轨迹保留不删（生成数据按删除保护 A 级对待），后续失败分析可用作参考 rollout。

## 7. goal 套件接线清单（相对现有 object 管线的差异点）

**接线状态（2026-08-25 逐条对照实际产物核实）：**

| 条目 | 状态 | 证据 |
|---|---|---|
| 7.1 场景加载 + settle 验证抽屉/旋钮不自动 | ✅ | `sample_goal_suite` settle 门含 `articulation_drift<0.005`，12 布局全过 |
| 7.2 goal 专属相机参数 + 重投影自检 | ✅ | `camera_params.json` 自检 4/4。**顺带纠正一条约定**：`project_points_from_world_to_camera` 返回的行是 upright 约定，索引本数据集存储的原始 GL 图像需 `H-1-row`（实测：奶酪投影行 154 vs 分割质心 100，盘子 171 vs 86，列坐标精确吻合） |
| 7.3 seg id 表 + cabinet 拆 middle/top drawer 部件级 id | ⚠️ **可行性已验证，实现未做** | `scripts/probe_drawer_seg.py` → `output/goal_drawer_seg/drawer_seg_probe.json`：instance 级只有整体 `wooden_cabinet_1`（11 个 id），但 **element 级（38 个 id）按 `geom_bodyid` 归组可得部件掩码**。关闭状态下 middle=412px、top=469px，均高于 60px 门槛，初始帧可接地。关节↔body 命名逐个验证正确（`{lvl}_level` 驱动 `cabinet_{lvl}`，y 由 −0.245 移到 −0.085）。**关键陷阱**：静止抽屉的掩码会因邻居打开而暴涨——开中间抽屉使 `cabinet_top` 从 469px 涨到 4249px（缺口露出上层抽屉箱体），因此**绝不可用"柜体部件里最大的掩码"来选指令所指的抽屉** |
| 7.4 object_geometry.json | ❌ **未做** | 部分信息散在 `output/goal_geometry/goal_event_ee.json`（实测作业点偏移），无正式几何导出 |
| 7.5 states 维度实测不硬编码 | ✅ | 实测 79 = time+qpos(41)+qvel(37) |
| 7.6 fixture 位姿 wxyz 写 sidecar + **像素验证** | ✅ | `scripts/verify_goal_fixture_pixels.py` → `fixture_pixel_check.json`：12 布局复现像素差 ≤0.23%，**阴性对照（不应用 fixture_edits）9.8–28.7%**；投影 8 类点 7 类 12/12，酒架底座 5/12 因被柜体遮挡（其任务关键点槽位 12/12） |
| 7.7 每条轨迹记录 fixture 末位姿 | ✅ **不需要** | fixture 无 free joint、为焊接体，撞不动（§1.3 的前提有误，见文首） |

未做的 7.3/7.4 均为 OC 观测渲染的前置工作；7.6 的阴性对照同时证明了 `states`+attrs 足以重建场景，即 OC 观测可走状态重放、无需重跑物理。

1. **场景加载**：goal 的 KITCHEN 场景 + 三个 articulated fixture（cabinet 抽屉关节、stove 旋钮）；settle 时验证抽屉不自行滑开、旋钮不自转。
2. **相机参数**：goal 套件 agentview 与 object/spatial 不同（逐套件相机不同是既有教训，spatial 上用错参数曾偏 320px）。用 `dump_camera_params.py` 从 goal env 导出，**先跑重投影残差自检**（把已知 3D 点投回像素与渲染对照）再进生成。
3. **seg id 表**：OBJECT_ORDER 是 object 套件的，为 goal 新建（4 可动物体 + 3 fixture + gripper），其中 **cabinet 需拆出 middle_drawer 与 top_drawer 两个部件级 id**（柜体与各抽屉分别可寻址）；沿用"图像上下翻转后存储（upright）"的仓库约定，并跑翻转指纹检查（中线物体正确、上下成对互换 = 漏翻）。
4. **物体几何**：`dump_object_geometry.py` 补导 wine_bottle、plate、akita_black_bowl、cream_cheese 及三个 fixture（含 cabinet 顶面高度、rack 插槽位姿、stove cook region / 旋钮位置）。
5. **states 维度**：goal 场景 qpos/qvel 维度与 object（58/51）不同（fixture free joint + 抽屉/旋钮关节），一切按 `env.sim.get_state().flatten()` 实测，不硬编码。
6. **fixture 位姿写入**：四元数按 MuJoCo 顺序（wxyz）写 sidecar 的既有约定沿用（graphslot 仓库 fixture sidecar 那次 commit 的教训），并用像素验证。
7. **fixture 末位姿记录**：每条轨迹（生成与评估）结束时记录三个 fixture 位姿——free joint 意味着被撞会滑，谓词不受影响但布局标签会漂，分析时需能过滤"fixture 被撞离位"的 episode。

## 8. 交付物

1. 训练 hdf5：按任务分文件，格式沿 `docs/dataset_format_field_reference.md`；只含训练格的 180 条（减产率损耗）。
2. 评估 init state 文件：12 布局 × 9 任务全格（cf 评估用 seen 布局格，unseen 评估用 unseen 布局格）。
3. 隔离轨迹目录（cf + unseen 格的可行性证明轨迹）。
4. **manifest.json**：12 个布局的完整参数（fixture 位姿、物体名义位）、任务×布局分配矩阵、全部随机种子、每格的源 demo id 列表、每格产率、布局采样门的检查结果（可见性/可达/查重/最近邻距离逐布局最小值）。
5. goal 相机参数与几何导出件 + 重投影自检报告。

## 9. 执行顺序

1. **接线**（§7 的 1–5）→ 重投影残差自检通过为门；
2. **试点**：任选 2 个布局，跑 `put_the_bowl_on_the_stove`（单端）与 `put_the_bowl_on_the_plate`（双端）各 ~30 条，实测产率并校准 §4/§6 中所有标注"待试点校准"的参数；同轮试 push 任务定去留、试 §6.2 的初始扰动耐受性；
3. **布局采样**：采 12 个布局过全部门，锁定 manifest；
4. **全格生成** 540 条（减产率损耗），逐格记录；
5. **打包与自检**：位置覆盖图、逐格产率表、训练/评估初始 ee 位姿分布对比、泄漏守卫数字。

## 10. 帧级感知增广（只渲染静态帧，不生成轨迹）

P2 先例：object 套件 4000 帧布局增广续训感知，unseen 绑定尾部 p90 1.80→1.60。goal 上复刻该配方，作为感知消融 arm（两个 arm 策略 demo 完全相同、只差感知训练，读数差异可干净归因）。

- **采样**：fixture 位姿 + 可动物体位置在门约束内**连续采样**（不限于 12 个布局——帧没有重放约束，可以铺满分布）；yaw 可放宽到评估探针会用到的范围（±10~20°），这是帧比轨迹便宜的地方。机器人以 §6.2 协议的抖动初始姿态入画。每帧 settle 后渲染。
- **泄漏守卫（本节的关键纪律）**：帧采样**排除 4 个 unseen 布局的邻域**（建议排除半径待定，与 §4 查重距离同尺度），并在 manifest 里报全体帧到各 unseen 布局的最近距离（逐布局最小值）。否则感知端提前见过 unseen 布局，unseen 评估在全系统层面失效。cf 格所在的 seen 布局不需排除（cf 考验的是指令×布局组合，布局本身就是 seen）。
- **标签**：GT mask + 3D 位置，覆盖 4 个可动物体与 3 个 fixture。mask 生成注意 GL 缓冲垂直翻转指纹检查（§7.3 同一条）。**槽粒度（已决）：cabinet 必须拆出 middle_drawer 与 top_drawer 两个部件级实体**，各自有独立 mask 与 3D 位置标签（与 §7.3 的 seg id 拆分对应）——"open the middle drawer" 的接地要求指令名词能落到具体抽屉，柜体一个整槽分不开 middle/top。先跑 graphslot 仓库的 `probe_drawer_seg.py` 验证抽屉部件在 mask 层确实可分（这是可行性验证，不再是方案选择）；灶台旋钮/cook region 是否再细分，以该探针结果顺带定。帧生成在此之后。
- **规模**：参照 P2 量级（~4000 帧）起步，试点后调。
- **判定读数**：离线先判——留出 fixture 位姿上的绑定误差 p90 / 最坏帧（P2 同款仪器）。感知在新 fixture 位姿上对不准，策略层评估没有解释空间，就地止损。
- **交付物**：帧数据集 + 标签 + 帧 manifest（采样参数、种子、泄漏守卫数字），并入 §8。

## 11. 范围外（记录以免丢失）

- 致盲探针（冻结视觉输入跑原版套件）：在 graphslot 策略仓库先于一切评估执行，它决定所有读数的解释框架。
- fixture 大角度 yaw / 站位置换探针：评估端 far-OOD 探针，不在本次生成范围。
