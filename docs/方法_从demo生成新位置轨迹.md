# 从源 Demo 分步生成新位置轨迹 —— 实现方法记录

> 本文记录 DemoGen→LIBERO 管线的核心实现:如何把一条人类演示(source demo)
> 改造成"物体摆在任意新位置"的成功轨迹。按管线执行顺序分步说明,并记录
> 每一步的关键发现与踩坑。代码位于 `src/demogen_libero/` 与 `scripts/`。

---

## 总览

```
源 demo (HDF5)                     新场景(网格/连续采样的物体摆放)
   │ ①分段                             │ ③场景采样(防遮挡)
   ▼                                   ▼
(f1,f2,f3) 两阶段边界   +   obj_t / tar_t(目标物/篮子的平移偏移)
   │                                   │
   └───────────► ② 轨迹合成 ◄──────────┘
                 synthesize_uniform
                 (弧长重参数化 + 保留源速度轮廓)
                        │
                        ▼
                 ④ 闭环 replay(前馈+反馈+收敛等待)
                 replay_uniform
                        │
                        ▼
                 ⑤ 成功过滤 + ⑥ OC 格式记录(含 subtask 标注)
```

核心原理(DemoGen):LIBERO 的动作是 **7 维 OSC 增量**
`[dx,dy,dz,dax,day,daz,grip]`,位置增量具有**平移不变性**——
把物体和末端一起平移后,同一段相对动作仍然有效。因此:

- **skill 段**(抓取/放置,含接触):动作**逐帧原样保留**;
- **motion 段**(自由移动):按新起终点**重新规划**。

---

## ① 源 demo 分段(trajectory.py)

把取放任务切成 4 段:
`motion_1(接近) → skill_1(抓) → motion_2(搬运) → skill_2(放)`,
边界为 `(f1, f2, f3)`。

**自动分段 `auto_segment(state, action)`**(state = EE 绝对位置序列):

| 边界 | 判据 |
|---|---|
| `f1`(抓取开始) | 夹爪指令首次由 -1 变 +1 的帧 |
| `f2`(搬运开始) | `f1` 之后 EE 速度首次恢复 > 9mm/帧,再 +2 帧裕量 |
| `f3`(放置开始) | 最后一次夹爪张开指令前 10 帧(把入篮下放包进 skill_2) |

**二次抓取扩展 `segment_regrasp`**:部分人类演示有"抓起-放下-重抓"
(夹爪多个开合周期)。规则改为:`f1` = **首次**闭爪,`f2` = **末次**闭爪后
速度恢复,`f3` = **末次**开爪前 10 帧——整个重抓过程折叠进 skill_1,
对单抓 demo 自动退化为 `auto_segment`(逐帧一致已验证)。

**坑**:自动分段假设单次抓放;不筛查会把重抓 demo 的分段切错
(曾导致一条 transport 只有 1 帧的病态标注)。所以源 demo 先经
`screen_sources.py` 分类:`healthy`(单抓且开环零偏移 replay 成功)/
`regrasp`(多周期,走 regrasp 分段)/ `bad`(开环复现失败,弃用——
接触裕度毫米级的演示,任何微扰即脱手,闭环也救不回)。

---

## ② 轨迹合成 synthesize_uniform(trajectory.py)

给定偏移 `obj_t`(目标物新位置 − 源位置)和 `tar_t`(篮子偏移),
生成**参考路径 ref_path** 与**基础动作 base_actions**:

1. **skill 段原样**:`s1 = action[f1:f2]`、`s2 = action[f3:]` 逐帧保留;
   其参考路径 = 源 EE 路径 + 恒定偏移(skill_1 加 `obj_t`,skill_2 加 `tar_t`)。
2. **motion 段偏移 ramp,且提前完成**:motion_1 的参考路径 =
   源路径 + `w(t)·obj_t`,其中 `w(t)` 在段内**前 70%**(`ramp_frac=0.7`)
   从 0 线性升到 1 后保持——横移在巡航高度完成,**末端下降段纯竖直**。
   - 坑:最初 ramp 铺满整段,下降时还在横移 → 低空侧扫撞倒邻近物体、
     闭爪时还在滑移把瓶子夹歪。提前完成后两类失败消失。
3. **弧长重参数化(时长自适应)**:motion 段按路径弧长以源 demo 的
   平均速度重新采样——**帧数随实际路程增长**,而不是锁定源帧数。
   - 坑 1(帧数固定):远近位置轨迹帧数一样 → 远的每步更大/更快,
     速度分布病态(v_mean 5.9~9.4mm/步不等)。
   - 坑 2(纯匀速):把源"接近物体时减速"的轮廓抹掉 → 冲太快抓取失败。
   - 解:重采样时保留**归一化速度轮廓**(归一化时间→归一化弧长的映射
     取自源段),时长 ∝ 路程、快慢节奏跟随人类演示。修后全数据集
     v_mean 收敛到 5.8~6.9mm/步。
4. motion 段的旋转增量按帧数比例缩放重采样(总旋转量守恒),
   夹爪指令按段保持。

---

## ③ 场景采样(gridscene.py)

场景以**绝对坐标**定义,可套到任意源 demo 的初始 state 上
(同一摆放 × 多条源 demo = `demos-per-scene`):

- **目标物**:工作区内**连续均匀采样**(非网格),farthest-point 打分
  (远离已用位置)使整个数据集摊开;不加篮子距离奖励(会造成远离篮子
  的系统性偏置,踩过)。
- **干扰物**:0.11m 网格 + 15% 抖动,两两间距 ≥ `min_spacing`(0.10m),
  距篮子 ≥ 0.12m;**物体身份固定在场景里**(多源共享同一布局)。
- **篮子**:参考位置 ±1cm 抖动 → `tar_t`。
- **防遮挡(双层)**:
  1. 几何过滤:与相机视线严重共线(横向偏距 < 0.035m)的摆放拒绝;
  2. 渲染硬门槛:摆好后渲染第 0 帧,每个物体的分割像素数须达标——
     **按尺寸自适应**(z 高 <5cm 的扁平物体 60px,其余 150px)。
     坑:一刀切 150px 把 butter(7.7×4×1.8cm)在远端的**无遮挡**
     小分割块误判为"被遮挡",整个任务成功率从 92% 崩到 2%。
- 未选中的干扰物停放到场外 (x≥2.0);自由关节四元数归一化、qvel 清零。
- `apply_scene` 把绝对坐标写进源 demo 的 110 维 init state 的
  对应 qpos 槽位,得到新初始状态,同时给出 `obj_t / tar_t`。

---

## ④ 闭环 replay(libero_replay.replay_uniform)

**为什么必须闭环**:LIBERO 的 OSC 控制器每步以"当前**实际** EE 位置 +
增量"为目标,单步只收敛到 80~90%,损耗**开环下无法追回**。源动作里
人类已隐式补偿了这个损耗,但我们叠加的修正量没有——偏移 25cm 时抓取
时刻 EE 还差 4.4cm,必然抓空。这是早期"大偏移全失败"的根因。

执行策略(motion 段):

```
act[:3] = clip( ff_gain·(ref[t+1]−ref[t])/0.05      # 前馈:参考速度×1.25 补欠冲
              + clip(gain·(ref[t]−ee)/0.05, ±0.4)   # 反馈:拉回参考路径,限幅 2cm/步
              , ±1 )
```

skill 段严格开环(动作原样),保持接触动力学与源示范一致。

**收敛等待(settle)**:进入 skill_1/skill_2 之前,插入至多 20 帧
"纯反馈、夹爪保持"的过渡帧,直到 EE 距参考点 < 5mm 才闭爪/下放——
大偏移的残余误差在这里吸收(这是 25cm 偏移能成功的关键)。
settle 帧的阶段归属:抓取前修正 → skill_1,放置前 → skill_2。

**记录语义**(BC 友好,按需求修正过):
- **步前对齐**:`obs[t]` 是执行 `action[t]` **之前**的观测
  (`states[0]`/`ee_pos[0]` = 初始场景),消除"到位后重放动作"的前冲;
- 采集前 **5 帧零动作预热**(不记录),让物理/控制器稳定;
- 结束后追加 **3 帧零动作保持**(记录,归 skill_2)。

---

## ⑤ 成功过滤

replay 全程逐帧查 `env.check_success()`,任一帧成立即算成功;
失败的尝试**换一条源 demo 重试**(每场景至多 `demos_per_scene +
scene_retries` 次),仍不满则弃场景。典型一次成功率:单抓任务
85~99%,重抓源 ~40%(接触敏感,属预期)。

---

## ⑥ 记录(oc_obs.py)

成功轨迹写为 LIBERO-OC 格式(与参考 `output/demo` 逐字段一致)并附加:

- `obs/`:双相机 RGB、深度(uint8 厘米 + 无损 uint16 毫米)、实例分割
  (id:robot夹爪=50,目标=60,篮子=70,干扰物 80~120)、本体感知;
- `obs/obj_pos (T,7,3)`、`obs/obj_quat (T,7,4)`:逐帧物体真值位姿
  (世界系,xyzw),直接抄 robosuite obs,下游不必解析 state;
- `subtask_id` + `subtasks` attr:闭词表子任务标注(arXiv:2607.06403)——
  `transit(接近,object=目标物) → move(首次闭爪→末次开爪,
  object=目标物, destination=篮子) → transit(撤回) → idle(收尾)`,
  重抓周期天然折叠进同一个 move;
- `phase_id`(内部 4 阶段,工具用)、metainfo(逐帧 bbox、
  `target_object`/`goal_object`、subtasks);
- 数据集目录自带 `camera_params.json`(内参 + agentview 外参 +
  手眼 T_ee_cam)与 `object_geometry.json`(mesh 顶点 AABB 中心偏置 +
  尺寸;篮子原点 z 偏 +7.38cm)。

---

## 关键结论速查

| 问题 | 结论 |
|---|---|
| 为什么 skill 段能搬家 | OSC 增量动作平移不变;EE 与物体同步偏移后相对几何不变 |
| 开环为何不行 | OSC 每步以实际位置为锚,单步欠冲 10~20%,修正量开环追不回 |
| 大偏移的最后一厘米 | 抓取/放置前 settle 至 <5mm 再进 skill 段 |
| 横移的时机 | motion 段前 70% 完成,下降段纯竖直,避免低空侧扫与斜夹 |
| 速度一致性 | 弧长重参数化(时长∝路程)+ 保留源归一化速度轮廓 |
| 脆弱源 demo | 先筛:开环零偏移复现失败的直接弃用;重抓的单独走 regrasp 分段 |
| 遮挡判定 | 几何视线过滤只做粗筛,渲染分割像素数(尺寸自适应阈值)是硬门槛 |
| 轨迹里的"过冲勾形" | 继承自人类源示范(源过冲 8.5cm > 生成 4.8cm),非合成伪影 |

## 相关脚本

| 脚本 | 用途 |
|---|---|
| `scripts/screen_sources.py` | 源 demo 健康筛查(healthy/regrasp/bad) |
| `scripts/run_grid_oc_demo.py` | 主生成:场景采样→合成→闭环 replay→OC 记录(`--task/--segment/--primary-mode`) |
| `scripts/run_all_tasks.sh` / `topup_all.sh` | 10 任务批量生产 / 补齐到目标条数 |
| `scripts/build_eval_suite.py` | 与训练不相交的评估初始场景(自适应距离阈值) |
| `scripts/append_regrasp_demos.py` | 原始重抓 demo 的状态回放式追加 |
| `scripts/visualize_init_states.py` / `visualize_phases.py` | 初始状态拼图 / 阶段图+视频 |
| `scripts/dump_camera_params.py` / `dump_object_geometry.py` | 相机参数 / 物体几何导出 |
