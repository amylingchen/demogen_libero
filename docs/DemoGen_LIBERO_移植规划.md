# DemoGen → LIBERO 移植方案规划

> 目标读者:实现者。本文只做规划,不含最终代码。沙箱里已验证的原型脚本仅作为
> 数学正确性的参考证据(见第 10 节),实现阶段重写为正式工程代码。

---

## 1. 目标

把 DemoGen 的"少量源示范 → 空间增广出大量示范"方法,移植到 LIBERO 仿真,
用于训练 3D Diffusion Policy(DP3)。要求:

- 采用 **DemoGen 式直接改轨迹**(搬移物体中心技能段 + 重规划自由段),
  不是 MimicGen 式"在原轨迹上拼接再执行"。
- 点云采用 **LIBERO 物体几何点云表示**(从 MuJoCo mesh/geom 采样),
  不用相机深度反投影。
- 生成的轨迹**丢进 LIBERO 仿真 replay**,用任务成功判定过滤。
- 产物为 DP3 可直接训练的 zarr 数据集。

---

## 2. 背景与关键结论

### 2.1 DemoGen 机制(源码 `demogen.py`)

任务按夹爪状态切成 4 段(two-stage 取放):
`motion-1(自由移动到物体) → skill-1(抓) → motion-2(搬运到目标) → skill-2(放)`。
给定物体偏移 `obj_trans`、目标偏移 `tar_trans`:

- 两个 **skill 段**:动作**逐帧原样保留**,点云里对应物体点整体平移。
- 两个 **motion 段**:动作**重新插值**,把 home 接到"搬移后的技能起点"。

### 2.2 关键结论:动作是相对增量,state 是绝对(已代码验证)

| 量 | 含义 | 增广时处理 |
|---|---|---|
| `action[:3]` | 逐帧末端**增量 delta** | motion 段重插值,skill 段原样 |
| `action[3:]` | 旋转 + 夹爪 | 原样保留(DemoGen 只平移) |
| `state[:3]` | **绝对**末端位置 | 加累计平移 `trans_sofar` |

**LIBERO 动作 = 7 维 OSC_POSE 增量 `[dx,dy,dz,dax,day,daz,grip]`,与之一一对应,
无需动作空间转换。** 相对动作平移不变 → 技能段可整段搬移而动作不变,这是本方法成立的根本。

---

## 3. 已定设计决策

1. **生成路线**:DemoGen 合成动作 + LIBERO 仿真 replay(成功过滤 + 几何点云渲染)。
2. **任务结构**:two-stage 取 + 放。
3. **点云表示**:物体几何点云(MuJoCo geom 采样),放弃相机深度。
4. **动作表示**:延用 LIBERO 原生 7 维 OSC 增量(与 DemoGen 一致,零转换)。
5. **分段方式**:少量源示范 → 手工指定三个分段帧(推荐,零偏移 replay 后读帧号)。

---

## 4. 待拍板决策

| # | 决策点 | 选项 | 建议 |
|---|---|---|---|
| D1 | 机械臂点云怎么生成 | (a) 整体按 `trans_sofar` 刚性平移(DemoGen 原味)  (b) 按 `robot0_joint_pos` FK 重新采样 | 先用 (a) 跑通;若手臂穿模/位姿不符再上 (b) |
| D2 | 是否保留纯离线变体 | 只做 replay / 同时保留离线对照 | 同时保留,便于消融"replay 过滤"的价值 |
| D3 | 偏移采样 | grid / random | 先 grid 便于可视化覆盖,后 random 增量 |
| D4 | 物体旋转 | 只平移 / 平移+旋转 | 先只平移(与 DemoGen 一致),旋转作为后续扩展 |

---

## 5. 系统架构与模块

```
LIBERO HDF5 源示范
   │  convert:  读 state / action / init_state
   ▼
轨迹合成 (trajectory)  ── synthesize_two_stage(state, action, frames, obj_t, tar_t)
   │        输出:重定位后的 7 维增量动作序列
   ▼
仿真 replay (libero_replay)
   │   1) reset 到源初始 sim state
   │   2) 把物体/目标 body 位姿平移 obj_t / tar_t
   │   3) 开环 replay 动作序列
   │   4) 每帧用几何点云渲染 (objectcloud.scene_cloud)
   │   5) 任务成功判定过滤
   ▼
写出 DP3 zarr (generate)  ── agent_pos / point_cloud / action / episode_ends
```

模块清单:

- `convert.py` — LIBERO/robomimic HDF5 → 每条源 demo 的 `state / action / init_state`。
- `trajectory.py` — `synthesize_two_stage(...)`(two_stage_augment 线性分支移植)+ 偏移生成。
- `objectcloud.py` — MuJoCo geom 采样 → (N,6) 场景点云;含分组编辑 `edit_cloud_by_group`。
- `libero_replay.py` — 建 env、平移物体、replay、渲染点云、成功过滤。
- `generate.py` — 编排 + 配置 + 写 zarr。
- `tests/` — 数学核心单测(轨迹、几何点云)。

---

## 6. 数据流与数据格式

- 源:LIBERO HDF5,`data/demo_i/{actions, states, obs/robot0_eef_pos, ...}`。
- 中间:每条源 demo 的 `state[:, :3]=绝对EE`、`action=7维增量`、`init_state=首帧sim state`。
- 产物:zarr,键与 DemoGen 一致,DP3 loader 可直接用:
  - `data/agent_pos` (ΣT, Ds)
  - `data/point_cloud` (ΣT, 1024, 6)  # xyzrgb
  - `data/action` (ΣT, 7)
  - `meta/episode_ends` (n_episodes,)

---

## 7. 关键实现细节

### 7.1 分段(frames)
每条源 demo 手工给 `(skill_1_frame, motion_2_frame, skill_2_frame)`=抓开始/搬运开始/放开始。
做法:零偏移跑一遍 replay 渲染视频,读帧号。放进配置 `FRAMES[demo_key]`。

### 7.2 轨迹合成
- motion-1:`start = state[0][:3] - action[0][:3]`(home),`end = state[skill_1-1][:3] + obj_t`,
  匀速增量;逐帧累计 `trans_sofar[:2] += step - source_action[:3]`。
- skill-1:动作原样,state 加 `trans_sofar`。
- motion-2:`trans_togo = tar_t - obj_t`,重插值。
- skill-2:动作原样,state 加 `tar_t`(= translate_all_frames)。

### 7.3 几何点云(objectcloud)
- 关心的 body:物体、目标、机械臂各连杆、桌面。
- 每个 body 的 geom 在 body 系**预采样一次**(box/球/圆柱解析;mesh 用顶点/面片采样)。
- 每帧用 `data.geom_xpos / geom_xmat` 刚性变换 → 拼接 → FPS 到 1024。
- 分割白送(每点带 body 标签)。
- **一致性铁律**:生成用的 body 集合 + n_points,必须和 DP3 评估时**完全相同**。

### 7.4 物体位姿平移(replay 里)
通过物体 free joint 的 qpos(`[x y z qw qx qy qz]`)前三位加偏移,`sim.forward()`。
需填 `obj_joint / tar_joint` 名(查 `sim.model.joint_names`)。

### 7.5 机械臂点云(D1)
默认 (a):把机械臂那组点整体按 `trans_sofar` 平移(与 DemoGen 完全一致)。
可选 (b):读 `robot0_joint_pos` 做 FK 重新采样(对应 DP3 的 `imagin_robot`)。

---

## 8. 实现步骤 / 里程碑

- **M0 打通读写**:convert 读 1 条 demo;写一个假 episode 到 zarr,DP3 loader 能加载。
- **M1 零偏移自检**:replay 源动作本身(obj_t=tar_t=0),确认 replay 管线 + 成功判定接通,
  并读出正确分段帧。
- **M2 几何点云**:接 `objectcloud.scene_cloud`,可视化确认物体/机械臂/桌面点正确、无穿模。
- **M3 单点偏移**:一个非零偏移,合成 → replay → 成功;可视化点云与轨迹一致。
- **M4 批量生成**:grid 偏移 × 多源 demo,统计成功率,写出完整 zarr。
- **M5 训练验证**:用生成数据训 DP3,在 LIBERO 随机初始化上评估成功率;
  与"仅源示范"基线对比。
- **M6(可选)**:纯离线变体对照(D2)、旋转扩展(D4)、机械臂 FK(D1-b)。

---

## 9. 测试与验收

- **单测(离线,必做)**:
  - 轨迹:skill 段动作不变、末端-物体/目标相对几何保持、增量积分重建绝对轨迹、到达搬移后目标上方。
  - 几何点云:采样点在物体表面、刚性变换保距、分组平移等价于改位姿、FPS 尺寸正确。
- **集成(仿真)**:M1 零偏移必须成功;M3/M4 报告成功率(建议 grid 先看覆盖)。
- **端到端(验收)**:M5 中生成数据训练的 DP3,在未见初始位姿上成功率显著高于源示范基线。

---

## 10. 风险与注意事项

- **特权信息**:几何点云依赖 sim 真值位姿,适合 LIBERO 仿真研究,**无 gen2real**,不可迁真机。
- **开环 replay 漂移**:OSC 增量开环对准静态桌面任务通常够;漂移则减小单步增量或增控制步数。
- **分段错误是头号坑**:分段帧错 → 搬移点错位、动作拼接崩。零偏移 replay 逐条核对。
- **一致性**:动作空间/归一化、点云 body 集合/点数,生成与评估必须严格一致。
- **只平移**:当前不含物体旋转;旋转需同时旋转 skill 段增量的旋转分量与位置增量所在坐标系。

---

## 11. 已验证的原型(参考,非最终代码)

沙箱中已用纯 numpy 验证核心数学,实现阶段可据此重写:

- `demogen_libero/trajectory.py` + `tests/test_core.py`:two-stage 重定位,8 项检查全过。
- `demogen_libero/objectcloud.py` + `tests/test_objectcloud.py`:几何采样/变换/分组编辑/FPS,全过。
- `demogen_libero/{convert,libero_replay,generate}.py`:LIBERO 对接骨架(标注 `ADAPT` 处待填)。

这些文件保留作为实现参考;正式实现时按本规划重整为工程代码。
