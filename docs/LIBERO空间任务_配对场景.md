# LIBERO_Spatial：关系场景与配对双关系数据

`libero_object` 的做法是把目标物体搬到桌面任意位置再重定向轨迹。这套办法搬不到
`libero_spatial`，因为**那里的指令本身就是一条空间关系**——"拿起**灶台上**的黑碗"、
"拿起**盘子和 ramekin 之间**的黑碗"、"拿起 **ramekin 旁边**的黑碗"。碗一旦自由移动，
指令就不成立了。

本文覆盖由此衍生的两件事：

1. **关系刚性组** —— 如何在保持指令为真的前提下增广空间场景。
2. **配对双关系场景** —— 一个场景里两个碗各自处于不同关系，让语言成为唯一的选择依据。

---

## 1. 关系刚性组

十个 `libero_spatial` 任务共用同一个场景：两个完全相同的黑碗、一个盘子、一个饼干盒、
一个 ramekin，以及两个 fixture（平底灶台、木柜）。目标恒为 `akita_black_bowl_1`，
放置终点恒为 `plate_1`，第二个碗是视觉上完全相同的干扰物。

增广移动的是一个**刚性组**：

```
组 = {目标碗} ∪ {锚点物体} ∪ {锚点 fixture}
```

施加单一平面变换（绕组质心的平移 + 偏航）。内部关系——也就是指令——被精确保留，
而整个组可以在桌面上移动。每个成员保持自己的 z，所以叠在 ramekin 上的碗仍然叠着，
灶台上的碗仍然在灶眼上。

自由物体通过自由关节的 qpos 移动。**fixture 没有自由关节**：它们通过
`model.body_pos` / `body_quat` 移动，而这是模型数据，`env.reset()` 会重建并清除——
所以 fixture 的修改必须在每次 reset 之后、回放之前重新施加
（`spatial_scene.apply_fixture_edits`）。

下游合成完全不变：`synthesize_uniform` 只需要目标的净位移 `obj_t` 和盘子的 `tar_t`。
碗是圆的，抓取对偏航不敏感，技能段不用改。

逐任务的锚点定义在 `src/demogen_libero/tasks_config_spatial.json`，
采样器是 `src/demogen_libero/spatial_scene.py`。

---

## 2. 配对双关系场景

### 动机

一个场景只有一条关系时，干扰碗会被推离所有锚点，于是模型可以学到一条捷径：
*挑那个靠近某个东西的碗*。完全不读指令也能拿高分。

配对场景让**两个**碗都处于有意义的关系中——一个在灶台上、一个在盘子旁——
捷径失效，指令成为唯一的判别依据。

LIBERO 其实已经隐含地这么做了：`on_the_stove` 的源场景里 bowl_1 在灶眼上、
bowl_2 在柜顶上；`on_the_wooden_cabinet` 是同一布局角色互换。同类还有
on_ramekin↔on_cookie_box、next_to_ramekin↔next_to_cookie_box、
from_table_center↔next_to_plate。这里做的是把它系统化、可控化，而不是固定的几组。

### 一个几何，两条演示

每个任务的 BDDL 目标都是 `(And (On akita_black_bowl_1 plate_1))`，OC 格式的 seg id 60
也绑定在同一个名字上。所以指令所指的那个碗被写成 `akita_black_bowl_1`，另一个碗接管
剩下的位姿。

两个碗是同一个资产——实测确认：各 41 个几何体，类型、尺寸、rgba、摩擦系数完全相同，
总质量都是 0.005591 kg。**因此交换位姿与交换名字在像素上完全等价**，而且不动 BDDL、
不动 seg id 约定、不动 `target_joint`、不动 `obj_pos` 顺序。反之，若把目标改成
`akita_black_bowl_2`，seg 60 会静默地落到干扰物身上。

### 什么判定一个场景有效

包围体积对这些资产无法裁定碰撞。灶台的灶眼是一个凸起的圆环，其包围盒把坐在里面的碗
整个吞掉；碗沿则合理地悬在扁平盘子的边缘之上。圆盘、有向包围盒、按高度分层分解，
三者要么拒绝 LIBERO 自己发布的场景，要么就检测不出真实重叠。所以判据是**实测量**，
每一个都在全部 500 个真实源初始状态上标定：

| 项目 | 判据 | 真实场景上的取值 |
|---|---|---|
| 碰撞 | MuJoCo 接触穿透深度 | 最大 0.00147 m |
| 稳定性 | 沉降后的**关系漂移** | 最大 0.01919 m |
| 可见性 | 渲染分割面积 | 最小 491 px |

决定稳定性的是关系漂移而非原始位移：LIBERO 自己的"碗在 ramekin 上"场景会重新就位
0.019 m，完全正常，所以位移阈值分不清无害的重新就位和碗滑落。

**只由有效场景构成的对照不算对照**——它无法区分"检查器在工作"和"检查器永远说通过"。
因此 `screen_spatial_pairs.py` 还会跑证伪探针：人为构造的重叠配置，必须被拒绝。
它抓到过一个真实缺陷：作用在分层之间的接触容差静默地让 21 个物体对中的 11 对
永远检测不出碰撞，因为盘子、饼干盒和 ramekin 的分层厚度小于该容差。

还有一条标定结论值得知道：俯视共线的"反遮挡"预筛会拒绝 50 个真实场景中的 44 个，
而那些场景的渲染面积都在 494 px 以上。agentview 是**俯拍**的，xy 平面上的共线并不
构成像素遮挡。用渲染门限即可。

### 实测几何与手工常数的差异

`spatial_scene.OBJECT_RADIUS` / `FIXTURE_RADIUS` 是视觉估计，与网格实测差别明显：

| 物体 | 代码值 | 实测 |
|---|---|---|
| akita_black_bowl | 0.075 | **0.056** |
| plate | 0.100 | **0.069** |
| ramekin | 0.060 | **0.045** |
| cookies | 0.050 | 0.052 |
| flat_stove | 0.120（圆盘） | **0.297 × 0.190 长方体** |
| wooden_cabinet | 0.130（圆盘） | **0.313 × 0.283 长方体** |

灶台影响最大：它的 body **原点位于其轮廓中心之外 0.0965 m**，而"碗在灶台上"成立于
距该原点 0.146 m 处（灶眼）。木柜的偏航是 155°，所以必须用本体坐标系的 AABB。
实测桌面范围 x ∈ [-0.5, 0.5]、y ∈ [-0.6, 0.6]，台面 z = 0.900。

### 哪些配对可行

45 个任务组合中，6 个没能通过平面判据，因为两条关系所指的位置比互斥半径（0.24 m）更近：

| 配对 | 命名点间距 | 结论 |
|---|---|---|
| between + next_to_ramekin | 0.170 m | 歧义 |
| between + on_ramekin | 0.153 m | 歧义（见下） |
| between + next_to_plate | 0.133 m | 歧义 |
| next_to_ramekin + on_ramekin | 0.126 m | **可用——"在上"对"在旁"** |
| on_cookie_box + next_to_cookie_box | 0.117 m | **可用——"在上"对"在旁"** |
| on_cabinet + in_drawer | 0.151 m | 歧义（判据局限） |

**"在上"对"在旁"。** 其中两个根本不是歧义：对它们来说平面距离量错了东西——区分
"碗**在**蒸碗**上**"和"碗**在**蒸碗**旁**"的是**支撑**，不是位置。在锚物自身坐标系中
实测的命名点偏移：

| 关系 | 相对锚物的偏移 | 含义 |
|---|---|---|
| on_ramekin | 0.0115 m | 同心——搁在上面 |
| next_to_ramekin | 0.1218 m | 在旁边 |
| on_cookie_box | 0.000016 m | 同心 |
| next_to_cookie_box | 0.1166 m | 在旁边 |

两只碗最终仍相距 0.117–0.126 m，而实测碗半径为 0.056 m——它们并不接触。用
`screen_spatial_pairs.py --shared-anchor include|only` 启用：当一侧偏移同心
（< 0.05 m）而另一侧不同心时，该配对被接纳。

有两个条件是结构性的而非风格性的，判据对二者都做了强制：

- **共享锚物必须是自由物体。** `place_shared_group` 只放置锚物一次，再把两只碗挂在
  它上面，这需要一个关节来写入位姿。固定装置没有关节，于是两个独立抽样的组会各自
  携带一份灶台/木柜副本，最后写入的那一份会静默胜出。这正是 `on_cabinet + in_drawer`
  被排除的原因——尽管它的偏移（0.009 m 对 0.144 m）通过了支撑判据，*看起来*是可接纳的。
- **两条关系都不能指涉其他任何东西。** 只被其中一条关系指涉的锚物不在共享组内，会被
  当作独立物体散放，那条关系于是直接不成立。`between + on_ramekin` 里的盘子就会如此。

所以该判据恰好接纳两个配对，而不是四个。

其余 39 个在默认灶台 y 带下都能采样，唯一例外是 `from_table_center + on_the_stove`；
把该带放宽到 (-0.26, 0.10) 可达 93%，代价是其他灶台配对略有下降。采样成本跨越四个
数量级，从每场景 4 次到 133,000 次。

**反向盘子采样。** 当一条关系指涉盘子时，盘子随组移动，却仍须落在 0.18 × 0.16 m 的
可达放置区内。在 0.40 × 0.50 m 的工作区里抽碗，这是一个 1–6% 的事件，而这些配对
89–92% 的拒绝正是由此而来。改为在放置区内抽盘子、再按同一刚性偏移反推碗，探索的是
同一个可行集，只是从小的那一侧进入：含盘子关系的配对快 4–7 倍，其余配对逐位不变。

---

## 3. 训练 / 反事实 / 未见 划分

两条正交的轴：**摆放**（见过 vs 未见）与**指令**（训练过的 vs 另一个任务的）。

- **train** —— 目标落在 seen 格。
- **counterfact** —— *同一个几何*，另一个碗作目标，另一个任务的指令。图像逐像素相同，
  正确答案不同。这个集合用单任务数据造不出来。
- **unseen** —— 目标落在 unseen 格，取自未用于训练的几何。

格子来自工作区上的**共享 4×4 棋盘格**（格子 0.100 × 0.125 m），`(cx + cy)` 为奇数即
unseen。共享棋盘已经让每个任务的 seen 比例落在 41%–59% 之间，无需按任务单独划分。

两件必须做对的事：

**角色分配必须按任务平衡。** 若交由配对枚举顺序决定，`on_the_wooden_cabinet` 和
`in_the_top_drawer` 的训练格数为零（永远当搭档），而 `next_to_the_ramekin`、
`on_the_ramekin`、`between` 的反事实格数为零。

**格子必须按每条演示在抖动后重算。** 抖动是 ±0.02 m 而格子是 0.100 m，62% 的目标
距格线不到抖动幅度，沿用基准场景的标签会大面积污染划分。

提交前已验证可解性：反事实场景 94%、未见场景 97% 能产出可行轨迹，训练场景是 98%——
unseen 格并不系统性地更难。

---

## 4. 脚本

| 脚本 | 作用 |
|---|---|
| `screen_spatial_sources.py` | 逐任务源筛选（healthy / regrasp / bad） |
| `smoke_spatial_inits.py` | 逐任务渲染初始状态拼图，不跑轨迹 |
| `run_spatial_oc_demo.py` | 单关系生成（关系刚性组） |
| `run_all_spatial.sh` | 十个任务，可续跑 |
| `screen_spatial_pairs.py` | 配对标定、对照、证伪、产出率 |
| `plot_pair_distribution.py` | 逐配对摆放散点 |
| `plot_bowl_distribution.py` | 逐任务源 vs 生成的摆放对比 |
| `build_pair_split.py` | train / counterfact / unseen 划分方案 |
| `add_pair_episodes.py` | 向已有方案追加新配对的 episode |
| `probe_pair_trajectories.py` | 轨迹成功率与场景可解性 |
| `run_pair_oc_demo.py` | 按划分方案生成配对数据 |
| `run_all_pairs.sh` | 一个 split，多任务并行 |
| `patch_fixture_objects.py` | 给早于"生成时内联记录"的数据补上灶台和木柜 |
| `patch_regrasp_label.py` | 标注含"抓取失败后重抓"的演示 |

典型流程：

```bash
PY=.venv/Scripts/python.exe          # Linux 上为 .venv/bin/python

# 十个任务的单关系数据
$PY scripts/screen_spatial_sources.py
bash scripts/run_all_spatial.sh

# 配对数据
$PY scripts/screen_spatial_pairs.py --n 40            # 可行性 + 标定
$PY scripts/build_pair_split.py --train-scenes 25 --unseen-scenes 10
$PY scripts/probe_pair_trajectories.py --split train --n 10 --source-retries 6
bash scripts/run_all_pairs.sh train 5
$PY scripts/patch_fixture_objects.py                  # 仅用于内联记录之前的批次
```

按任务并行是安全的，因为每个任务写自己的 HDF5。本项目记录过的那个故障
（多 worker 叠加、Win32 错误 33）需要**同一文件**的争用才会发生。

### 给已经生成过的方案补配对

重跑 `build_pair_split.py` 会把所有场景重新编号，这会让磁盘上已有数据 scene_log 里的
`scene_index` 失效——而 `patch_fixture_objects.py` 和 `build_pair_eval_suite.py` 都会
解引用它。`add_pair_episodes.py` 改为只追加：新几何放在 `plan["scenes"]` 末尾，已有
索引仍指向同一几何，新 episode 带 `batch` 标签。

```bash
$PY scripts/screen_spatial_pairs.py --n 40 --shared-anchor only --merge
$PY scripts/add_pair_episodes.py --batch shared_anchor --train-scenes 10
$PY scripts/run_pair_oc_demo.py --split train --task on_the_ramekin \
    --batch shared_anchor --target-count 52
$PY scripts/build_pair_eval_suite.py --split counterfact
$PY scripts/build_pair_eval_suite.py --split unseen
```

`--batch` 同时改变进度的计数方式：hdf5 里已有的数据来自其他 episode，把它们算作新
episode 的进度会导致新 episode 被全部跳过。`--target-count` 让一个任务在 hdf5 达到
指定总数时停止，这才是"补齐"而不是"翻倍"。

---

## 5. 含 fixture 的物体顺序

指令本身指涉灶台和木柜，所以这两个作为物体记录，追加在五个自由物体**之后**，
60–100 的 id 保持不变：

| 序号 | 实例 | seg id |
|---|---|---|
| 0 | akita_black_bowl_1（目标） | 60 |
| 1 | plate_1（放置终点） | 70 |
| 2 | akita_black_bowl_2（干扰） | 80 |
| 3 | cookies_1 | 90 |
| 4 | glazed_rim_porcelain_ramekin_1 | 100 |
| 5 | flat_stove_1 | 110 |
| 6 | wooden_cabinet_1 | 120 |

`run_pair_oc_demo.py` 现在在**生成过程中**就记录它们（提取器从仿真里追加两个 fixture
的位姿，因为 robosuite 的逐物体观测只对带自由关节的 body 存在；同时写入 `drawer_pos` /
`drawer_qpos`）。传 `--no-fixtures` 可回到五物体布局。

最早的 485 条演示早于此，是事后由 `patch_fixture_objects.py` 补上的，下面两条注意事项
出自那里。那条路径不重新合成轨迹：每一帧的完整仿真状态都已存储，所以只需恢复状态
重新渲染。

- `env._get_observations()` 返回的是**缓存**观测，重渲染必须传 `force_update=True`。
  不加的话输出看起来合理，实际显示的是环境重置后的状态（掩码 IoU 为 0.0008 而非 0.9997）。
- fixture 位姿属于模型数据，且没有逐条演示记录。它由 frame-0 状态反推——抖动使一个组
  绕其碗做刚体运动，所以碗的 xy 给出平移、四元数给出偏航——并且每条演示都通过重渲染
  验证：原有五个物体的掩码必须与存储值一致，而 fixture 位置错误会通过遮挡破坏这一点。

抽屉是独立的 body（`wooden_cabinet_1_cabinet_top`），跟随 `wooden_cabinet_1_top_level`
滑动关节移动，但实例分割把整个柜子当作一个实例。要把抽屉单独分出来需要逐 geom
（"element"）分割，本文未做。
