# 线关系引导支撑精修：执行记录

日期：2026-10-23。任务：`docs/codex_tasks/line_conditioned_support_refinement_20261023.md`。
分支：`codex/line-support-refinement-20261023`。
执行前 HEAD：`e2c058fb9727b06127cf966f8a5bb34f8f8374f5`。

## 本轮做了什么

已完成第一个可复核里程碑：缓存清查、旧预测逐点复算、GT和坐标核对、预测驱动的局部样本接口。
没有训练新模型，没有修改三维坐标、点数、候选参数或旧结果。下列改善数字全部来自**历史模型输出的重新核查**，不是新模型收益。

新增文件：

- `audit_line_support_refinement.py`：非破坏性核查入口，拒绝覆盖输出目录。
- `models/line_support_patches.py`：无GT的坐标变换、RGB采样、patch提出、交线坐标、重叠logit融合。
- `tests/test_line_support_patches.py`：10项接口测试。
- 同步已有必要依赖 `recover_cache_projection.py` 及其测试；该依赖本轮没有修改。

本轮输出仅在本地：`local_outputs/line_support_audit_20261023_v1/`。
包含 `protocol.json`、`data_manifest.json`、`baseline_diagnosis.json`、`completion.json`、
两份 `prediction_inputs/*.json`、10张 `figures/*.png`。
GitHub 只同步代码、测试、计划和本记录；不上传数据集、模型、NPZ或原始场景图片。

## 已验证的实际数据流

Stage2 writer 为 `export_stage2_learned_region_merge_editables.py::export_one`，保留像素/视图元数据。
Stage3 reader/mapper 为 `export_stage3_scene_plane_fusion.py`，现有路径确实调用
`make_pairs -> inference -> global_aligner -> compute_global_alignment -> scene.get_pts3d()`。
已阅读 vendored DUSt3R API 与 resize/crop 实现。本轮复用全局点缓存，不重跑全局对齐，也不修改Stage2/Stage3。

本轮预测输入来自 guided_ransac NPZ，显式验证 `pixel_coordinate_order=xy`、
`pixel_coordinate_space=dust3r_aligned_pointmap`。GT、旧预测与原点云的视图、像素、行号逐一相等；
GT附带的XYZ与预测XYZ完全一致。原候选文件SHA-256与历史推理凭据一致。

旧GT文件中的预处理元数据通过白名单导出，重新核对原图尺寸与DUSt3R裁剪规则。
输出预测输入不包含GT标签、layout路径或GT相机。相机投影仅从预测XYZ/像素恢复，
采用不相交的拟合/核查点；10视图重投影P95为0.1085–0.1316 px。
这验证了全局缓存与图像映射的一致性，不等于测得真实重建精度。

标签来自 `layout.json` 的 `visible_mask`；本轮全部图像为 `perspective/empty`。
因此不能将这批实验描述为家具遮挡下的可见实物表面去噪。
重新渲染的稠密GT与原评价缓存完全一致，5px欧氏边界距离也逐点一致。

## 数值复核

对两个方法分支 finite_control、finite_support 的全部历史字段均进行了精确断言。
finite_support 结果如下：

| 场景 | 原总错误 | 旧模型后总错误 | 原边界错误 | 旧模型后边界错误 | 改正 / 误改 |
| --- | ---: | ---: | ---: | ---: | ---: |
| scene00199 | 65,351 | 18,941 | 18,608 | 11,975 | 47,099 / 689 |
| scene00190 | 243,207 | 194,087 | 45,388 | 40,039 | 56,518 / 7,398 |

两场景合计边界错误63,996 -> 52,014，下降18.72%；这仍是历史开发结果。

旧模型剩余错误被划为互斥类别：

| 场景 | 固定映射未覆盖身份 | 有身份但原未分配 | 有身份、距GT边界≤5px | 有身份、距GT边界>5px |
| --- | ---: | ---: | ---: | ---: |
| scene00199 | 0 | 0 | 11,975 | 6,966 |
| scene00190 | 65,592 | 1,390 | 31,707 | 95,398 |

scene00199仅5个剩余错误距GT边界超过20px；适合局部边界精修的开发诊断。
scene00190仍有21,940个“已匹配且已分配”的错误距GT边界超过32px。
64x64局部窗口是否足够，需要实际覆盖与上下文实验，不能直接假定窄边界精修能解决。

scene00190固定映射没有覆盖的两个GT身份为17和34，共65,592点。
它们并非没有重建点：GT17的点主要被候选0/1占有，GT34主要被候选2占有。
因此只能称“固定身份映射未覆盖”，不能据此断言对应物理平面不存在；需要进一步区分合并与缺失。
这些点保持计入完整指标，不能删除或只报告可修复子集。

补充指标：固定身份、共同观测域、5px内侧带的Boundary IoU变体：
scene00199为0.306452 -> 0.548471，scene00190为0.100948 -> 0.156050。
稀疏缺口不生成伪边界，图像外框计入边界。它不是官方重匹配Boundary IoU，
不能直接与论文表中的Boundary IoU数字比较；正式实验仍需统一协议。

## 数据清查与风险

在 `local_outputs/*/items/*/guided_ransac/*.npz` 找到21个场景、21套缓存。
相对冻结RGB边界网络的历史协议：4个属于训练、8个属于开发、9个属于历史测试。
没有不在该历史协议中的候选缓存。历史测试不等于训练泄漏，但此前已评估/查看过，不能再次当作全新盲测。
当前原始 `data/Structured3D` 有200个场景；RGB训练协议含180训练、10开发、10历史测试场景。
这不是整个磁盘所有历史缓存的穷尽清查，也没有证明上游平面网络的完整训练交集。

所以：旧4场景可用于实现拟合检查，旧199/190可用于开发诊断；正式测试集尚未落实。
不要把旧DEV换名为TEST。后续需要审计上游划分，选择真正未用于调参的数据，或明确重训/新缓存范围。

## P1接口试验

- 每视图6个预测标签边界patch＋2个普通区域patch，固定种子、64x64，共80个。
- patch只由预测标签提出，未使用GT错误、GT边界或交线方向。三个实验组可以共享同一索引。
- 候选池暂为全部原有场景候选；不悄悄排除难例。
- RGB按显式半像素逆变换做双线性采样，不宣称与DUSt3R的Lanczos预处理逐字节相同。
- 交线齐次符号规范化；候选轴顺序显式；重叠patch汇总logit后一次决策，原未分配/未覆盖点保持原值。
- 当前只建立接口与诊断索引，未生成完整训练张量、未定义空间网络、未运行M1/M2/M3。

## 实际执行与检查

环境：现有 `E:/anaconda/envs/ai/python.exe`，没有安装新依赖。

```text
python -m unittest discover -s tests -p test_line_support_patches.py -v
python audit_line_support_refinement.py --output local_outputs/line_support_audit_20261023_v1
python -m py_compile audit_line_support_refinement.py models/line_support_patches.py tests/test_line_support_patches.py recover_cache_projection.py
python -c "import runpy; runpy.run_path('tests/test_recover_cache_projection.py',run_name='__main__')"
```

10项接口测试和2项投影检查通过，语法检查通过；核查程序exit 0。10张图生成，人工抽查两个场景view0：
共同域与颜色一致，展示历史改正/误改和预测边界选出的局部放大。
蓝色为所有预测平面对交线，不代表它们都是真实接触边界；复杂场景中的多余线正是需要学习适用范围的动机，
但还不能据此证明学习门控一定有效。

## 下一步（尚未执行）

1. 完成上游Stage1训练场景/建筑划分审计与最近邻方法机制对照。
2. 接入旧4个训练场景，建立输入/监督分开的完整patch数据集，统计采样覆盖，不按GT挑patch。
3. 实现普通空间精修M1，先小样本拟合与显存/耗时测量，再进行同预算M2/M3比较。
4. 在独立测试数据未落实之前，不报告新方法泛化，也不启动大规模缓存/训练。

仓库安全：原有241项删除、7项修改及其他未跟踪历史文件不纳入本次提交；
CURRENT_STATE只新增本轮段落，不覆盖此前记录。不得使用全量git add或强制推送。

## 第二轮执行：空间基线与首轮线对照（已完成）

基于已同步提交 `16836b104fda0ee68632ce72fb3adb66c4cbbad5` 继续实施。
实际完成，不是计划：

- 接入scene00000–00003，20视图，合计2,711,687点；每视图96个GT无关64x64训练patch，共1,920个。
- 推理采用64px窗口、48px步长覆盖全视图，重叠logit平均后一次分配；不按GT选择推理区域。
- 输入/监督分开：RGB、原始候选mask、几何残差先建立；随后读取布局GT并做一次场景级身份匹配。
- 固定XYZ、点序、候选平面；未分配点保持原值。缺失匹配身份不参与归属损失，但始终计入完整评价。
- 实现35,545参数的普通空间精修M1；再实现同初始化、同样本序列、同损失的M1/M2/M3对照。

新入口与模块：
`spatial_support_data.py`、`models/spatial_support_refiner.py`、
`train_spatial_support_pilot.py`、`evaluate_spatial_support_dev.py`、
`spatial_line_features.py`、`models/spatial_line_refiner.py`、
`train_spatial_line_ablation.py`、`audit_spatial_line_pilot.py`。
同步已有必要依赖 `models/ownership_supervision.py`，未修改其历史匹配方法。

### 第一阶段M1小拟合

输出：`local_outputs/spatial_support_m1_train4_20261023_v1/`；固定200步、batch4、seed42、AdamW lr1e-3。
损失：原本正确点2倍CE +0.5 present-class Dice。探针损失2.393909 ->1.369852。
四场景总错误829,570 ->748,464；边界错误164,757 ->158,297；改正94,061、误改12,955。
这些是训练场景拟合，不是泛化。输入/候选顺序不变性及空间编码器梯度均验证。

RTX4060Ti8GB：训练循环6.856秒，约0.0343秒/步；PyTorch峰值分配271,697,920字节（约259 MiB）。
上述不含读取、准备、评价和CUDA上下文开销，不等于整卡显存占用或大规模训练预算。
权重SHA-256：`95a6a068a1e247caf2f555adacfec2f2b17565068ea04731326eb782c12c4378`。

冻结M1后，仅一次旧DEV2评价：`local_outputs/spatial_support_m1_dev2_20261023_v1/`。
总错误308,558 ->268,231，边界63,996 ->62,705。scene190边界45,388 ->45,412，略有退化。
没有据此挑选checkpoint或调学习率/训练步数。

### 同预算M1/M2/M3

输出：`local_outputs/spatial_line_m123_train4_dev2_20261023_v1/`。

- M1：线特征置零，线贡献关闭。
- M2：候选平面对条件化的交线特征，固定0.5作用权重。
- M3：相同特征，逐位置、逐候选学习sigmoid权重，初始0.5。
- 线权重实际乘在候选竞争logit的线分支项上；不是只画一张热图。
- 六个线特征为：当前/目标平面对可用性、朝目标面的有符号距离、距离衰减、沿线坐标、两侧分离置信、原交线修正得分。
- 使用已预测的原平面交线和图像先验修正结果，不输入GT交线；2D修正不保证多视图3D一致性。
- 每组200步，batch4、相同初始state_dict、相同采样、相同损失和优化器，不在DEV选择模型。
- 每组实例化46,107参数，但M1额外分支关闭、M2门控未使用；**有效容量不完全相同**，需要线无关的活动分支对照。不能把全部增益直接归于线语义。

完整TRAIN4指标（训练拟合）：

| 方法 | 总错误 | 边界错误 | 改正 | 误改 |
| --- | ---: | ---: | ---: | ---: |
| 原始 | 829,570 | 164,757 | 0 | 0 |
| M1 | 748,465 | 158,297 | 94,060 | 12,955 |
| M2 | 654,724 | 144,314 | 208,507 | 33,661 |
| M3 | 633,270 | 139,392 | 245,892 | 49,592 |

完整历史DEV2指标（冻结迁移，非盲测）：

| 方法 | 总错误 | 边界错误 | 改正 | 误改 |
| --- | ---: | ---: | ---: | ---: |
| 原始 | 308,558 | 63,996 | 0 | 0 |
| M1 | 268,232 | 62,705 | 46,587 | 6,261 |
| M2 | 224,857 | 56,362 | 100,932 | 17,231 |
| M3 | 206,657 | 51,831 | 119,319 | 17,418 |
| 历史finite（参考，预算/输入不同） | 213,028 | 52,014 | 103,617 | 8,087 |

独立M1和匹配对照M1总错误相差1点；CUDA浮点路径有微小差异，并未设置确定性算法模式。
不能宣称逐位确定性；正式阶段需多seed、确定性设置说明与场景级统计。

M3相对原始：总错误下降33.02%、边界下降19.01%，改正/误改约6.85。
M3相对M2：总错误减少18,200、边界减少4,531、误改增加187。
M3相对历史finite：总错误减少6,371，但边界只减少183，误改增加9,331。
因此本轮**不替换历史主线**，不宣称已经解决边界问题、达到20%最终验收或优于Plane-DUSt3R。

三组训练循环合计25.940秒，峰值分配292,827,648字节（约279 MiB）；不含数据与评价。
门控第三步已有非零梯度。DEV门控均值0.78186/0.78555（重复覆盖的tile像素统计），
不能仅凭均值证明学到了有限作用范围；需要加入全局可学习标量权重对照，排除整体强度调节。
当前模块是可学习的局部线作用权重，不是已恢复有限3D交线端点的模型。

M3权重SHA-256：`8468bfc009ab7cda0780ef5282a9be8ee2703ac36d30cda31f8fe0809f7718c8`。
M1/M2哈希、模型/数据/代码凭据见各输出的protocol、completion及prediction_inputs。

### 事后只读核查

`audit_spatial_line_pilot.py`已执行，写出上述对照目录的`postrun_audit.json`。
训练230条视图内交线记录的原始提议、修正前系数、候选源文件路径与SHA均核对通过；
坐标为显式aligned-pointmap xy整数中心。DEV三组指标独立逐点重算完全一致，未分配点没有被修改。

M3误改分布：

| 场景 | 距GT边界≤5px | 5–20px | >20px |
| --- | ---: | ---: | ---: |
| scene00199 | 2,617 | 282 | 0 |
| scene00190 | 4,026 | 4,367 | 6,126 |

scene190误改主要集中在候选2→6（4,388点）、2→4（4,235点）、2→0（1,500点）。
这是开发集诊断，不能在推理中依据这些GT统计直接禁止特定编号或裁剪区域。
下一步在TRAIN上寻找可学习的对应证据并做受控实验，而非继续盲目加训练步数。

### 执行命令与测试

```text
python train_spatial_support_pilot.py --output local_outputs/spatial_support_m1_train4_20261023_v1 --steps 200 --batch 4
python evaluate_spatial_support_dev.py --checkpoint local_outputs/spatial_support_m1_train4_20261023_v1/last.pt --output local_outputs/spatial_support_m1_dev2_20261023_v1
python train_spatial_line_ablation.py --output local_outputs/spatial_line_m123_train4_dev2_20261023_v1 --steps 200
python audit_spatial_line_pilot.py --run local_outputs/spatial_line_m123_train4_dev2_20261023_v1 --output local_outputs/spatial_line_m123_train4_dev2_20261023_v1/postrun_audit.json
python -m unittest discover -s tests -p "test_spatial*.py" -v
python -m unittest discover -s tests -p test_line_support_patches.py -v
```

四个执行入口均exit0。共12张结果图（M1 TRAIN4、M1 DEV2、三组TRAIN4/DEV2）；
最终18项单元测试（8项空间/线模型＋10项坐标接口）、2项投影检查全部通过；新增模块和入口语法检查通过。
已查看M1 scene00003/00199以及三组两个DEV视图0，均为同点域、同颜色的二维投影，不是移动点坐标的重建图。
近邻方法机制对照、上游全部数据划分、独立测试、容量对照、全局标量门控和多seed仍未完成。

## 第三轮：活动容量、全局标量及冻结门控干预

基于提交`a2f562b`继续；仍是4个训练场景和2个历史开发场景，不是独立测试。本轮不改几何、损失、数据、学习率、patch或步数。

### 实现与实验约束

- MC：与M3完全相同的活动分支结构/参数数目，六个额外通道重复既有RGB、观测mask、候选mask、几何残差，不输入任何线特征。额外贡献在观测像素上开启，门控参与训练；测试验证对线输入任意替换不变。
- MS：相同线特征和线分支，仅将逐位置门控换成一个sigmoid标量（复用gate bias，初始0.5），其余gate权重不参与计算。
- M3：原逐候选、逐位置门控重跑。每组200步，batch4，64x64，seed42，相同初始化/批次/损失/优化器；未根据DEV挑选checkpoint。
- 三组仍实例化46,107参数。MC与M3活动结构相同；MC与M3的作用mask不同，这是无线路径与线条件化方法的输入/机制差别，不宣称所有计算相同。MS只有一个活动gate参数。
- 保留旧checkpoint结构、旧默认M1/M2/M3入口；新增`--arms`。本轮新目录，未覆盖旧结果。

输出：`local_outputs/spatial_line_controls_train4_dev2_20261023_v1/`。

| 历史DEV2 | 总归属错误↓ | 5px边界错误↓ | 改正↑ | 误改↓ |
| --- | ---: | ---: | ---: | ---: |
| 原始 | 308,558 | 63,996 | 0 | 0 |
| MC：活动容量匹配、无交线 | 254,367 | 62,609 | 65,399 | 11,208 |
| MS：全局可学习线权重 | 222,440 | 55,895 | 103,302 | 17,184 |
| M3：空间线门控 | 206,657 | 51,831 | 119,319 | 17,418 |

M3相对MC少10,778边界错误（17.21%）；相对MS少4,064（7.27%），误改多234。
MS最终标量0.542879，M3均值约0.78，短训练下仍有优化速度/强度差别。因此追加下面的冻结干预，不直接把此表解释为纯粹空间作用范围增量。

逐场景（总错误/边界错误/误改）：

| 场景 | MC | MS | M3 |
| --- | --- | --- | --- |
| scene00199 | 34,832 / 17,646 / 2,984 | 22,728 / 14,278 / 2,784 | 16,848 / 12,151 / 2,899 |
| scene00190 | 219,535 / 44,963 / 8,224 | 199,712 / 41,617 / 14,400 | 189,809 / 39,680 / 14,519 |

TRAIN拟合总错误/边界错误/误改：MC695,952/152,833/23,966；MS652,250/144,010/35,740；M3633,271/139,392/49,593。
M3两个DEV逐点指标与前轮一致；TRAIN比前轮多1错误点，未启用确定性算法，不能声称逐位复现。

### 冻结门控干预：不要把均值变大误写成定位能力

`diagnose_spatial_gate.py`读取冻结M3，用TRAIN4全部视图的可用线tile像素统计均值0.785677892。重叠tile重复计数，口径与推理一致；未读取TRAIN GT作标定，不使用DEV调常量。
保持线分支与其他网络参数不变，仅将局部门控替换为此常量。预测保存后才读取DEV评价目标；原checkpoint哈希不变。

| 冻结模型干预 | 总错误↓ | 边界错误↓ | 改正↑ | 误改↓ |
| --- | ---: | ---: | ---: | ---: |
| M3 + TRAIN平均常量门控 | 217,184 | 54,505 | 104,905 | 13,531 |
| M3原空间门控 | 206,657 | 51,831 | 119,319 | 17,418 |

空间门控少2,674边界错误（4.91%），但多3,887误改。两DEV场景边界均改善，不能据此忽略代价。
常量版本scene00199/00190分别为总错误21,284/195,900、边界13,722/40,783、误改2,670/10,861。
TRAIN gate标准差0.075–0.083，DEV190为0.08375，说明不是严格常数；但最小门控仍约0.47–0.48，并未显示明确的“区域外关闭”。
这只是已训练模型的门控依赖诊断，不是从头训练到收敛的常量权重基线，也不能证明已经学到有限线段端点。

### 可视化、检查和运行成本

六张对比图保持同点域/相机/颜色，局部窗口来自预测边界而非GT错误。已查看两个DEV视图0：scene199的天花板交界更接近GT但仍不平滑；scene190仍有明显内部错分及碎片，不能仅用总指标掩盖。
`visualize_spatial_projection_audit.py`额外生成两个DEV视图0的全局XYZ投影图（合为一张），显式源索引/像素坐标对齐、来源SHA检查。互斥DLT验证点的P95为0.1230/0.1085px；这是坐标一致性，不是真实三维精度改善。

三组训练循环共39.520秒（不含准备/评价）；峰值allocated292,832,256字节=279.27MiB，reserved520,093,696字节=496MiB；不是总显卡占用，不是大规模训练。
新增两项单元测试后，空间相关10项、坐标/patch10项和投影2项通过；本轮6个Python文件语法检查通过。
`postrun_audit.json`再次复核230条训练线来源、所有DEV指标、未分配点及行索引，全部通过。没有运行因旧归档删除而不完整的全仓测试集。

本轮checkpoint SHA-256：

- MC：`a489c388b7e99ca3e9442ca22a0825e077dc4183a6ea2bbb0dbd08fd482c292b`
- MS：`2d1d9275003981b71dae6ab35f29bb6b78e08d462c2b8f768795bf3bfe3b9702`
- M3：`2d60d7406173376789cff67409696e3be583dd0a468b9c803fbf2fbbc31e036e`

执行命令（均exit0）：

```text
python train_spatial_line_ablation.py --output local_outputs/spatial_line_controls_train4_dev2_20261023_v1 --steps 200 --arms MC MS M3
python audit_spatial_line_pilot.py --run local_outputs/spatial_line_controls_train4_dev2_20261023_v1 --output local_outputs/spatial_line_controls_train4_dev2_20261023_v1/postrun_audit.json
python visualize_spatial_projection_audit.py --run local_outputs/spatial_line_controls_train4_dev2_20261023_v1 --output local_outputs/spatial_line_controls_train4_dev2_20261023_v1/projection_check
python diagnose_spatial_gate.py --run local_outputs/spatial_line_controls_train4_dev2_20261023_v1 --output local_outputs/spatial_line_controls_train4_dev2_20261023_v1/gate_intervention
python -m unittest discover -s tests -p "test_spatial*.py" -v
python -m unittest discover -s tests -p test_line_support_patches.py -v
```

结论：本轮初步支持线条件化分支及空间门控有增量，但误改明显多于历史finite（17,418 vs 8,087），不替换主线。下一步先做64/128上下文单因素对照，并在TRAIN定位保护失败；不因显存空闲而同时加网络、数据、训练步数。独立数据、上游划分、最近邻比较及多seed仍待完成。
