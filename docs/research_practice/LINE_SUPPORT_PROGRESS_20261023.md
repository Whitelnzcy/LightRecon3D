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
