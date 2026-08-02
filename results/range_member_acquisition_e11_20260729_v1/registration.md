# LightRecon3D 快速获取任务：远程 ZIP 成员级 Range 提取

## Registration ID

```text
E11_PREREG_RANGE_MEMBER_ACQUISITION_20260729_v1
```

## 执行授权

```text
execution_authorized = true
```

本任务收到后立即执行，不再生成未授权回执，不下载 14 个完整
perspective archives。

## 1. 唯一目标

使用已经冻结的 128 个候选和远程 ZIP central directory，只下载这些
候选场景实际需要的 perspective-empty RGB 成员，并建立不少于 77 个
Clean Raw scenes。`Structured3D_annotation_3d.zip` 允许完整下载并永久
保留。本轮不得运行 DUSt3R、Stage1、B/M、训练或评价。

## 2. 采用的冻结上游

只读采用并重新验证 E9 的 `candidate_order_128.json`、inventory 与
disqualifying scene 集合、README 与 errata 内容 hash、两套 remote
probe 和 remote member indexes。必须逐文件验证 SHA-256。任何 hash
不一致判定 `Blocked_Upstream_Mismatch`，不得重新选择候选。

## 3. Source domains

三个 domain 独立：

* P：`perspective_empty_01`–`14`，候选 RGB 来源；
* PB：`perspective_empty_00`、`15`、`16`、`17`，只做边界验证；
* A：`Structured3D_annotation_3d.zip`，sealed annotation 来源。

要求 128/128 候选在 P 中恰好出现一次，在 PB 中出现零次，在 A 中各
存在唯一 `annotation_3d.json`。Annotation 不得参与 perspective
member-path 冲突检测。不满足判定 `Blocked_Source_Mapping`。

## 4. 下载前成员清单

根据 central directory 生成唯一 `range_member_manifest.json`。每个
成员记录 scene ID、archive URL、normalized member path、local-header
offset、compression method、compressed/uncompressed size、CRC32、
target path、source domain、archive total length、central-directory
SHA-256 和 strong ETag（若有）。

仅把 P 域中以 `/rgb_rawlight.png` 结尾的非目录成员定义为本轮
perspective-empty RGB 成员；这是冻结索引中唯一的 RGB 文件类型。

先统计 selected member count、compressed/uncompressed bytes、
`full_archive_bytes = 163154566636` 和 download reduction ratio，随后
继续执行，不等待人工确认。

## 5. 远程对象复核

Range 下载前重新验证涉及 archive 的 final URL、total length、fixed
head/tail hash、central-directory hash 和 strong ETag（若提供）。硬
身份变化判定 `Blocked_Remote_Object_Changed`。Content-Length 响应头
是否出现、Last-Modified 或弱 ETag 变化只记录。

## 6. ZIP 成员级 Range 下载

每个目标 member 必须 Range 读取并验证 local header，计算数据起点，
读取恰好 compressed size 字节并严格验证 Content-Range；记录压缩
SHA-256；按 method 0 或 8 解压；验证长度、CRC32 和文件 SHA-256；写入
staging，fsync 后原子移动并更新 per-member receipt。其他 compression
method 判定 `Blocked_Unsupported_Compression`。禁止完整 perspective
archive fallback。

## 7. 断点恢复

保存 `member_download_state.json`，状态粒度到单个 member。已完成且
CRC/SHA-256 验证通过的成员跳过；未完成成员从头下载；不做成员内部
续传。网络中断判定 `Paused_Resumable`，再次运行继续剩余成员。

## 8. Annotation 获取

完整下载 `Structured3D_annotation_3d.zip` 到
`D:\Structured3D_official_archives_e11\annotations\`，支持安全断点
续传，验证完整长度、whole-file SHA-256 和 central directory，并永久
保留。只向
`F:\Structured3D_clean_e11\sealed_annotations\scene_xxxxx\` 提取 128
个候选的 `annotation_3d.json`。只允许计算路径、size、CRC32 和
SHA-256，禁止解析 JSON 内容。

## 9. 输出位置

Perspective RGB：
`F:\Structured3D_clean_e11\raw_rgb\scene_xxxxx\`

Annotation：
`F:\Structured3D_clean_e11\sealed_annotations\scene_xxxxx\`

Staging：
`F:\Structured3D_clean_e11\staging\`

Manifest：
`E:\Study\code\LightRecon3D\results\range_member_acquisition_e11_20260729_v1\`

E 盘不得存储图像或压缩成员。

## 10. 磁盘门槛

开始前要求：

```text
F_free >= selected_uncompressed_bytes + 32 GiB
D_free >= annotation_archive_bytes + 5 GiB
E_free >= 1 GiB
```

运行中 F_free 小于 24 GiB 时完成当前文件后判定
`Paused_Low_Disk`。

## 11. Clean Raw 资格

场景必须属于冻结 128 候选且不属于历史 disqualifying 集合；P/A 映射
唯一；所有选定 RGB 成员完成并具有 compressed SHA-256、CRC32、
uncompressed size 和基本图像解码验证；annotation 已 sealed 并 hash；
provenance 完整；未读取 GT 语义；未运行模型或评价。满足后标记
`Clean_Raw`。不得按图像内容或未来模型质量筛选。

## 12. Ready 门槛

`Clean_Raw >= 77` 判定 `Ready_Clean_Raw_Pool`，保留所有成功场景。
不足 77 判定 `Blocked_Insufficient_Clean_Raw_Scenes`，不得补候选、改
salt 或使用 183 个未知场景。

## 13. 禁止调用

DUSt3R/Stage1 input preparation、global-cache、direct-support、B/M
inference/training、checkpoint selection、calibration、evaluator、GT
matcher、Plane metrics、expansion、refit、plane fit、RANSAC、GT
semantic parsing 的调用计数必须全部为零。本轮不冻结输入准备流程。

## 14. 工程错误处理

不改变候选、member 清单、远端身份规则或 Clean Raw 定义的 HTTP
重试、Windows 路径、日志、临时文件清理和 ZIP header 解析实现错误，
允许在同一 Registration 下修复；每次修复须独立 commit、记录 diff 与
source hash 并运行测试。修改候选、source domain、scene 资格、GT
禁令、远端身份或评价用途必须停止并新注册。

## 15. 返回文件

```text
docs/codex_returns/CODEX_RETURN_range_member_acquisition_e11_20260729.md
```

返回固定为 12 节：Upstream verification、Source-domain mapping、
Selected-member download estimate、Remote identity verification、Range
download progress、Annotation archive、Extraction and integrity、Clean Raw
pool、Forbidden-call audit、Verdict、Mechanical next action、Artifacts。

若 Ready，下一动作只能写：

> 已建立不少于 77 个 Clean Raw scenes。停止，等待新的输入物化任务。

## 最终机械规则

不下载 14 个完整 perspective archives。使用远程 central directory
定位冻结 128 个 candidate 的成员字节区间，通过 HTTP Range 逐 member
下载、解压、CRC 和 SHA-256 验证；annotation archive 完整下载但保持
sealed。Clean Raw 达到 77 即 Ready。
