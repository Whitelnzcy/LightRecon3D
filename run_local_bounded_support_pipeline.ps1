param(
    [ValidateSet("export-smoke", "export-train128", "train-v1")]
    [string]$Stage = "export-smoke",
    [int]$Steps = 1200
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ExportPython = "E:\anaconda\envs\ai\python.exe"
$TrainPython = "E:\anaconda\python.exe"
$Dataset = Join-Path $Root "data\Structured3D"
$Dust3RWeights = Join-Path $Root "checkpoints\DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
$OutputRoot = Join-Path $Root "outputs\learned_plane_params_v1"
$TeacherDir = Join-Path $OutputRoot "gt_support_teacher_train128_local_v1"

if ($Stage -eq "export-smoke") {
    & $ExportPython (Join-Path $Root "export_gt_support_teacher_npz.py") `
        --root_dir $Dataset `
        --weights_path $Dust3RWeights `
        --output_dir (Join-Path $OutputRoot "gt_support_teacher_smoke_local_v1") `
        --split train `
        --start_idx 0 `
        --count 1 `
        --max_planes 8 `
        --max_points 24000
    exit $LASTEXITCODE
}

if ($Stage -eq "export-train128") {
    & $ExportPython (Join-Path $Root "export_gt_support_teacher_npz.py") `
        --root_dir $Dataset `
        --weights_path $Dust3RWeights `
        --output_dir $TeacherDir `
        --split train `
        --start_idx 0 `
        --count 128 `
        --max_planes 8 `
        --max_points 24000
    exit $LASTEXITCODE
}

& $TrainPython (Join-Path $Root "train_bounded_plane_support_head.py") `
    --input_dir $TeacherDir `
    --pattern "*.npz" `
    --output_dir (Join-Path $OutputRoot "bounded_support_head_train128_local_v1") `
    --save_checkpoint (Join-Path $OutputRoot "bounded_support_head_train128_local_v1\train128.pt") `
    --num_planes 8 `
    --max_points_per_sample 18000 `
    --patch_pixel_size 0.08 `
    --min_patch_points 12 `
    --steps $Steps `
    --sample_batch_size 4 `
    --hidden_dim 192 `
    --lr 0.0007 `
    --weight_decay 0.0001 `
    --teacher_weight 1.0 `
    --smooth_weight 0.06 `
    --boundary_weight 0.18 `
    --hard_boundary_weight 0.80 `
    --hard_boundary_min_edge_conf 0.25 `
    --residual_weight 0.04 `
    --line_smooth_suppress 0.60 `
    --boundary_margin 0.10 `
    --boundary_error_weight 3.5 `
    --boundary_error_min_edge_conf 0.25 `
    --support_logit_weight 0.15 `
    --boundary_support_logit_weight 0.05 `
    --outside_support_penalty 0.40 `
    --support_grid_size 0.055 `
    --support_dilate_cells 2 `
    --support_min_label_conf 0.60 `
    --log_every 100 `
    --seed 20260611
exit $LASTEXITCODE
