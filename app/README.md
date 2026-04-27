# PGD-UNet Streamlit Inference App

Run from the project root:

```bash
streamlit run app/streamlit_app.py
```

Default model:

```text
app/checkpoints/best.pth
outputs/pgd_unet/kvasir_seg/unet_resnet152_teacher/loss_seg_kd_sparsity/output_kneedle_auto_no/2_pruning/artifacts/blueprint.json
```

If the copied checkpoint in `app/checkpoints` is not available, the app falls back to:

```text
outputs/pgd_unet/<dataset>/unet_resnet152_teacher/loss_seg_kd_sparsity/<output_folder>/3_student/checkpoints/best.pth
```

Upload an image, then the app returns the predicted binary mask, probability map, and overlay.
