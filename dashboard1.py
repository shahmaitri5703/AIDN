import streamlit as st
import torch
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import io
import zipfile
from pathlib import Path
import time
from io import BytesIO
import torch.nn.functional as F


from base.baseTrainer import load_state_dict
from models import get_model
from utils import util
from dataset.torch_bicubic import imresize



# ──────────────────────────────────────────────── CONFIG ────────────────────────────────────────────────
class Config:
    # network / arch (from AIDN paper)
    rgb_range = 1.0
    arch = 'InvEDRS_arb'
    up_sampler = 'sampleB'
    down_sampler = 'sampleB'
    n_resblocks = 16
    n_feats = 64
    fixed_scale = False
    scale = 4
    rescale = None
    n_colors = 3
    res_scale = 1.0
    quantization = True
    quantization_type = 'round_soft'
    K = 4           # max scale factor
    num_experts_SAconv = 4
    num_experts_CRM = 8
    jpeg = False    # this flag is for training; here we only simulate JPEG at test time
    base_resolution = 4


    # path to official AIDN checkpoint
    model_path = "LOG/DIV2K/pre-train/AIDN.pth.tar"



cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ──────────────────────────────────────────────── MODEL ────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔥 Loading AIDN model…")
def load_model():
    if not Path(cfg.model_path).is_file():
        st.error(f"❌ Model missing: {cfg.model_path}")
        st.stop()
    model = get_model(cfg, None).to(device)
    ckpt = torch.load(cfg.model_path, map_location=device)
    load_state_dict(model, ckpt["state_dict"])
    return model.eval()



model = load_model()



# ──────────────────────────────────────────────── METRICS & UTILITIES ────────────────────────────────────────────────
def tensor_to_uint8(t, enhance=False):
    """
    t: (1, C, H, W) or (C, H, W) torch tensor in [0,1] or [0,rgb_range]
    """
    if t.dim() == 4:
        t = t.squeeze(0)
    img = util.tensor2img(t)  # returns float [0,1] numpy HxWx3
    if enhance:
        # only for visualization of embedded LR; not used in any metric
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 2.8
        img = np.clip(img, 0, 1)
    return (img * 255.0).clip(0, 255).astype(np.uint8)



def apply_jpeg(img_np, quality):
    pil = Image.fromarray(img_np)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf))



def to_y_channel(img_uint8):
    """
    Convert RGB uint8 image to Y channel (float32, [0,255]).
    Used to approximate the paper's Y-channel PSNR/SSIM.
    """
    img = img_uint8.astype(np.float32)
    # OpenCV expects BGR
    bgr = img[:, :, ::-1]
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]
    return y



def crop_border(img, s):
    """
    Crop ceil(s) pixels on each border, as in the paper.
    img: HxW or HxWxC numpy
    """
    h, w = img.shape[:2]
    b = int(np.ceil(s))
    if h <= 2 * b or w <= 2 * b:
        return img
    return img[b:h-b, b:w-b, ...]



def calc_psnr(a, b, use_y=False, scale=1.0, crop=True):
    """
    a, b: uint8 images, same shape.
    use_y: if True, compute PSNR on Y channel, otherwise on RGB.
    crop: if True, crop ceil(scale) on borders (Y-channel style).
    """
    if use_y:
        ay = to_y_channel(a)
        by = to_y_channel(b)
        if crop:
            ay = crop_border(ay, scale)
            by = crop_border(by, scale)
        diff = ay.astype(np.float32) - by.astype(np.float32)
    else:
        if crop:
            a = crop_border(a, scale)
            b = crop_border(b, scale)
        diff = a.astype(np.float32) - b.astype(np.float32)


    mse = np.mean(diff ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))



def calc_ssim(a, b, use_y=False, scale=1.0, crop=True):
    """
    Simple SSIM approximation (global) + optional Y-channel & border crop.
    For exact paper numbers, you would replace this by a window-based SSIM.
    """
    if use_y:
        ay = to_y_channel(a)
        by = to_y_channel(b)
        if crop:
            ay = crop_border(ay, scale)
            by = crop_border(by, scale)
        x = ay.astype(np.float32) / 255.0
        y = by.astype(np.float32) / 255.0
    else:
        if crop:
            a = crop_border(a, scale)
            b = crop_border(b, scale)
        x = a.astype(np.float32) / 255.0
        y = b.astype(np.float32) / 255.0


    mu1, mu2 = np.mean(x), np.mean(y)
    sigma1_sq, sigma2_sq = np.var(x), np.var(y)
    sigma12 = np.mean((x - mu1) * (y - mu2))
    c1, c2 = 0.01**2, 0.03**2
    return (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2) / (
        (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    )



# ──────────────────────────────────────────────── CORE PROCESS ────────────────────────────────────────────────
def preprocess_hr(img_pil, max_size, center_crop):
    """
    Convert PIL to normalized tensor, apply auto-resize and optional center crop.
    """
    hr_np = np.array(img_pil).astype(np.float32) / 255.0
    # (H, W, C) -> (1, C, H, W)
    hr = torch.from_numpy(hr_np.transpose(2, 0, 1)).unsqueeze(0).to(device)


    # Auto-resize (for UI; not in original paper, but practical)
    _, _, orig_h, orig_w = hr.shape
    if max(orig_h, orig_w) > max_size:
        s = max_size / max(orig_h, orig_w)
        new_h = int(round(orig_h * s))
        new_w = int(round(orig_w * s))
        hr = F.interpolate(hr, size=(new_h, new_w), mode='bilinear', align_corners=False)


    # Optional center crop to multiple-of-8
    if center_crop:
        h, w = hr.shape[2], hr.shape[3]
        ch = min(max_size, (h // 8) * 8)
        cw = min(max_size, (w // 8) * 8)
        hr = hr[:, :, (h - ch) // 2:(h + ch) // 2, (w - cw) // 2:(w + cw) // 2]


    return hr



def process_image(
    img_pil,
    scale=1.5,
    use_quant=True,
    jpeg_quality=100,
    max_size=512,
    center_crop=False,
    metric_y_channel=False,
    metric_crop_border=True,
):
    """
    HR -> (bicubic LR, embedded LR, AIDN SR, Bicubic SR)
    If jpeg_quality < 100, JPEG is applied to embedded LR before restoration.
    """
    # Preprocess HR
    hr = preprocess_hr(img_pil, max_size=max_size, center_crop=center_crop)


    # Bicubic LR (guidance LR, as in paper's f(I_HR))
    lr = imresize(hr, scale=1.0 / scale)


    # Quantization switch (affects internal quantization layer)
    cfg.quantization = use_quant


    # Forward through AIDN
    t0 = time.time()
    with torch.no_grad():
        encoded_lr, sr = model(hr, float(scale))
    t = time.time() - t0


    # Convert to uint8 for visualization
    hr_vis = tensor_to_uint8(hr)
    # bicubic LR visualized (rescaled to [0,1] internally by util)
    lr_vis = tensor_to_uint8(lr)
    # embedded LR (what is transmitted); enhanced only visually
    enc_vis = tensor_to_uint8(torch.clamp(encoded_lr, 0, 1), enhance=True)
    sr_vis = tensor_to_uint8(sr)
    bic_vis = tensor_to_uint8(imresize(lr, scale))


    # JPEG simulation on embedded LR (AIDN+ style)
    if jpeg_quality < 100:
        enc_jpeg = apply_jpeg(enc_vis, jpeg_quality)
        # Normalize and feed into restoration network
        enc_t = torch.from_numpy(
            enc_jpeg.astype(np.float32).transpose(2, 0, 1) / 255.0
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            # Only restoration uses LR as input in the paper pipeline
            _, sr_new = model(enc_t, float(scale))
        sr_vis = tensor_to_uint8(sr_new)


    # Align shapes for metrics/visuals
    min_h = min(
        hr_vis.shape[0],
        sr_vis.shape[0],
        bic_vis.shape[0],
        enc_vis.shape[0],
        lr_vis.shape[0],
    )
    min_w = min(
        hr_vis.shape[1],
        sr_vis.shape[1],
        bic_vis.shape[1],
        enc_vis.shape[1],
        lr_vis.shape[1],
    )


    imgs = {
        "GT_HR": hr_vis[:min_h, :min_w],
        "Bicubic_LR": lr_vis[:min_h, :min_w],
        "Embedded_LR_AIDN": enc_vis[:min_h, :min_w],
        "AIDN_SR": sr_vis[:min_h, :min_w],
        "Bicubic_SR": bic_vis[:min_h, :min_w],
    }


    # Metrics (by default RGB; optionally Y-channel + border crop)
    psnr_aidn = calc_psnr(
        imgs["AIDN_SR"],
        imgs["GT_HR"],
        use_y=metric_y_channel,
        scale=scale,
        crop=metric_crop_border,
    )
    ssim_aidn = calc_ssim(
        imgs["AIDN_SR"],
        imgs["GT_HR"],
        use_y=metric_y_channel,
        scale=scale,
        crop=metric_crop_border,
    )
    psnr_bic = calc_psnr(
        imgs["Bicubic_SR"],
        imgs["GT_HR"],
        use_y=metric_y_channel,
        scale=scale,
        crop=metric_crop_border,
    )
    ssim_bic = calc_ssim(
        imgs["Bicubic_SR"],
        imgs["GT_HR"],
        use_y=metric_y_channel,
        scale=scale,
        crop=metric_crop_border,
    )


    metrics = {
        "PSNR(AIDN)": psnr_aidn,
        "SSIM(AIDN)": ssim_aidn,
        "PSNR(Bicubic)": psnr_bic,
        "SSIM(Bicubic)": ssim_bic,
    }


    return imgs, metrics, t



# ──────────────────────────────────────────────── UI ────────────────────────────────────────────────
st.set_page_config(page_title="🔥 AIDN Demo", layout="wide", page_icon="🔥")


col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("# 🔥 **AIDN Live**")
with col2:
    st.markdown(
        """
    **IEEE TIP 2023** • Scale-Arbitrary Invertible Image Downscaling  
    - Embed HR → imperceptible LR  
    - Restore HR solely from embedded LR  
    - Robust to JPEG (AIDN+)
    """
    )


# ──────────────────────────────────────────────── GLOBAL SETTINGS (SIDEBAR) ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Global Settings")
    max_size = st.slider("Max image size (px)", 128, 1024, 512, 64,
                         help="Images larger than this will be auto-resized before processing.")
    metric_y = st.checkbox("Metrics on Y-channel (paper style)", False,
                            help="If checked, PSNR/SSIM are computed on the luminance channel only.")
    metric_crop = st.checkbox("Crop border for metrics", True,
                               help="Crop ceil(scale) pixels on each border before computing metrics.")


tab_main, tab_restore, tab_benchmark, tab_batch, tab_webcam = st.tabs(
    ["🎯 Main Demo", "🔄 Restore HR", "🏆 Benchmark", "📦 Batch", "📸 Webcam"]
)


# ──────────────────────────────────────────────── MAIN DEMO ────────────────────────────────────────────────
with tab_main:
    col_ctrl, col_main = st.columns([1, 3.5])
    with col_ctrl:
        scale = st.select_slider("Scale s", [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0], 1.5)
        quant = st.checkbox("Enable quantization layer", True)
        jpeg_q = st.slider("JPEG Quality on Embedded LR", 30, 100, 100, 5)
        st.session_state['crop_center'] = st.checkbox("Center crop to multiple-of-8", False)


    with col_main:
        uploaded = st.file_uploader("📁 Upload HR image", type=["png", "jpg", "jpeg"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="**Input HR (original)**")


            col_btn1, _ = st.columns(2)
            with col_btn1:
                if st.button("🚀 **Run AIDN**", type="primary", use_container_width=True):
                    with st.spinner("Processing..."):
                        results, metrics, t = process_image(
                            img,
                            scale=scale,
                            use_quant=quant,
                            jpeg_quality=jpeg_q,
                            max_size=max_size,
                            center_crop=st.session_state['crop_center'],
                            metric_y_channel=metric_y,
                            metric_crop_border=metric_crop,
                        )
                        st.session_state.results = results
                        st.session_state.metrics = metrics
                        st.session_state.t = t
                    st.success(f"✅ Done in **{t:.2f}s**")


            if 'results' in st.session_state:
                cols = st.columns(5)
                for col, (title, im) in zip(
                    cols,
                    {
                        "GT_HR": "Ground Truth HR",
                        "Bicubic_LR": "Bicubic LR (guidance)",
                        "Embedded_LR_AIDN": "Embedded LR (AIDN)",
                        "AIDN_SR": "AIDN SR",
                        "Bicubic_SR": "Bicubic SR",
                    }.items(),
                ):
                    with col:
                        st.image(
                            st.session_state.results[title],
                            caption=im,
                        )


                c1, c2, c3, c4 = st.columns(4)
                c1.metric("PSNR AIDN", f"{st.session_state.metrics['PSNR(AIDN)']:.2f}")
                c2.metric("SSIM AIDN", f"{st.session_state.metrics['SSIM(AIDN)']:.4f}")
                c3.metric("PSNR Bicubic", f"{st.session_state.metrics['PSNR(Bicubic)']:.2f}")
                c4.metric("SSIM Bicubic", f"{st.session_state.metrics['SSIM(Bicubic)']:.4f}")


                # ZIP download
                zip_buf = BytesIO()
                with zipfile.ZipFile(zip_buf, "w") as zf:
                    for name, arr in st.session_state.results.items():
                        pil_img = Image.fromarray(arr)
                        buf = BytesIO()
                        pil_img.save(buf, "PNG")
                        zf.writestr(f"{name}_{uploaded.name}_x{scale}.png", buf.getvalue())
                zip_buf.seek(0)
                st.download_button(
                    "💾 **Download all results (ZIP)**",
                    zip_buf,
                    f"AIDN_{uploaded.name}_x{scale}_q{jpeg_q}.zip",
                    "application/zip",
                )


# ──────────────────────────────────────────────── RESTORE FROM LR ────────────────────────────────────────────────
with tab_restore:
    st.subheader("🔄 Restore HR solely from embedded LR")
    st.markdown("Use an **embedded LR** image (output of AIDN embedding network or this demo).")
    lr_img_file = st.file_uploader("📁 Upload embedded LR", type=["png", "jpg", "jpeg"])
    if lr_img_file:
        lr_img = Image.open(lr_img_file).convert("RGB")
        st.image(lr_img, caption="**Input embedded LR**")
        s = st.select_slider("Assumed scale s", [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0], 1.5)
        if st.button("🔮 **Restore HR from LR**", type="primary"):
            lr_np = np.array(lr_img).astype(np.float32) / 255.0
            lr_t = torch.from_numpy(lr_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
            with torch.no_grad():
                _, restored = model(lr_t, float(s))
            st.image(tensor_to_uint8(restored), caption=f"**Restored HR (×{s})**")


# ──────────────────────────────────────────────── BENCHMARK ────────────────────────────────────────────────
with tab_benchmark:
    st.header("🏆 Quick Benchmark (single image)")
    set_img = st.file_uploader("Upload a test image (HR)", type=["png", "jpg"], key="bench_img")
    if set_img:
        img = Image.open(set_img).convert("RGB")
        scales = [1.5, 2.0, 4.0]
        if st.button("🏆 **Run at preset scales**"):
            benchmark_results = []
            for s in scales:
                _, m, t = process_image(
                    img,
                    scale=s,
                    max_size=max_size,
                    center_crop=True,
                    metric_y_channel=metric_y,
                    metric_crop_border=metric_crop,
                )
                benchmark_results.append(
                    {
                        "Scale": s,
                        "PSNR(AIDN)": m["PSNR(AIDN)"],
                        "SSIM(AIDN)": m["SSIM(AIDN)"],
                        "PSNR(Bicubic)": m["PSNR(Bicubic)"],
                        "SSIM(Bicubic)": m["SSIM(Bicubic)"],
                        "Time (s)": t,
                    }
                )
            st.dataframe(pd.DataFrame(benchmark_results))


# ──────────────────────────────────────────────── BATCH ────────────────────────────────────────────────
with tab_batch:
    st.header("📦 Batch Processing")
    batch_files = st.file_uploader(
        "Select multiple HR images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )
    if batch_files and st.button("⚡ **Process Batch (default s=1.5)**"):
        batch_results = []
        progress_bar = st.progress(0)
        for i, f in enumerate(batch_files):
            img = Image.open(f).convert("RGB")
            _, m, t = process_image(
                img,
                scale=1.5,
                max_size=max_size,
                center_crop=True,
                metric_y_channel=metric_y,
                metric_crop_border=metric_crop,
            )
            batch_results.append(
                {
                    "File": f.name,
                    "Scale": 1.5,
                    "PSNR(AIDN)": f"{m['PSNR(AIDN)']:.2f}",
                    "SSIM(AIDN)": f"{m['SSIM(AIDN)']:.4f}",
                    "PSNR(Bicubic)": f"{m['PSNR(Bicubic)']:.2f}",
                    "SSIM(Bicubic)": f"{m['SSIM(Bicubic)']:.4f}",
                    "Time (s)": f"{t:.2f}",
                }
            )
            progress_bar.progress((i + 1) / len(batch_files))
        st.dataframe(pd.DataFrame(batch_results))


# ──────────────────────────────────────────────── WEBCAM ────────────────────────────────────────────────
with tab_webcam:
    st.header("📸 Live Webcam")
    camera_img = st.camera_input("📸 Take a photo")
    if camera_img:
        img = Image.open(camera_img).convert("RGB")
        st.image(img, caption="**Live capture**")
        col1, _ = st.columns(2)
        with col1:
            if st.button("🔥 **AIDN Live SR (s=1.5)**", type="primary"):
                results, metrics, t = process_image(
                    img,
                    scale=1.5,
                    max_size=max_size,
                    center_crop=True,
                    metric_y_channel=metric_y,
                    metric_crop_border=metric_crop,
                )
                st.session_state.webcam_results = results
                st.session_state.webcam_t = t


        if 'webcam_results' in st.session_state:
            st.image(
                st.session_state.webcam_results["AIDN_SR"],
                caption=f"**Live Super-Resolution (×1.5, {st.session_state.webcam_t:.2f}s)**",
            )