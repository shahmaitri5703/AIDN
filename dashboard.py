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
    rgb_range = 1.0; arch = 'InvEDRS_arb'; up_sampler = 'sampleB'; down_sampler = 'sampleB'
    n_resblocks = 16; n_feats = 64; fixed_scale = False; scale = 4; rescale = None
    n_colors = 3; res_scale = 1.0; quantization = True; quantization_type = 'round_soft'
    K = 4; num_experts_SAconv = 4; num_experts_CRM = 8; jpeg = False; base_resolution = 4
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

# ──────────────────────────────────────────────── UTILITIES ────────────────────────────────────────────────
def tensor_to_uint8(t, enhance=False):
    img = util.tensor2img(t.squeeze(0))
    if enhance: 
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 2.8
        img = np.clip(img, 0, 1)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)

def apply_jpeg(img_np, quality): 
    pil = Image.fromarray(img_np)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf))

def calc_psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32))**2)
    return 100.0 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

def calc_ssim(a, b):
    a = a.astype(np.float32)/255.0
    b = b.astype(np.float32)/255.0
    mu1, mu2 = np.mean(a), np.mean(b)
    sigma1_sq, sigma2_sq = np.var(a), np.var(b)
    sigma12 = np.mean((a - mu1) * (b - mu2))
    c1, c2 = 0.01**2, 0.03**2
    return (2*mu1*mu2 + c1) * (2*sigma12 + c2) / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))

# ──────────────────────────────────────────────── PROCESS ────────────────────────────────────────────────
def process_image(img_pil, scale=1.5, use_quant=True, jpeg_quality=100):
    hr_np = np.array(img_pil).astype(np.float32) / 255.0
    hr = torch.from_numpy(hr_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
    
    # Auto-resize
    _, _, orig_h, orig_w = hr.shape
    max_size = 512
    if max(orig_h, orig_w) > max_size:
        s = max_size / max(orig_h, orig_w)
        new_h = int(round(orig_h * s))
        new_w = int(round(orig_w * s))
        hr = F.interpolate(hr, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    # Center crop
    if st.session_state.get('crop_center', False):
        h, w = hr.shape[2], hr.shape[3]
        ch = min(512, h // 8 * 8)
        cw = min(512, w // 8 * 8)
        hr = hr[:, :, (h-ch)//2:(h+ch)//2, (w-cw)//2:(w+cw)//2]
    
    lr = imresize(hr, scale=1.0 / scale)
    cfg.quantization = use_quant
    
    t0 = time.time()
    with torch.no_grad():
        encoded_lr, sr = model(hr, float(scale))
    t = time.time() - t0
    
    def t2i(t, enhance=False):
        return tensor_to_uint8(t, enhance)
    
    hr_vis = t2i(hr)
    lr_vis = t2i(lr)
    enc_vis = t2i(torch.clamp(encoded_lr, 0, 1), enhance=True)
    sr_vis = t2i(sr)
    bic_vis = t2i(imresize(lr, scale))
    
    # JPEG simulation
    if jpeg_quality < 100:
        enc_vis = apply_jpeg(enc_vis, jpeg_quality)
        enc_t = torch.from_numpy(enc_vis.transpose(2,0,1).astype(np.float32)/255.0).unsqueeze(0).to(device)
        with torch.no_grad():
            _, sr_new = model(enc_t, float(scale))
        sr_vis = t2i(sr_new)
    
    # Shape alignment
    min_h = min(hr_vis.shape[0], sr_vis.shape[0], bic_vis.shape[0], enc_vis.shape[0], lr_vis.shape[0])
    min_w = min(hr_vis.shape[1], sr_vis.shape[1], bic_vis.shape[1], enc_vis.shape[1], lr_vis.shape[1])
    
    imgs = {
        "Input (HR)": hr_vis[:min_h, :min_w],
        "Downsampled (LR)": lr_vis[:min_h, :min_w],
        "Encoded LR": enc_vis[:min_h, :min_w],
        "AIDN SR": sr_vis[:min_h, :min_w],
        "Bicubic SR": bic_vis[:min_h, :min_w]
    }
    
    metrics = {
        "PSNR(AIDN)": calc_psnr(imgs["AIDN SR"], imgs["Input (HR)"]),
        "SSIM(AIDN)": calc_ssim(imgs["AIDN SR"], imgs["Input (HR)"]),
        "PSNR(Bicubic)": calc_psnr(imgs["Bicubic SR"], imgs["Input (HR)"]),
        "SSIM(Bicubic)": calc_ssim(imgs["Bicubic SR"], imgs["Input (HR)"])
    }
    
    return imgs, metrics, t

# ──────────────────────────────────────────────── UI ────────────────────────────────────────────────
st.set_page_config(page_title="🔥 AIDN Demo", layout="wide", page_icon="🔥")

# ──────────────────────────────────────────────── CLEAN HEADER (NO STATUS) ────────────────────────────────────────────────
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("# 🔥 **AIDN Live**")
with col2:
    st.markdown("""
    **IEEE TIP 2023** • Scale-Arbitrary Invertible Downscaling
    - Embed HR → imperceptible LR 
    - Perfect LR → HR restoration
    - JPEG/quantization robust
    """)

# ──────────────────────────────────────────────── SIMPLIFIED SIDEBAR ────────────────────────────────────────────────
st.sidebar.title("🎛️ Controls")
st.sidebar.slider("Max image size", 256, 1024, 512, key="max_size")
st.sidebar.markdown("---")

# ──────────────────────────────────────────────── TABS ────────────────────────────────────────────────
tab_main, tab_restore, tab_benchmark, tab_batch, tab_webcam = st.tabs([
    "🎯 Main Demo", "🔄 Restore LR", "🏆 Benchmark", "📦 Batch", "📸 Webcam"
])

with tab_main:
    col_ctrl, col_main = st.columns([1, 3.5])
    with col_ctrl:
        scale = st.select_slider("Scale", [1.0,1.2,1.5,2.0,2.5,3.0,4.0], 1.5)
        quant = st.checkbox("Quantization", True)
        jpeg_q = st.slider("JPEG Quality", 30, 100, 92, 5)
        st.session_state['crop_center'] = st.checkbox("Center Crop", False)
    
    with col_main:
        uploaded = st.file_uploader("📁 Upload HR", type=["png","jpg","jpeg"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="**Input HR**")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🚀 **Run AIDN**", type="primary", use_container_width=True):
                    with st.spinner("Processing..."):
                        results, metrics, t = process_image(img, scale, quant, jpeg_q)
                        st.session_state.results = results
                        st.session_state.metrics = metrics
                        st.session_state.t = t
                    st.success(f"✅ **{t:.2f}s**")
            
            if 'results' in st.session_state:
                cols = st.columns(5)
                for col, (title, im) in zip(cols, st.session_state.results.items()):
                    with col:
                        st.image(im, caption=title)
                
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("PSNR AIDN", f"{st.session_state.metrics['PSNR(AIDN)']:.2f}")
                c2.metric("SSIM AIDN", f"{st.session_state.metrics['SSIM(AIDN)']:.4f}")
                c3.metric("PSNR Bicubic", f"{st.session_state.metrics['PSNR(Bicubic)']:.2f}")
                c4.metric("SSIM Bicubic", f"{st.session_state.metrics['SSIM(Bicubic)']:.4f}")
                
                # ZIP DOWNLOAD
                zip_buf = BytesIO()
                with zipfile.ZipFile(zip_buf, "w") as zf:
                    for name, arr in st.session_state.results.items():
                        pil_img = Image.fromarray(arr)
                        buf = BytesIO()
                        pil_img.save(buf, "PNG")
                        zf.writestr(f"{name.replace(' ','_')}_x{scale}.png", buf.getvalue())
                zip_buf.seek(0)
                st.download_button("💾 **Download ZIP**", zip_buf, f"AIDN_x{scale}_q{jpeg_q}.zip", "application/zip")

with tab_restore:
    st.subheader("🔄 Restore from Encoded LR")
    lr_img_file = st.file_uploader("📁 Upload Encoded LR", type=["png","jpg","jpeg"])
    if lr_img_file:
        lr_img = Image.open(lr_img_file).convert("RGB")
        st.image(lr_img, caption="**Input LR**")
        s = st.select_slider("Upscale", [1.0,1.2,1.5,2.0,2.5,3.0,4.0], 1.5)
        if st.button("🔮 **Restore HR**", type="primary"):
            lr_np = np.array(lr_img).astype(np.float32)/255.0
            lr_t = torch.from_numpy(lr_np.transpose(2,0,1)).unsqueeze(0).to(device)
            with torch.no_grad():
                _, restored = model(lr_t, float(s))
            st.image(tensor_to_uint8(restored), caption=f"**Restored HR (×{s})**")

with tab_benchmark:
    st.header("🏆 Benchmark")
    set5_img = st.file_uploader("Upload test image", type=["png","jpg"])
    if set5_img:
        img = Image.open(set5_img).convert("RGB")
        scales = [1.5, 2.0, 4.0]
        if st.button("🏆 **Benchmark Scales**"):
            benchmark_results = []
            for s in scales:
                _, m, t = process_image(img, s)
                benchmark_results.append({"Scale": s, **m, "Time": t})
            st.dataframe(pd.DataFrame(benchmark_results))

with tab_batch:
    st.header("📦 Batch Processing")
    batch_files = st.file_uploader("Select images", type=["png","jpg","jpeg"], accept_multiple_files=True)
    if batch_files and st.button("⚡ **Process Batch**"):
        batch_results = []
        progress_bar = st.progress(0)
        for i, f in enumerate(batch_files):
            img = Image.open(f).convert("RGB")
            r, m, t = process_image(img)
            batch_results.append({"File": f.name, **m, "Time": f"{t:.2f}s"})
            progress_bar.progress((i+1) / len(batch_files))
        st.dataframe(pd.DataFrame(batch_results))

with tab_webcam:
    st.header("📸 Live Webcam")
    camera_img = st.camera_input("📸 Take photo")
    if camera_img:
        img = Image.open(camera_img).convert("RGB")
        st.image(img, caption="**Live Capture**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔥 **AIDN Live SR**", type="primary"):
                results, metrics, t = process_image(img)
                st.session_state.webcam_results = results
                st.session_state.webcam_t = t
        
        if 'webcam_results' in st.session_state:
            st.image(st.session_state.webcam_results["AIDN SR"], 
                    caption=f"**Live Super-Resolution** ({st.session_state.webcam_t:.2f}s)")
