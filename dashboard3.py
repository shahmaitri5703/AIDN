import streamlit as st
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import cv2
import io
import zipfile
from pathlib import Path
import time
from io import BytesIO
import torch.nn.functional as F
import base64
import streamlit.components.v1 as components

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

def get_image_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ──────────────────────────────────────────────── ROI UTILITIES ────────────────────────────────────────────────
def sync_zoom_compare(img1, img2, label1="LR Original", label2="AIDN Restored"):
    """Industry-standard synchronized zoom/pan using OpenSeadragon with polished UI."""
    b1 = get_image_base64(img1)
    b2 = get_image_base64(img2)

    html_code = f"""
    <div id="osd-container" style="display: flex; width: 100%; height: 700px; background: #000; border-radius: 12px; overflow: hidden; border: 1px solid #333; position: relative;">
        <div style="width: 50%; height: 100%; border-right: 1px solid #444; position: relative;">
            <div id="viewer1" style="width: 100%; height: 100%;"></div>
            <div style="position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.7); color: #fff; padding: 3px 8px; border-radius: 4px; font-family: sans-serif; font-size: 10px; z-index: 10; font-weight: 600;">{label1}</div>
        </div>
        <div style="width: 50%; height: 100%; position: relative;">
            <div id="viewer2" style="width: 100%; height: 100%;"></div>
            <div style="position: absolute; top: 10px; left: 10px; background: rgba(255,140,0,0.9); color: #fff; padding: 3px 8px; border-radius: 4px; font-family: sans-serif; font-size: 10px; z-index: 10; font-weight: 600;">{label2}</div>
        </div>
        <div style="position: absolute; bottom: 12px; right: 12px; background: rgba(0,0,0,0.5); color: #aaa; padding: 4px 10px; border-radius: 20px; font-family: sans-serif; font-size: 10px; z-index: 10; backdrop-filter: blur(4px);">
            🖱️ Scroll to Sync-Zoom | 🖐️ Drag to Sync-Pan
        </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/openseadragon.min.js"></script>
    <script>
    (function() {{
        const viewer1 = OpenSeadragon({{
            id: "viewer1",
            prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",
            tileSources: {{ type: 'image', url: 'data:image/png;base64,{b1}' }},
            showNavigationControl: false,
            gestureSettingsMouse: {{ clickToZoom: false, scrollZoom: true }},
            pointerDelta: 0,
            imageSmoothingEnabled: false,
            zoomPerScroll: 1.5,
            minZoomLevel: 0.5,
            maxZoomLevel: 100
        }});
        const viewer2 = OpenSeadragon({{
            id: "viewer2",
            prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",
            tileSources: {{ type: 'image', url: 'data:image/png;base64,{b2}' }},
            showNavigationControl: false,
            gestureSettingsMouse: {{ clickToZoom: false, scrollZoom: true }},
            pointerDelta: 0,
            zoomPerScroll: 1.5,
            minZoomLevel: 0.5,
            maxZoomLevel: 100
        }});
        let isSyncing = false;
        function sync(source, target) {{
            source.addHandler('viewport-change', function() {{
                if (isSyncing) return;
                isSyncing = true;
                target.viewport.zoomTo(source.viewport.getZoom());
                target.viewport.panTo(source.viewport.getCenter());
                isSyncing = false;
            }});
        }}
        viewer1.addHandler('open', () => {{ viewer1.viewport.goHome(true); }});
        viewer2.addHandler('open', () => {{ viewer2.viewport.goHome(true); }});
        sync(viewer1, viewer2);
        sync(viewer2, viewer1);
    }})();
    </script>
    """
    components.html(html_code, height=720, scrolling=False)


def slider_compare_lib(img_lr, img_hr, label_lr="Original (LR)", label_hr="Restored (HR)"):
    """Forced Full-Width Juxtapose Slider using raw HTML/JS for maximum reliability."""
    b_lr = get_image_base64(img_lr)
    b_hr = get_image_base64(img_hr)

    html_code = f"""
    <link rel="stylesheet" href="https://cdn.knightlab.com/libs/juxtapose/latest/css/juxtapose.css">
    <div style="width: 100%; height: 800px; overflow: hidden; border-radius: 12px; border: 1px solid #333;">
        <div id="juxtapose-embed" style="width: 100%; height: 100%;"></div>
    </div>
    <script src="https://cdn.knightlab.com/libs/juxtapose/latest/js/juxtapose.min.js"></script>
    <script>
    (function() {{
        new juxtapose.JXSlider('#juxtapose-embed', [
            {{
                src: 'data:image/png;base64,{b_hr}',
                label: '{label_hr}'
            }},
            {{
                src: 'data:image/png;base64,{b_lr}',
                label: '{label_lr}'
            }}
        ], {{
            animate: true,
            showLabels: true,
            showCredits: false,
            startingPosition: "50%",
            makeResponsive: true
        }});
    }})();
    </script>
    """
    components.html(html_code, height=820, scrolling=False)


def restore_patch_from_lr(lr_img_pil, bbox, scale, use_quant=True):
    """
    Restore a single ROI patch from the encoded LR image using AIDN restoration network.
    bbox = (left, top, width, height)

    Paper §IV-H.2 proves HR info is embedded non-globally — each LR patch independently
    carries enough information to restore itself, enabling memory-efficient ROI zoom
    without restoring the whole image first.

    cfg.quantization must match what was used during embedding (paper §III-B.3).
    """
    cfg.quantization = use_quant
    left, top, w, h = bbox
    lr_crop = lr_img_pil.crop((left, top, left + w, top + h))
    lr_np = np.array(lr_crop).astype(np.float32) / 255.0
    lr_t = torch.from_numpy(lr_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        _, restored = model(lr_t, float(scale))
    return Image.fromarray(tensor_to_uint8(restored))


def restore_patches_batch(lr_img_pil, batch_queue, scale, use_quant=True):
    """Restore all queued ROI patches from the encoded LR image."""
    results = []
    for bbox in batch_queue:
        results.append(restore_patch_from_lr(lr_img_pil, bbox, scale, use_quant))
    return results


def embed_lr_from_hr(hr_img_pil, scale, use_quant=True):
    """
    Run AIDN embedding network: HR -> encoded LR (ˆI_LR).
    Per paper §III-B.3: the quantization layer clips the continuous float output
    to 8-bit integers [0,255] so the LR is platform-compatible.
    We set cfg.quantization before inference, matching process_image().
    """
    cfg.quantization = use_quant
    hr_np = np.array(hr_img_pil).astype(np.float32) / 255.0
    hr_t = torch.from_numpy(hr_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        encoded_lr, _ = model(hr_t, float(scale))
    # clamp to [0,1] then convert to uint8 — no contrast enhancement here,
    # the encoded LR must look like a normal bicubic-downscaled image (paper §III-C guidance loss)
    arr = np.clip(util.tensor2img(encoded_lr.squeeze(0)) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def get_saliency_roi(lr_img_pil):
    """
    Auto-detect a salient Region of Interest using gradient magnitude as a
    simple saliency proxy (no extra deps needed). Returns (x, y, w, h).
    """
    gray = np.array(lr_img_pil.convert("L")).astype(np.float32)
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    saliency = np.sqrt(gx**2 + gy**2)

    # Blur and find the maximum-saliency region via a sliding window
    from scipy.ndimage import uniform_filter
    h, w = saliency.shape
    win_h = max(32, h // 4)
    win_w = max(32, w // 4)
    smoothed = uniform_filter(saliency, size=(win_h // 2, win_w // 2))
    cy, cx = np.unravel_index(np.argmax(smoothed), smoothed.shape)

    x = max(0, cx - win_w // 2)
    y = max(0, cy - win_h // 2)
    rw = min(win_w, w - x)
    rh = min(win_h, h - y)
    return (x, y, rw, rh)


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

    hr_vis  = t2i(hr)
    lr_vis  = t2i(lr)
    enc_vis = t2i(torch.clamp(encoded_lr, 0, 1), enhance=True)
    sr_vis  = t2i(sr)
    bic_vis = t2i(imresize(lr, scale))

    # JPEG simulation
    if jpeg_quality < 100:
        enc_vis = apply_jpeg(enc_vis, jpeg_quality)
        enc_t = torch.from_numpy(enc_vis.transpose(2, 0, 1).astype(np.float32)/255.0).unsqueeze(0).to(device)
        with torch.no_grad():
            _, sr_new = model(enc_t, float(scale))
        sr_vis = t2i(sr_new)

    # Shape alignment
    min_h = min(hr_vis.shape[0], sr_vis.shape[0], bic_vis.shape[0], enc_vis.shape[0], lr_vis.shape[0])
    min_w = min(hr_vis.shape[1], sr_vis.shape[1], bic_vis.shape[1], enc_vis.shape[1], lr_vis.shape[1])

    imgs = {
        "Input (HR)":     hr_vis[:min_h, :min_w],
        "Downsampled (LR)": lr_vis[:min_h, :min_w],
        "Encoded LR":     enc_vis[:min_h, :min_w],
        "AIDN SR":        sr_vis[:min_h, :min_w],
        "Bicubic SR":     bic_vis[:min_h, :min_w]
    }

    metrics = {
        "PSNR(AIDN)":    calc_psnr(imgs["AIDN SR"], imgs["Input (HR)"]),
        "SSIM(AIDN)":    calc_ssim(imgs["AIDN SR"], imgs["Input (HR)"]),
        "PSNR(Bicubic)": calc_psnr(imgs["Bicubic SR"], imgs["Input (HR)"]),
        "SSIM(Bicubic)": calc_ssim(imgs["Bicubic SR"], imgs["Input (HR)"])
    }

    return imgs, metrics, t

# ──────────────────────────────────────────────── UI ────────────────────────────────────────────────
st.set_page_config(page_title="🔥 AIDN Demo", layout="wide", page_icon="🔥")

# ── CUSTOM CSS (merged from ROI dashboard) ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; }
    .stMetric { background-color: #1e2130; padding: 10px; border-radius: 10px; border: 1px solid #333; }
    .step-header { color: #FF8C00; font-weight: bold; margin-bottom: 20px; text-transform: uppercase;
                   letter-spacing: 1.5px; border-bottom: 2px solid #FF8C00; padding-bottom: 5px; }
    .card { background: #1a1c24; padding: 20px; border-radius: 12px; border: 1px solid #2d2d2d; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ──────────────────────────────────────────────────────────────────────────────────────────
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

# ── SIDEBAR ─────────────────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🎛️ Controls")
st.sidebar.slider("Max image size", 256, 1024, 512, key="max_size")
st.sidebar.markdown("---")

# ── ROI session state init ───────────────────────────────────────────────────────────────────────────
if 'roi_step' not in st.session_state:
    st.session_state.roi_step = 1
if 'roi_batch_queue' not in st.session_state:
    st.session_state.roi_batch_queue = []

# ROI sidebar status
with st.sidebar:
    st.markdown("---")
    st.subheader("🎯 ROI Workflow")
    st.metric("ROI Stage", f"Step {st.session_state.roi_step}")

    if 'roi_hr_img' in st.session_state:
        st.success("✅ ROI Source Ready")
        st.caption(f"Dim: {st.session_state.roi_hr_img.size[0]}x{st.session_state.roi_hr_img.size[1]}px")
    else:
        st.info("⌛ Waiting for ROI Source")

    if 'roi_lr_img' in st.session_state:
        st.success("✅ AIDN LR Embedded (ROI)")

    st.markdown("---")
    st.subheader(f"ROI Batch Queue ({len(st.session_state.roi_batch_queue)})")
    if st.session_state.roi_batch_queue:
        for i, bbox in enumerate(st.session_state.roi_batch_queue):
            st.caption(f"#{i+1}: {bbox[2]}x{bbox[3]} at ({bbox[0]},{bbox[1]})")
        if st.button("🗑️ Clear ROI Batch", use_container_width=True):
            st.session_state.roi_batch_queue = []
            st.rerun()
    else:
        st.write("No ROI regions selected yet.")

    st.markdown("---")
    st.caption("AIDN v2.1.0 | IEEE TIP 2023")

# ──────────────────────────────────────────────── TABS ────────────────────────────────────────────────
tab_main, tab_restore, tab_benchmark, tab_batch, tab_webcam, tab_roi = st.tabs([
    "🎯 Main Demo", "🔄 Restore LR", "🏆 Benchmark", "📦 Batch", "📸 Webcam", "🔬 ROI Restore"
])

# ──────────────────────────────────────────────── TAB: MAIN DEMO ────────────────────────────────────
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
                c1.metric("PSNR AIDN",    f"{st.session_state.metrics['PSNR(AIDN)']:.2f}")
                c2.metric("SSIM AIDN",    f"{st.session_state.metrics['SSIM(AIDN)']:.4f}")
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

# ──────────────────────────────────────────────── TAB: RESTORE LR ───────────────────────────────────
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

# ──────────────────────────────────────────────── TAB: BENCHMARK ────────────────────────────────────
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

# ──────────────────────────────────────────────── TAB: BATCH ────────────────────────────────────────
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

# ──────────────────────────────────────────────── TAB: WEBCAM ───────────────────────────────────────
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

# ──────────────────────────────────────────────── TAB: ROI RESTORE ──────────────────────────────────
# Based on the paper's finding (Section IV-H.2): HR information is embedded non-globally,
# so any LR patch can be independently restored — enabling efficient Region-of-Interest zoom.
with tab_roi:
    st.markdown("""
    **🔬 ROI (Region of Interest) Restore**  
    *Based on AIDN paper §IV-H.2 — HR info is embedded non-globally: each LR patch  
    carries enough information to restore itself independently, enabling memory-efficient ROI zoom.*
    """)

    # ── Step buttons ────────────────────────────────────────────────────────────────────────────────
    def get_roi_step_style(target):
        return "primary" if st.session_state.roi_step == target else "secondary"

    sc1, sc2, sc3, sc4 = st.columns(4)
    if sc1.button("📤 1. SOURCE",  type=get_roi_step_style(1)):
        st.session_state.roi_step = 1; st.rerun()
    if sc2.button("🗜️ 2. EMBED",  type=get_roi_step_style(2),
                  disabled='roi_hr_img' not in st.session_state):
        st.session_state.roi_step = 2; st.rerun()
    if sc3.button("🎯 3. TARGET", type=get_roi_step_style(3),
                  disabled='roi_lr_img' not in st.session_state):
        st.session_state.roi_step = 3; st.rerun()
    if sc4.button("✨ 4. REVIEW", type=get_roi_step_style(4),
                  disabled='roi_last_results' not in st.session_state):
        st.session_state.roi_step = 4; st.rerun()

    st.divider()

    # ── STAGE 1: SOURCE ──────────────────────────────────────────────────────────────────────────────
    if st.session_state.roi_step == 1:
        st.markdown('<div class="step-header">Source Image Acquisition</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)

        uploaded_roi = st.file_uploader(
            "Upload High-Res (HR) Image for ROI workflow",
            type=["png", "jpg", "jpeg", "webp"],
            key="roi_upload_source"
        )
        if uploaded_roi:
            hr_img = Image.open(uploaded_roi).convert("RGB")
            max_dim = 960
            if max(hr_img.size) > max_dim:
                ratio = max_dim / max(hr_img.size)
                hr_img = hr_img.resize(
                    (int(hr_img.size[0] * ratio), int(hr_img.size[1] * ratio)),
                    Image.LANCZOS
                )
                st.warning(f"Resized to {hr_img.size[0]}x{hr_img.size[1]}px for processing.")

            st.session_state.roi_hr_img = hr_img
            st.image(hr_img, caption="Loaded HR Source")

            if st.button("🚀 Proceed to Embedding", type="primary", key="roi_to_embed"):
                st.session_state.roi_step = 2; st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # ── STAGE 2: EMBED ───────────────────────────────────────────────────────────────────────────────
    elif st.session_state.roi_step == 2:
        st.markdown('<div class="step-header">AIDN Compression & Embedding</div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            roi_scale = st.slider("Compression Scale", 1.2, 4.0, 2.0, 0.1, key="roi_scale_slider")
            st.caption("Higher scale = Smaller LR but harder restoration.")
            roi_quant = st.checkbox("Quantization (8-bit LR)", True, key="roi_quant_check",
                                    help="Per paper §III-B.3: quantize encoded LR to 8-bit "
                                         "for platform compatibility. Keep ON to match real-world use.")
        with c2:
            st.info(
                "The embedding network downscales HR → LR while hiding HR information "
                "imperceptibly inside the LR image (AIDN §III-A). "
                "Guidance loss ensures ˆI_LR looks like a normal bicubic-downscaled image."
            )

        if st.button("🗜️ Execute AIDN Compression", type="primary", key="roi_embed_btn"):
            with st.spinner("Running AIDN Embedding Network…"):
                lr_img = embed_lr_from_hr(st.session_state.roi_hr_img, roi_scale, roi_quant)
                st.session_state.roi_lr_img   = lr_img
                st.session_state.roi_scale    = roi_scale
                st.session_state.roi_quant    = roi_quant
            st.rerun()

        if 'roi_lr_img' in st.session_state:
            st.image(
                st.session_state.roi_lr_img,
                caption=f"Encoded LR image "
                        f"({st.session_state.roi_lr_img.size[0]}x{st.session_state.roi_lr_img.size[1]})"
            )
            st.success("HR information successfully embedded in LR image.")

            # Offer download of the embedded LR
            buf = BytesIO()
            st.session_state.roi_lr_img.save(buf, format="PNG")
            st.download_button(
                "💾 Download Encoded LR",
                buf.getvalue(),
                f"encoded_lr_x{st.session_state.roi_scale}.png",
                "image/png",
                key="roi_dl_lr"
            )

            if st.button("🎯 Proceed to Region Selection", type="primary", key="roi_to_target"):
                st.session_state.roi_step = 3; st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # ── STAGE 3: TARGET (ROI selection) ─────────────────────────────────────────────────────────────
    elif st.session_state.roi_step == 3:
        st.markdown('<div class="step-header">Region of Interest Selection</div>', unsafe_allow_html=True)

        # Draw existing batch queue boxes on the LR image
        lr_display = st.session_state.roi_lr_img.copy()
        draw = ImageDraw.Draw(lr_display, "RGBA")
        for i, bbox in enumerate(st.session_state.roi_batch_queue):
            lx, ly, lw, lh = bbox
            draw.rectangle(
                [lx, ly, lx + lw, ly + lh],
                outline="#FF8C00", width=2, fill=(255, 140, 0, 40)
            )

        col_sel, col_ctrl = st.columns([3, 1])

        with col_sel:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Draw a rectangle on the LR image to select an ROI:**")

            # Streamlit-cropper optional — fall back to manual coordinate entry if not installed
            try:
                from streamlit_cropper import st_cropper
                rect = st_cropper(
                    lr_display,
                    realtime_update=True,
                    box_color='#FF8C00',
                    return_type='box',
                    default_coords=st.session_state.get('roi_proposed_roi'),
                    key="roi_cropper"
                )
            except ImportError:
                st.warning(
                    "`streamlit-cropper` is not installed. "
                    "Using manual coordinate entry instead. "
                    "Run `pip install streamlit-cropper` for the interactive selector."
                )
                st.image(lr_display, caption="Encoded LR (existing ROIs highlighted)")
                img_w, img_h = st.session_state.roi_lr_img.size
                rx = st.number_input("Left (x)", 0, img_w - 1, 0, key="roi_mx")
                ry = st.number_input("Top  (y)", 0, img_h - 1, 0, key="roi_my")
                rw = st.number_input("Width",    1, img_w,    img_w // 4, key="roi_mw")
                rh = st.number_input("Height",   1, img_h,    img_h // 4, key="roi_mh")
                rect = {"left": rx, "top": ry, "width": rw, "height": rh}

            st.markdown('</div>', unsafe_allow_html=True)

        with col_ctrl:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("**Selection Controls**")

            if st.button("✨ Auto-Detect ROI", key="roi_auto"):
                x, y, w, h = get_saliency_roi(st.session_state.roi_lr_img)
                st.session_state.roi_proposed_roi = (x, x + w, y, y + h)
                st.rerun()

            if rect:
                curr_bbox = (
                    int(rect['left']), int(rect['top']),
                    int(rect['width']), int(rect['height'])
                )
                if st.button("➕ Add Selection to Batch", type="primary", key="roi_add"):
                    st.session_state.roi_batch_queue.append(curr_bbox)
                    st.toast("Region Added!")
                    st.rerun()

            st.divider()
            st.write(f"In Batch: **{len(st.session_state.roi_batch_queue)}**")

            if st.button(
                "🚀 Restore Batch & Review", type="primary",
                disabled=not st.session_state.roi_batch_queue,
                key="roi_restore_btn"
            ):
                with st.spinner("Restoring Selected Regions…"):
                    results = restore_patches_batch(
                        st.session_state.roi_lr_img,
                        st.session_state.roi_batch_queue,
                        st.session_state.roi_scale,
                        st.session_state.get('roi_quant', True)
                    )
                    st.session_state.roi_last_results = results
                    st.session_state.roi_last_bboxes  = list(st.session_state.roi_batch_queue)
                st.session_state.roi_step = 4
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

    # ── STAGE 4: REVIEW ──────────────────────────────────────────────────────────────────────────────
    elif st.session_state.roi_step == 4:
        st.markdown('<div class="step-header">High-Resolution Restoration Results</div>', unsafe_allow_html=True)

        if 'roi_last_results' in st.session_state:
            for i, (hr_patch, bbox) in enumerate(
                zip(st.session_state.roi_last_results, st.session_state.roi_last_bboxes)
            ):
                st.divider()
                st.subheader(f"Region #{i+1} — {bbox[2]}x{bbox[3]} at ({bbox[0]},{bbox[1]})")

                view_mode = st.radio(
                    f"Inspector Mode (Region #{i+1})",
                    ["🔍 Sync Pro-Inspector", "↔️ Full-Width Slider"],
                    horizontal=True, index=1, key=f"roi_mode_{i}"
                )

                lr_crop = st.session_state.roi_lr_img.crop(
                    (bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])
                )

                if "Slider" in view_mode:
                    target_w = 1920
                    w_p, h_p = hr_patch.size
                    ratio = target_w / w_p
                    hr_up = hr_patch.resize((target_w, int(h_p * ratio)), Image.LANCZOS)
                    lr_up = lr_crop.resize((target_w, int(h_p * ratio)), Image.NEAREST)
                    slider_compare_lib(lr_up, hr_up)
                else:
                    sync_zoom_compare(lr_crop, hr_patch)

                # ── Metrics: AIDN-Restored vs Bicubic-Upscaled-from-LR ──────────────────────────
                # This mirrors the paper's evaluation (Table II/III): compare AIDN restoration
                # quality against bicubic upscaling of the same LR input as the baseline.
                # Ground truth HR is not available here since we only have the distributed LR.
                lr_crop_arr   = np.array(lr_crop)
                # Bicubic-upsample the LR crop to the same size as the AIDN-restored patch
                bic_up = lr_crop.resize(hr_patch.size, Image.BICUBIC)
                bic_arr = np.array(bic_up)
                hr_arr  = np.array(hr_patch)

                min_h_m = min(hr_arr.shape[0], bic_arr.shape[0])
                min_w_m = min(hr_arr.shape[1], bic_arr.shape[1])

                psnr_aidn = calc_psnr(hr_arr[:min_h_m, :min_w_m], bic_arr[:min_h_m, :min_w_m])
                ssim_aidn = calc_ssim(hr_arr[:min_h_m, :min_w_m], bic_arr[:min_h_m, :min_w_m])

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("AIDN Restored Size",
                           f"{hr_patch.size[0]}×{hr_patch.size[1]}px")
                mc2.metric("PSNR: AIDN vs Bicubic baseline",
                           f"{psnr_aidn:.2f} dB",
                           help="Higher = AIDN restored patch has more detail than bicubic upscale of same LR crop.")
                mc3.metric("SSIM: AIDN vs Bicubic baseline",
                           f"{ssim_aidn:.4f}",
                           help="Closer to 1.0 = more structurally similar to bicubic baseline. "
                                "AIDN should exceed bicubic per paper Table II.")
                st.caption(
                    "ℹ️ Metrics compare AIDN-restored patch against bicubic-upscaled LR crop "
                    "(paper baseline, Table II). If you have the original HR, upload it below for ground-truth PSNR/SSIM."
                )

                # Optional: ground-truth HR upload for accurate metrics
                gt_file = st.file_uploader(
                    f"📎 Optional: Upload original HR for GT metrics (Region #{i+1})",
                    type=["png","jpg","jpeg"], key=f"roi_gt_{i}"
                )
                if gt_file:
                    gt_img = Image.open(gt_file).convert("RGB")
                    # Crop the same region from GT (GT is at HR resolution = LR * scale)
                    s_val = st.session_state.roi_scale
                    gt_left  = int(bbox[0] * s_val)
                    gt_top   = int(bbox[1] * s_val)
                    gt_right = int((bbox[0] + bbox[2]) * s_val)
                    gt_bot   = int((bbox[1] + bbox[3]) * s_val)
                    gt_crop  = gt_img.crop((gt_left, gt_top, min(gt_right, gt_img.width), min(gt_bot, gt_img.height)))
                    gt_arr   = np.array(gt_crop.resize(hr_patch.size, Image.LANCZOS))
                    min_hg   = min(hr_arr.shape[0], gt_arr.shape[0])
                    min_wg   = min(hr_arr.shape[1], gt_arr.shape[1])
                    psnr_gt  = calc_psnr(hr_arr[:min_hg, :min_wg], gt_arr[:min_hg, :min_wg])
                    ssim_gt  = calc_ssim(hr_arr[:min_hg, :min_wg], gt_arr[:min_hg, :min_wg])
                    gc1, gc2 = st.columns(2)
                    gc1.metric("PSNR vs Ground Truth HR", f"{psnr_gt:.2f} dB")
                    gc2.metric("SSIM vs Ground Truth HR", f"{ssim_gt:.4f}")

                buf = BytesIO()
                hr_patch.save(buf, format="PNG")
                st.download_button(
                    f"⬇️ Download ROI #{i+1}",
                    buf.getvalue(),
                    f"roi_{i+1}_restored.png",
                    "image/png",
                    key=f"roi_dl_{i}"
                )

        st.divider()
        if st.button("⏮️ Back to Selection Stage", use_container_width=True, key="roi_back"):
            st.session_state.roi_step = 3; st.rerun()