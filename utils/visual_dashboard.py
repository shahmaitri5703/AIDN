import plotly.graph_objects as go
import plotly.io as pio
import base64
from pathlib import Path

def img_to_base64(path):
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode()

def make_dashboard(image_entries, save_path, scale):
    """
    image_entries: list of dicts
    Each dict contains:
      name, bicubic, encoded, sr, gt, heatmap, psnr, ssim
    """

    fig = go.Figure()

    buttons = []
    for idx, e in enumerate(image_entries):
        visible = [False] * len(image_entries)
        visible[idx] = True

        fig.add_trace(
            go.Image(
                source=f"data:image/png;base64,{img_to_base64(e['sr'])}",
                visible=(idx == 0),
                name=e["name"]
            )
        )

        buttons.append(
            dict(
                label=e["name"],
                method="update",
                args=[
                    {"visible": visible},
                    {
                        "title": f"{e['name']} | PSNR {e['psnr']:.2f} | SSIM {e['ssim']:.4f}"
                    }
                ]
            )
        )

    fig.update_layout(
        title=f"AIDN Interactive Dashboard (Scale x{scale})",
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                x=0.0,
                y=1.15
            )
        ],
        height=720
    )

    pio.write_html(fig, save_path, auto_open=True)
