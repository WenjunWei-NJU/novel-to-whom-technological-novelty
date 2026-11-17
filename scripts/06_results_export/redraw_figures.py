"""Redraw the five submission figures from the included CSV inputs."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

# Figure-level headings in the archived drawings. The title artist retains its
# layout box so the original Figure 2 panel crop still uses the same coordinates.
HEADINGS = {
    2: ("Nonlinear career composition", 174),
    3: ("Career stage: distributions and conditional associations", 249),
    4: ("Peer histories and opportunity coverage", 180),
    5: ("Sensitivity of the prespecified composition contrast", 186),
}
DRAW_WITHOUT_HEADING = """
import runpy
import sys
from matplotlib.figure import Figure
original_text = Figure.text
removed = 0

def text_without_heading(self, x, y, text, *args, **kwargs):
    global removed
    artist = original_text(self, x, y, text, *args, **kwargs)
    if text == sys.argv[2]:
        artist.draw = lambda renderer: None
        removed += 1
    return artist

Figure.text = text_without_heading
runpy.run_path(sys.argv[1], run_name="__main__")
assert removed == 1, removed
"""

def draw(figure, spec, output, data_dir):
    with TemporaryDirectory(prefix=f"career-stage-figure-{figure}-") as directory:
        work = Path(directory)
        script = work / spec["script"]
        shutil.copy2(Path(__file__).with_name(script.name), script)
        data = work / "source_data"
        data.mkdir()
        for name in spec["inputs"]:
            shutil.copy2(data_dir / f"figure{figure}" / name, data / name)

        heading, trim = HEADINGS.get(figure, ("", 0))
        command = [sys.executable, str(script)]
        if heading:
            command = [sys.executable, "-c", DRAW_WITHOUT_HEADING, str(script), heading]
        subprocess.run(command, cwd=work, check=True)

        generated = (
            work / "Figure5_sensitivity_5panels" if figure == 5
            else work / "figures" / f"Figure{figure}" / f"Figure{figure}"
        )
        png = output / f"Figure{figure}.png"
        with Image.open(generated.with_suffix(".png")) as image:
            if figure == 2:
                # Remove original panel D. The approved main-text figure uses A–C.
                image.crop((0, trim, 4356, 1949)).save(png, dpi=(600, 600))
            elif trim:
                image.crop((0, trim, image.width, image.height)).save(png, dpi=(600, 600))
            else:
                shutil.copy2(generated.with_suffix(".png"), png)
        if figure != 2:
            shutil.copy2(generated.with_suffix(".pdf"), output / f"Figure{figure}.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "figure_inputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("figures", nargs="+", type=int, choices=range(1, 6))
    args = parser.parse_args()
    specs = json.loads((ROOT / "config" / "figures.json").read_text())
    for figure in args.figures:
        for name in specs[str(figure)]["inputs"]:
            path = args.data_dir / f"figure{figure}" / name
            if not path.is_file():
                parser.error(f"Required Figure {figure} input does not exist: {path}")
    args.output.mkdir(parents=True, exist_ok=True)
    for figure in args.figures:
        draw(figure, specs[str(figure)], args.output.resolve(), args.data_dir.resolve())


if __name__ == "__main__":
    main()
