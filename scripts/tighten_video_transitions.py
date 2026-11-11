"""Cut three silent spans near the last transitions: python scripts/tighten_video_transitions.py

Trims synchronized audio/video intervals from
deliverables/project_demo_2min_polished.mp4 and writes
deliverables/project_demo_tighter_transitions.mp4 plus a retimed .srt.
Refuses to run if either output already exists.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "deliverables" / "project_demo_2min_polished.mp4"
SOURCE_SRT = SOURCE.with_suffix(".srt")
OUTPUT = ROOT / "deliverables" / "project_demo_tighter_transitions.mp4"
OUTPUT_SRT = OUTPUT.with_suffix(".srt")
BUILD_DIR = Path(r"C:\Users\pdaadh\ua_video_tools\project_demo\build_tighter")

CUTS = [(43.45, 45.95), (90.45, 93.95), (108.55, 109.75)]

SRT_TIME = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})"
)


def build_filter() -> str:
    bounds = [0.0] + [t for cut in CUTS for t in cut] + [None]
    spans = [(bounds[i], bounds[i + 1]) for i in range(0, len(bounds) - 1, 2)]
    parts = []
    labels = []
    for i, (start, end) in enumerate(spans):
        vtrim = f"trim=start={start}"
        atrim = f"atrim=start={start}"
        if end is not None:
            vtrim += f":end={end}"
            atrim += f":end={end}"
        parts.append(f"[0:v]{vtrim},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]{atrim},asetpts=PTS-STARTPTS[a{i}]")
        labels += [f"[v{i}]", f"[a{i}]"]
    parts.append(
        "".join(labels) + f"concat=n={len(spans)}:v=1:a=1[vout][aout]"
    )
    return ";".join(parts)


def srt_to_ms(match: re.Match) -> tuple[int, int]:
    g = [int(x) for x in match.groups()]
    start = ((g[0] * 60 + g[1]) * 60 + g[2]) * 1000 + g[3]
    end = ((g[4] * 60 + g[5]) * 60 + g[6]) * 1000 + g[7]
    return start, end


def retime_ms(t: int) -> int:
    removed = 0
    for start, end in CUTS:
        removed += max(0, min(t, round(end * 1000)) - round(start * 1000))
    return t - removed


def ms_to_srt(t: int) -> str:
    h, rem = divmod(t, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def retime_srt() -> None:
    text = SOURCE_SRT.read_text(encoding="utf-8-sig")

    def sub(match: re.Match) -> str:
        start, end = srt_to_ms(match)
        return f"{ms_to_srt(retime_ms(start))} --> {ms_to_srt(retime_ms(end))}"

    OUTPUT_SRT.write_text(SRT_TIME.sub(sub, text), encoding="utf-8")


def main() -> None:
    for path in (OUTPUT, OUTPUT_SRT):
        if path.exists():
            sys.exit(f"refusing to overwrite {path}")
    for path in (SOURCE, SOURCE_SRT):
        if not path.exists():
            sys.exit(f"missing source {path}")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    tmp_video = BUILD_DIR / OUTPUT.name
    if tmp_video.exists():
        tmp_video.unlink()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-i",
        str(SOURCE),
        "-filter_complex",
        build_filter(),
        "-map",
        "[vout]",
        "-map",
        "[aout]",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "30",
        "-color_range",
        "tv",
        "-color_primaries",
        "bt470bg",
        "-color_trc",
        "unknown",
        "-colorspace",
        "unknown",
        "-c:a",
        "aac",
        "-b:a",
        "256k",
        "-movflags",
        "+faststart",
        str(tmp_video),
    ]
    subprocess.run(cmd, check=True)
    shutil.move(str(tmp_video), OUTPUT)
    retime_srt()
    print(f"wrote {OUTPUT}")
    print(f"wrote {OUTPUT_SRT}")


if __name__ == "__main__":
    main()
