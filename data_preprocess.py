#!/usr/bin/env python3
"""Create contrast-enhanced PNG previews from TIFF imagery.

The source TIFF files are never modified. The generated PNG files are intended
for UI preview and example loading, while tools can still receive the original
TIFF paths through app-level aliases.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image


TIFF_SUFFIXES = {".tif", ".tiff"}


def _finite_values(array: np.ndarray, ignore_zero: bool = True) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    mask = np.isfinite(values)
    if ignore_zero:
        mask &= values != 0
    return values[mask]


def percentile_stretch_gray(
    array: np.ndarray,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    ignore_zero: bool = True,
) -> np.ndarray:
    """Stretch one grayscale band to uint8 using percentiles."""
    values = _finite_values(array, ignore_zero=ignore_zero)
    if values.size == 0:
        return np.zeros_like(array, dtype=np.uint8)

    p_low, p_high = np.percentile(values, [low_pct, high_pct])
    if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low:
        return np.zeros_like(array, dtype=np.uint8)

    stretched = (np.asarray(array, dtype=np.float32) - p_low) / (p_high - p_low) * 255.0
    stretched = np.nan_to_num(stretched, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(stretched, 0, 255).astype(np.uint8)


def percentile_stretch_rgb(
    array: np.ndarray,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    ignore_zero: bool = True,
) -> np.ndarray:
    """Stretch an RGB array channel-by-channel to uint8."""
    data = np.asarray(array)
    if data.ndim != 3 or data.shape[2] < 3:
        raise ValueError(f"Expected HxWx3 RGB array, got shape {data.shape}")

    output = np.zeros(data.shape[:2] + (3,), dtype=np.uint8)
    for channel in range(3):
        output[..., channel] = percentile_stretch_gray(
            data[..., channel],
            low_pct=low_pct,
            high_pct=high_pct,
            ignore_zero=ignore_zero,
        )
    return output


def _read_tif_with_rasterio(path: Path) -> Optional[np.ndarray]:
    try:
        import rasterio
    except Exception:
        return None

    try:
        with rasterio.open(path) as src:
            if src.count >= 3:
                return np.transpose(src.read([1, 2, 3]), (1, 2, 0))
            return src.read(1)
    except Exception:
        return None


def _read_tif_with_pillow(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode in {"RGB", "RGBA"}:
            return np.asarray(image.convert("RGB"))
        return np.asarray(image)


def read_tif_array(path: Path) -> np.ndarray:
    """Read a TIFF image as either HxW or HxWx3 numpy array."""
    array = _read_tif_with_rasterio(path)
    if array is not None:
        return array
    return _read_tif_with_pillow(path)


def build_preview_array(
    array: np.ndarray,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    ignore_zero: bool = True,
) -> np.ndarray:
    """Build a contrast-enhanced RGB preview array from grayscale or RGB data."""
    data = np.asarray(array)
    if data.ndim == 3 and data.shape[2] >= 3:
        return percentile_stretch_rgb(data[..., :3], low_pct, high_pct, ignore_zero)
    if data.ndim == 3 and data.shape[2] == 1:
        data = data[..., 0]
    if data.ndim != 2:
        raise ValueError(f"Unsupported image array shape: {data.shape}")

    gray = percentile_stretch_gray(data, low_pct, high_pct, ignore_zero)
    return np.stack([gray, gray, gray], axis=-1)


def default_preview_path(tif_path: Path, suffix: str = "_preview") -> Path:
    return tif_path.with_name(f"{tif_path.stem}{suffix}.png")


def convert_tif_to_png(
    tif_path: str | Path,
    output_path: str | Path | None = None,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    ignore_zero: bool = True,
    overwrite: bool = True,
) -> Optional[str]:
    """Convert one TIFF file to a contrast-enhanced PNG preview."""
    source = Path(tif_path).expanduser()
    if not source.is_file() or source.suffix.lower() not in TIFF_SUFFIXES:
        return None

    target = Path(output_path).expanduser() if output_path else default_preview_path(source)
    if target.exists() and not overwrite:
        return str(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    array = read_tif_array(source)
    preview = build_preview_array(array, low_pct=low_pct, high_pct=high_pct, ignore_zero=ignore_zero)
    Image.fromarray(preview).save(target)
    return str(target)


def iter_tif_files(input_dir: str | Path, recursive: bool = False) -> Iterable[Path]:
    directory = Path(input_dir).expanduser()
    pattern = "**/*" if recursive else "*"
    for path in sorted(directory.glob(pattern)):
        if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES:
            yield path


def convert_directory(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    ignore_zero: bool = True,
    recursive: bool = False,
    overwrite: bool = True,
) -> List[str]:
    """Convert all TIFF files in one directory."""
    source_dir = Path(input_dir).expanduser()
    target_dir = Path(output_dir).expanduser() if output_dir else None
    outputs: List[str] = []

    for tif_path in iter_tif_files(source_dir, recursive=recursive):
        if target_dir:
            relative = tif_path.relative_to(source_dir)
            output_path = target_dir / relative.with_name(f"{relative.stem}_preview.png")
        else:
            output_path = None
        result = convert_tif_to_png(
            tif_path,
            output_path=output_path,
            low_pct=low_pct,
            high_pct=high_pct,
            ignore_zero=ignore_zero,
            overwrite=overwrite,
        )
        if result:
            outputs.append(result)
    return outputs


def convert_paired_directories(
    sar_dir: str | Path,
    rgb_dir: str | Path,
    output_dir: str | Path | None = None,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    ignore_zero: bool = True,
    recursive: bool = False,
    overwrite: bool = True,
) -> List[str]:
    """Convert every TIFF in paired SAR and RGB directories."""
    outputs: List[str] = []
    if output_dir:
        output_root = Path(output_dir).expanduser()
        outputs.extend(
            convert_directory(
                sar_dir,
                output_dir=output_root / "sar",
                low_pct=low_pct,
                high_pct=high_pct,
                ignore_zero=ignore_zero,
                recursive=recursive,
                overwrite=overwrite,
            )
        )
        outputs.extend(
            convert_directory(
                rgb_dir,
                output_dir=output_root / "rgb",
                low_pct=low_pct,
                high_pct=high_pct,
                ignore_zero=ignore_zero,
                recursive=recursive,
                overwrite=overwrite,
            )
        )
        return outputs

    outputs.extend(
        convert_directory(
            sar_dir,
            low_pct=low_pct,
            high_pct=high_pct,
            ignore_zero=ignore_zero,
            recursive=recursive,
            overwrite=overwrite,
        )
    )
    outputs.extend(
        convert_directory(
            rgb_dir,
            low_pct=low_pct,
            high_pct=high_pct,
            ignore_zero=ignore_zero,
            recursive=recursive,
            overwrite=overwrite,
        )
    )
    return outputs


def _print_outputs(outputs: Sequence[str]) -> None:
    for output in outputs:
        print(f"[OK] {output}")
    print(f"Converted {len(outputs)} file(s).")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert TIFF imagery to contrast-enhanced PNG previews.")
    parser.add_argument("--input", help="Single TIFF file to convert.")
    parser.add_argument("--output", help="Output PNG path for single-file mode.")
    parser.add_argument("--input-dir", help="Directory containing TIFF files to convert.")
    parser.add_argument("--output-dir", help="Optional output directory for batch conversion.")
    parser.add_argument("--sar-dir", help="Directory containing SAR TIFF images.")
    parser.add_argument("--rgb-dir", help="Directory containing RGB TIFF images.")
    parser.add_argument("--low-pct", type=float, default=2.0, help="Low percentile for contrast stretching.")
    parser.add_argument("--high-pct", type=float, default=98.0, help="High percentile for contrast stretching.")
    parser.add_argument("--include-zero", action="store_true", help="Include zero values in percentile statistics.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input directories.")
    parser.add_argument("--skip-existing", action="store_true", help="Do not overwrite existing PNG previews.")
    args = parser.parse_args()

    overwrite = not args.skip_existing
    ignore_zero = not args.include_zero

    if args.input:
        output = convert_tif_to_png(
            args.input,
            output_path=args.output,
            low_pct=args.low_pct,
            high_pct=args.high_pct,
            ignore_zero=ignore_zero,
            overwrite=overwrite,
        )
        _print_outputs([output] if output else [])
        return 0

    if args.sar_dir and args.rgb_dir:
        outputs = convert_paired_directories(
            args.sar_dir,
            args.rgb_dir,
            output_dir=args.output_dir,
            low_pct=args.low_pct,
            high_pct=args.high_pct,
            ignore_zero=ignore_zero,
            recursive=args.recursive,
            overwrite=overwrite,
        )
        _print_outputs(outputs)
        return 0

    if args.input_dir:
        outputs = convert_directory(
            args.input_dir,
            output_dir=args.output_dir,
            low_pct=args.low_pct,
            high_pct=args.high_pct,
            ignore_zero=ignore_zero,
            recursive=args.recursive,
            overwrite=overwrite,
        )
        _print_outputs(outputs)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
