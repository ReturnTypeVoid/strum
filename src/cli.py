"""
STRUM CLI - Main entry point for all commands.

Usage:
    strum preprocess --input-dir ./raw --output-dir ./processed
    strum train drums --config configs/drums.yaml
    strum infer drums --input song.wav --output song_drums.mid
    strum batch --manifest manifest.json --workers 4
"""

import click
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="strum")
def main() -> None:
    """STRUM: Spectral Transcription & Rhythm Understanding Model."""
    pass


@main.command()
@click.option(
    "--input-dir", "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing raw audio files and charts"
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory for processed output (stems, manifest)"
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    default="configs/preprocessing.yaml",
    help="Preprocessing config file"
)
@click.option(
    "--instruments",
    type=click.Choice(["drums", "guitar", "bass", "vocals", "keys", "all"]),
    multiple=True,
    default=["drums"],
    help="Instruments to process (can specify multiple)"
)
def preprocess(
    input_dir: Path,
    output_dir: Path,
    config: Path,
    instruments: tuple[str, ...],
) -> None:
    """Preprocess audio files: separate stems, parse charts, create dataset."""
    from src.preprocessing.pipeline import run_preprocessing
    
    console.print(f"[bold blue]STRUM Preprocessing[/bold blue]")
    console.print(f"  Input:  {input_dir}")
    console.print(f"  Output: {output_dir}")
    console.print(f"  Instruments: {', '.join(instruments)}")
    
    run_preprocessing(
        input_dir=input_dir,
        output_dir=output_dir,
        config_path=config,
        instruments=list(instruments),
    )


@main.command()
@click.argument("instrument", type=click.Choice(["drums", "guitar", "bass", "vocals", "keys"]))
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Training config file (e.g., configs/drums.yaml)"
)
@click.option(
    "--resume", "-r",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Resume from checkpoint"
)
@click.option(
    "--manifest", "-m",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Dataset manifest file (default: from config)"
)
def train(
    instrument: str,
    config: Path,
    resume: Optional[Path],
    manifest: Optional[Path],
) -> None:
    """Train a model for the specified instrument."""
    console.print(f"[bold green]STRUM Training: {instrument}[/bold green]")
    console.print(f"  Config: {config}")
    if resume:
        console.print(f"  Resuming from: {resume}")
    
    if instrument == "drums":
        from src.models.drums import train_drums
        train_drums(
            config_path=config,
            checkpoint_path=resume,
            manifest_path=manifest,
        )
    else:
        console.print(f"[yellow]Training for {instrument} not yet implemented[/yellow]")


@main.command()
@click.argument("instrument", type=click.Choice(["drums", "guitar", "bass", "vocals", "keys", "all"]))
@click.option(
    "--input", "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input audio file"
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output chart file (.mid or .chart)"
)
@click.option(
    "--model", "-m",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Model checkpoint (default: best checkpoint from training)"
)
@click.option(
    "--stems-dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Pre-separated stems directory (skips Demucs)"
)
@click.option(
    "--tempo",
    type=float,
    default=120.0,
    help="Chart tempo in BPM"
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Detection threshold (instrument-specific defaults if not set)"
)
def infer(
    instrument: str,
    input: Path,
    output: Path,
    model: Optional[Path],
    stems_dir: Optional[Path],
    tempo: float,
    threshold: Optional[float],
) -> None:
    """Generate chart from audio file."""
    console.print(f"[bold cyan]STRUM Inference: {instrument}[/bold cyan]")
    console.print(f"  Input:  {input}")
    console.print(f"  Output: {output}")
    
    from src.inference.unified import UnifiedInference, InferenceConfig
    
    # Build config
    config = InferenceConfig(tempo_bpm=tempo)
    
    if model:
        if instrument in ['guitar', 'all']:
            config.guitar_checkpoint = str(model)
        if instrument in ['bass', 'all']:
            config.bass_checkpoint = str(model)
        if instrument in ['drums', 'all']:
            config.drums_checkpoint = str(model)
    
    if threshold is not None:
        if instrument in ['guitar', 'all']:
            config.guitar_threshold = threshold
        if instrument in ['bass', 'all']:
            config.bass_threshold = threshold
        if instrument in ['drums', 'all']:
            config.drums_threshold = threshold
    
    # Map instrument to list
    if instrument == 'all':
        instruments = ['drums', 'guitar', 'bass', 'vocals']
    else:
        instruments = [instrument]
    
    engine = UnifiedInference(config=config)
    engine.infer(
        audio_path=input,
        output_path=output,
        stems_dir=stems_dir,
        instruments=instruments,
    )


@main.command()
@click.option(
    "--input", "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input audio file (MP3, WAV, etc.)"
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(path_type=Path),
    default=Path("output"),
    help="Base output directory (song folder will be created inside)"
)
@click.option(
    "--instruments",
    type=click.Choice(["drums", "guitar", "bass", "vocals", "all"]),
    multiple=True,
    default=["all"],
    help="Instruments to chart"
)
@click.option(
    "--tempo",
    type=float,
    default=120.0,
    help="Chart tempo in BPM"
)
def chart(
    input: Path,
    output_dir: Path,
    instruments: tuple[str, ...],
    tempo: float,
) -> None:
    """Generate a full Clone Hero / YARG chart package from audio."""
    console.print(f"[bold cyan]STRUM Chart Generation[/bold cyan]")
    console.print(f"  Input: {input}")
    console.print(f"  Output: {output_dir}")
    
    from src.inference.unified import UnifiedInference, InferenceConfig
    
    # Build config
    config = InferenceConfig(tempo_bpm=tempo)
    
    # Map instruments
    if 'all' in instruments:
        instrument_list = ['drums', 'guitar', 'bass', 'vocals']
    else:
        instrument_list = list(instruments)
    
    # Create song folder name from filename
    name = input.stem
    # Clean up common suffixes
    for suffix in ["(Lyrics)", "(Official Video)", "(Audio)", "(Official Audio)", "[Official Video]"]:
        name = name.replace(suffix, "").strip()
    
    song_folder = output_dir / name
    console.print(f"  Song folder: {song_folder}")
    
    engine = UnifiedInference(config=config)
    engine.chart(
        audio_path=input,
        output_dir=song_folder,
        instruments=instrument_list,
    )


@main.command()
@click.option(
    "--manifest", "-m",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Manifest file with list of songs to process"
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for generated charts"
)
@click.option(
    "--workers", "-w",
    type=int,
    default=4,
    help="Number of parallel workers"
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    default="configs/inference.yaml",
    help="Inference config file"
)
def batch(
    manifest: Path,
    output_dir: Path,
    workers: int,
    config: Path,
) -> None:
    """Batch process multiple songs from a manifest."""
    console.print(f"[bold magenta]STRUM Batch Processing[/bold magenta]")
    console.print(f"  Manifest: {manifest}")
    console.print(f"  Output:   {output_dir}")
    console.print(f"  Workers:  {workers}")
    
    from src.inference.pipeline import run_batch_inference
    
    run_batch_inference(
        manifest_path=manifest,
        output_dir=output_dir,
        config_path=config,
        num_workers=workers,
    )


@main.command()
@click.option(
    "--manifest", "-m",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Dataset manifest file"
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output report file (default: stdout)"
)
@click.option(
    "--instrument",
    type=click.Choice(["drums", "guitar", "bass", "vocals", "keys", "all"]),
    default="all",
    help="Instrument to evaluate"
)
def evaluate(
    manifest: Path,
    output: Optional[Path],
    instrument: str,
) -> None:
    """Evaluate model performance on test set."""
    console.print(f"[bold yellow]STRUM Evaluation[/bold yellow]")
    console.print(f"  Manifest: {manifest}")
    console.print(f"  Instrument: {instrument}")
    
    from src.evaluation.metrics import run_evaluation
    
    run_evaluation(
        manifest_path=manifest,
        output_path=output,
        instrument=instrument,
    )


if __name__ == "__main__":
    main()
