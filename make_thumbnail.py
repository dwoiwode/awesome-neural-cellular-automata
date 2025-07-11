import argparse
import subprocess
from pathlib import Path


def create_thumbnail(
        path: Path | str,
        identifier: str,
        thumbnail_width: int,
        output_dir: Path | str = "./assets/thumbnails/"
):
    path = Path(path)
    output_dir = Path(output_dir)
    try:
        # Check if the input file exists
        if not path.is_file():
            print(f"Error: The file {path} was not found. (Absolute path: {path.absolute().resolve()})")
            return

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define output path
        output_path = output_dir / f"{identifier}.jpg"

        # Use the magick command to create a thumbnail
        if path.name.endswith(".pdf"):
            cmd =[
                "magick", "convert",
                "-background", "white",
                "-flatten", f"{path.absolute().resolve()}[0]",
                "-resize", f"{thumbnail_width}x",
            ]
        else:
            # Expecting image
            cmd = [
                "magick", "convert",
                str(path.absolute().resolve()),
                "-thumbnail", f"{thumbnail_width}x"
            ]

        # Add output path and execute command
        cmd += [str(output_path.absolute().resolve())]
        subprocess.run(cmd, check=True)

        print(f"Thumbnail successfully created at: {output_path}")

    except subprocess.CalledProcessError as e:
        print(f"Error during image processing: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a thumbnail from an image using ImageMagick.")
    parser.add_argument(
        "path",
        type=Path,
        help="Path to the image or pdf file to be converted into a thumbnail."
    )
    parser.add_argument(
        "identifier",
        type=str,
        help="Unique identifier for naming the thumbnail."
    )
    parser.add_argument(
        "-w", "--thumbnail_width",
        type=int,
        default=300,
        help="Desired width of the thumbnail."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=Path,
        default="./assets/thumbnails/",
        help="Directory where the thumbnail will be saved."
    )

    args = parser.parse_args()

    create_thumbnail(
        path=args.path,
        identifier=args.identifier,
        thumbnail_width=args.thumbnail_width,
        output_dir=args.output_dir
    )
