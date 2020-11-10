from pathlib import Path
from scripts.lens_undistorter import LensUndistorter
import click
import cv2
from tqdm import tqdm

SCRIPT_DIR = str(Path(__file__).parent)


@click.command()
@click.option("--input-dir", "-i", default=f"{SCRIPT_DIR}/data")
@click.option("--output-dir", "-o", default=f"{SCRIPT_DIR}/undistorted")
@click.option("--config-file", "-c", default=f"{SCRIPT_DIR}/cfg/camera_parameter.toml")
def main(input_dir, output_dir, config_file):
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir()

    input_image_pathes = Path(input_dir).glob("*.png")
    input_image_path_list = [str(input_image_path)
                             for input_image_path in input_image_pathes]

    lens_undistorter = LensUndistorter(config_file)

    for input_image_path in tqdm(input_image_path_list):
        base_name = Path(input_image_path).name
        image = cv2.imread(input_image_path)
        image_undist = lens_undistorter.correction(image)
        output_image_path = str(Path(output_dir_path, base_name))
        cv2.imwrite(output_image_path, image_undist)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()
