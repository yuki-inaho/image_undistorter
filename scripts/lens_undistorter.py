import cv2
import toml
import numpy as np
from scripts.camera_parameter import IntrinsicParameter
from pathlib import Path


class LensUndistorter:
    def __init__(self, toml_path):
        self._setting(toml_path)

    def _setting(self, toml_path):
        toml_dict = toml.load(open(toml_path))
        # Set Camera Settings
        self._image_width = toml_dict["Rgb"]["width"]
        self._image_height = toml_dict["Rgb"]["height"]
        self._DIM = (self._image_width, self._image_height)

        # Set Camera Parameters
        intrinsic_elems = ["fx", "fy", "cx", "cy"]
        self._intrinsic_params = IntrinsicParameter()
        self._intrinsic_params.set_intrinsic_parameter(
            *[toml_dict["Rgb"][elem] for elem in intrinsic_elems]
        )
        self._intrinsic_params.set_image_size(
            *[toml_dict["Rgb"][elem] for elem in ["width", "height"]])
        self._K_rgb_raw = np.array(
            [[self._intrinsic_params.fx, 0, self._intrinsic_params.cx],
             [0, self._intrinsic_params.fy, self._intrinsic_params.cy],
             [0, 0, 1]]
        )

        self._distortion_params = np.array(
            [toml_dict["Rgb"]["k{}".format(i+1)] for i in range(4)]
        )

        self._K_rgb = cv2.getOptimalNewCameraMatrix(
            self._K_rgb_raw, self._distortion_params,
            self._DIM, 0
        )[0]
        _map1, _map2 = cv2.fisheye.initUndistortRectifyMap(
            self._K_rgb_raw, self._distortion_params, np.eye(3),
            self._K_rgb, self._DIM, cv2.CV_16SC2
        )

        self._map1 = _map1
        self._map2 = _map2
        self._P_rgb = (self._K_rgb[0][0], 0., self._K_rgb[0][2], 0.,
                       0., self._K_rgb[1][1], self._K_rgb[1][2], 0.,
                       0., 0., 1., 0.)

    def correction(self, image):
        return cv2.remap(image, self._map1, self._map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def correction_with_mask(self, mask):
        return cv2.remap(mask, self._map1, self._map2, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    @property
    def K(self):
        return self._K_rgb

    @property
    def P(self):
        return self._P_rgb
