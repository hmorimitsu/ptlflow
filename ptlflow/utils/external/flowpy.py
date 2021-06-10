"""This code is mostly taken and adapted from flowpy.

https://gitlab-research.centralesupelec.fr/2018seznecm/flowpy
"""

#
# Original license below
#
# MIT License
#
# Copyright (c) 2020 Univ. Paris-Saclay, CNRS, CentraleSupelec,
#                    Thales Research & Technology
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
from pathlib import Path
from warnings import warn
import png
import struct
from collections import namedtuple
from itertools import accumulate

DEFAULT_TRANSITIONS = (15, 6, 4, 11, 13, 6)


def flow_to_rgb(flow, flow_max_radius=None, background="bright", custom_colorwheel=None):
    """
    Creates a RGB representation of an optical flow.

    Parameters
    ----------
    flow: numpy.ndarray
        3D flow in the HWF (Height, Width, Flow) layout.
        flow[..., 0] should be the x-displacement
        flow[..., 1] should be the y-displacement

    flow_max_radius: float, optional
        Set the radius that gives the maximum color intensity, useful for comparing different flows.
        Default: The normalization is based on the input flow maximum radius.

    background: str, optional
        States if zero-valued flow should look 'bright' or 'dark'
        Default: "bright"

    custom_colorwheel: numpy.ndarray
        Use a custom colorwheel for specific hue transition lengths.
        By default, the default transition lengths are used.

    Returns
    -------
    rgb_image: numpy.ndarray
        A 2D RGB image that represents the flow

    Raises
    ------
    ValueError
        If the background is invalid.

    See Also
    --------
    make_colorwheel

    """
    valid_backgrounds = ("bright", "dark")
    if background not in valid_backgrounds:
        raise ValueError(f"background should be one the following: {valid_backgrounds}, not {background}")

    wheel = make_colorwheel() if custom_colorwheel is None else custom_colorwheel

    complex_flow = flow[..., 0] + 1j * flow[..., 1]
    complex_flow, nan_mask = replace_nans(complex_flow)

    radius, angle = np.abs(complex_flow), np.angle(complex_flow)

    if flow_max_radius is None:
        flow_max_radius = np.max(radius)

    if flow_max_radius > 0:
        radius /= flow_max_radius

    ncols = len(wheel)

    # Map the angles from (-pi, pi] to [0, 2pi) to [0, ncols - 1)
    angle[angle < 0] += 2 * np.pi
    angle = angle * ((ncols - 1) / (2 * np.pi))

    # Make the wheel cyclic for interpolation
    wheel = np.vstack((wheel, wheel[0]))

    # Interpolate the hues
    (angle_fractional, angle_floor), angle_ceil = np.modf(angle), np.ceil(angle)
    angle_fractional = angle_fractional.reshape((angle_fractional.shape) + (1,))
    float_hue = (wheel[angle_floor.astype(np.int32)] * (1 - angle_fractional)
                 + wheel[angle_ceil.astype(np.int32)] * angle_fractional)

    ColorizationArgs = namedtuple("ColorizationArgs", [
        'move_hue_valid_radius',
        'move_hue_oversized_radius',
        'invalid_color'])

    def move_hue_on_V_axis(hues, factors):
        return hues * np.expand_dims(factors, -1)

    def move_hue_on_S_axis(hues, factors):
        return 255. - np.expand_dims(factors, -1) * (255. - hues)

    if background == "dark":
        parameters = ColorizationArgs(move_hue_on_V_axis, move_hue_on_S_axis,
                                      np.array([255, 255, 255], dtype=np.float32))
    else:
        parameters = ColorizationArgs(move_hue_on_S_axis, move_hue_on_V_axis,
                                      np.array([0, 0, 0], dtype=np.float32))

    colors = parameters.move_hue_valid_radius(float_hue, radius)

    oversized_radius_mask = radius > 1
    colors[oversized_radius_mask] = parameters.move_hue_oversized_radius(
        float_hue[oversized_radius_mask],
        1 / radius[oversized_radius_mask]
    )
    colors[nan_mask] = parameters.invalid_color

    return colors.astype(np.uint8)


def make_colorwheel(transitions=DEFAULT_TRANSITIONS):
    """
    Creates a color wheel.

    A color wheel defines the transitions between the six primary hues:
    Red(255, 0, 0), Yellow(255, 255, 0), Green(0, 255, 0), Cyan(0, 255, 255), Blue(0, 0, 255) and Magenta(255, 0, 255).

    Parameters
    ----------
    transitions: sequence_like
        Contains the length of the six transitions.
        Defaults to (15, 6, 4, 11, 13, 6), based on humain perception.

    Returns
    -------
    colorwheel: numpy.ndarray
        The RGB values of the transitions in the color space.

    Notes
    -----
    For more information, take a look at
    https://web.archive.org/web/20051107102013/http://members.shaw.ca/quadibloc/other/colint.htm

    """

    colorwheel_length = sum(transitions)

    # The red hue is repeated to make the color wheel cyclic
    base_hues = map(np.array,
                    ([255, 0, 0], [255, 255, 0], [0, 255, 0],
                     [0, 255, 255], [0, 0, 255], [255, 0, 255],
                     [255, 0, 0]))

    colorwheel = np.zeros((colorwheel_length, 3), dtype="uint8")
    hue_from = next(base_hues)
    start_index = 0
    for hue_to, end_index in zip(base_hues, accumulate(transitions)):
        transition_length = end_index - start_index

        colorwheel[start_index:end_index] = np.linspace(
            hue_from, hue_to, transition_length, endpoint=False)
        hue_from = hue_to
        start_index = end_index

    return colorwheel


def replace_nans(array, value=0):
    nan_mask = np.isnan(array)
    array[nan_mask] = value

    return array, nan_mask


def flow_write(output_file, flow, format=None):
    """
    Writes optical flow to file.

    Parameters
    ----------
    output_file: {str, pathlib.Path, file}
        Path of the file to write or file object.
    flow: numpy.ndarray
        3D flow in the HWF (Height, Width, Flow) layout.
        flow[..., 0] should be the x-displacement
        flow[..., 1] should be the y-displacement
    format: str, optional
        Specify in what format the flow is written, accepted formats: "png" or "flo"
        If None, it is guessed on the file extension

    See Also
    --------
    flow_read

    """

    output_format = guess_extension(output_file, override=format)

    with FileManager(output_file, "wb") as f:
        if output_format == "png":
            flow_write_png(f, flow)
        else:
            flow_write_flo(f, flow)


def flow_read(input_file, format=None):
    """
    Reads optical flow from file

    Parameters
    ----------
    input_file: {str, pathlib.Path, file}
        Path of the file to read or file object.
    format: str, optional
        Specify in what format the flow is read, accepted formats: "png" or "flo"
        If None, it is guess on the file extension

    Returns
    -------
    flow: numpy.ndarray
        3D flow in the HWF (Height, Width, Flow) layout.
        flow[..., 0] is the x-displacement
        flow[..., 1] is the y-displacement

    Notes
    -----

    The flo format is dedicated to optical flow and was first used in Middlebury optical flow database.
    The original definition can be found here: http://vision.middlebury.edu/flow/code/flow-code/flowIO.cpp

    The png format uses 16-bit RGB png to store optical flows.
    It was developped along with the KITTI Vision Benchmark Suite.
    More information can be found here: http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow

    The both handle flow with invalid ``invalid'' values, to deal with occlusion for example.
    We convert such invalid values to NaN.

    See Also
    --------
    flow_write

    """

    input_format = guess_extension(input_file, override=format)

    with FileManager(input_file, "rb") as f:
        if input_format == "png":
            output = flow_read_png(f)
        else:
            output = flow_read_flo(f)

    return output


def flow_read_flo(f):
    if (f.read(4) != b'PIEH'):
        warn(f"{f.name} does not have a .flo file signature")

    width, height = struct.unpack("II", f.read(8))
    result = np.fromfile(f, dtype="float32").reshape((height, width, 2))

    # Set invalid flows to NaN
    mask_u = np.greater(np.abs(result[..., 0]), 1e9, where=(~np.isnan(result[..., 0])))
    mask_v = np.greater(np.abs(result[..., 1]), 1e9, where=(~np.isnan(result[..., 1])))

    result[mask_u | mask_v] = np.NaN

    return result


def flow_write_flo(f, flow):
    SENTINEL = 1666666800.0  # Only here to look like Middlebury original files
    height, width, _ = flow.shape

    image = flow.copy()
    image[np.isnan(image)] = SENTINEL

    f.write(b'PIEH')
    f.write(struct.pack("II", width, height))
    image.astype(np.float32).tofile(f)


def flow_read_png(f):
    width, height, stream, *_ = png.Reader(f).read()

    file_content = np.concatenate(list(stream)).reshape((height, width, 3))
    flow, valid = file_content[..., 0:2], file_content[..., 2]

    flow = (flow.astype(np.float32) - 2 ** 15) / 64.

    flow[~valid.astype(bool)] = np.NaN

    return flow


def flow_write_png(f, flow):
    SENTINEL = 0.  # Only here to look like original KITTI files
    height, width, _ = flow.shape
    flow_copy = flow.copy()

    valid = ~(np.isnan(flow[..., 0]) | np.isnan(flow[..., 1]))
    flow_copy[~valid] = SENTINEL

    flow_copy = (flow_copy * 64. + 2 ** 15).astype(np.uint16)
    image = np.dstack((flow_copy, valid))

    writer = png.Writer(width, height, bitdepth=16, greyscale=False)
    writer.write(f, image.reshape((height, 3 * width)))


class FileManager:
    def __init__(self, abstract_file, mode):
        self.abstract_file = abstract_file
        self.opened_file = None
        self.mode = mode

    def __enter__(self):
        if isinstance(self.abstract_file, str):
            self.opened_file = open(self.abstract_file, self.mode)
        elif isinstance(self.abstract_file, Path):
            self.opened_file = self.abstract_file.open(self.mode)
        else:
            return self.abstract_file

        return self.opened_file

    def __exit__(self, exc_type, exc_value, traceback):
        if self.opened_file is not None:
            self.opened_file.close()


def guess_extension(abstract_file, override=None):
    if override is not None:
        return override

    if isinstance(abstract_file, str):
        return Path(abstract_file).suffix[1:]
    elif isinstance(abstract_file, Path):
        return abstract_file.suffix[1:]

    return Path(abstract_file.name).suffix[1:]
