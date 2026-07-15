import unittest
from unittest.mock import patch

from pyrecest.backend import array
from pyrecest.distributions.abstract_se3_distribution import AbstractSE3Distribution


class _RecordingAxes:
    def __init__(self, x_limits, y_limits, z_limits):
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.z_limits = z_limits
        self.set_xlim_calls = []
        self.set_ylim_calls = []
        self.set_zlim_calls = []

    @staticmethod
    def quiver(*_args, **_kwargs):
        return object()

    def get_xlim(self):
        return self.x_limits

    def get_ylim(self):
        return self.y_limits

    def get_zlim(self):
        return self.z_limits

    def set_xlim(self, limits):
        self.set_xlim_calls.append(tuple(float(value) for value in limits))

    def set_ylim(self, limits):
        self.set_ylim_calls.append(tuple(float(value) for value in limits))

    def set_zlim(self, limits):
        self.set_zlim_calls.append(tuple(float(value) for value in limits))


class _RecordingFigure:
    def __init__(self, axes):
        self.axes = axes

    def add_subplot(self, *_args, **_kwargs):
        return self.axes


class TestAbstractSE3PlotLimits(unittest.TestCase):
    @staticmethod
    def _plot_with_axes(axes):
        point = array([1.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0])
        with patch(
            "pyrecest.distributions.abstract_se3_distribution.plt.figure",
            return_value=_RecordingFigure(axes),
        ):
            AbstractSE3Distribution.plot_point(point)

    def test_plot_point_preserves_limits_containing_body_axes(self):
        axes = _RecordingAxes(
            (-100.0, 100.0),
            (-100.0, 100.0),
            (-100.0, 100.0),
        )

        self._plot_with_axes(axes)

        self.assertEqual(axes.set_xlim_calls, [])
        self.assertEqual(axes.set_ylim_calls, [])
        self.assertEqual(axes.set_zlim_calls, [])

    def test_plot_point_expands_limits_inside_required_bounds(self):
        axes = _RecordingAxes(
            (10.25, 10.75),
            (20.25, 20.75),
            (30.25, 30.75),
        )

        self._plot_with_axes(axes)

        self.assertEqual(axes.set_xlim_calls, [(5.0, 15.0)])
        self.assertEqual(axes.set_ylim_calls, [(15.0, 25.0)])
        self.assertEqual(axes.set_zlim_calls, [(25.0, 35.0)])


if __name__ == "__main__":
    unittest.main()
