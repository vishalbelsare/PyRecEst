import unittest

from pyrecest.filters.update_diagnostics import MeasurementUpdateDiagnostics


class _BackendBooleanScalar:
    dtype = "torch.bool"

    def __index__(self):
        return 1


class MeasurementUpdateDiagnosticsBackendBooleanTest(unittest.TestCase):
    def test_rejects_backend_boolean_measurement_count(self):
        with self.assertRaisesRegex(ValueError, "measurement_count"):
            MeasurementUpdateDiagnostics(
                measurement_count=_BackendBooleanScalar(),
            )

    def test_rejects_backend_boolean_active_index(self):
        with self.assertRaisesRegex(ValueError, "active_measurement_indices"):
            MeasurementUpdateDiagnostics(
                active_measurement_indices=[_BackendBooleanScalar()],
            )


if __name__ == "__main__":
    unittest.main()
