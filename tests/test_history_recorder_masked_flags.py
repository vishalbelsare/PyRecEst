import unittest

import numpy as np

from pyrecest.utils import HistoryRecorder


class HistoryRecorderMaskedFlagTest(unittest.TestCase):
    def test_rejects_masked_boolean_flags(self):
        for hidden_payload in (False, True):
            masked_flag = np.ma.array(hidden_payload, mask=True)

            with self.subTest(flag="register.pad_with_nan", hidden=hidden_payload):
                recorder = HistoryRecorder()
                with self.assertRaisesRegex(TypeError, "pad_with_nan"):
                    recorder.register("padded", pad_with_nan=masked_flag)

            with self.subTest(flag="record.pad_with_nan", hidden=hidden_payload):
                recorder = HistoryRecorder()
                with self.assertRaisesRegex(TypeError, "pad_with_nan"):
                    recorder.record("padded", [1.0], pad_with_nan=masked_flag)

            with self.subTest(flag="record.copy_value", hidden=hidden_payload):
                recorder = HistoryRecorder()
                recorder.register("events")
                with self.assertRaisesRegex(TypeError, "copy_value"):
                    recorder.record("events", {"value": 1}, copy_value=masked_flag)

    def test_accepts_unmasked_masked_array_boolean(self):
        flag = np.ma.array(True, mask=False)
        recorder = HistoryRecorder()

        recorder.register("padded", pad_with_nan=flag)
        history = recorder.record("padded", [1.0], pad_with_nan=flag)

        self.assertEqual(tuple(history.shape), (1, 1))


if __name__ == "__main__":
    unittest.main()
