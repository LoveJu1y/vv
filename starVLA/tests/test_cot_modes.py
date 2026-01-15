import unittest

from starVLA.training.trainer_utils.cot_mode_utils import CotMode, derive_flags_from_mode


class CotModeFlagsTest(unittest.TestCase):
    def test_enable_latent_reasoning(self):
        self.assertFalse(derive_flags_from_mode(CotMode.NONE)["enable_latent_reasoning"])
        self.assertTrue(derive_flags_from_mode(CotMode.IMPLICIT)["enable_latent_reasoning"])

    def test_emit_thinking_tokens(self):
        self.assertFalse(derive_flags_from_mode(CotMode.EXPLICIT)["emit_thinking_tokens"])
        self.assertTrue(derive_flags_from_mode(CotMode.IMPLICIT)["emit_thinking_tokens"])

    def test_reasoning_stage_mapping(self):
        self.assertEqual(derive_flags_from_mode(CotMode.NONE)["reasoning_stage"], 0)
        self.assertEqual(derive_flags_from_mode(CotMode.VLM_SEEN_NO_OUT)["reasoning_stage"], 1)
        self.assertEqual(derive_flags_from_mode(CotMode.EXPLICIT)["reasoning_stage"], 1)
        self.assertEqual(derive_flags_from_mode(CotMode.IMPLICIT)["reasoning_stage"], 4)


if __name__ == "__main__":
    unittest.main()
