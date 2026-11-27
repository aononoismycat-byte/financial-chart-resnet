"""Synthetic unit tests for src.explain Grad-CAM, right_quarter_share and line_lift."""
import unittest

import numpy as np
import torch

from src.charts import render_chart
from src.explain import gradcam, line_lift, right_quarter_share
from src.model import ResNet18Classifier


class ExplainTest(unittest.TestCase):
    def test_gradcam_shape_and_range_frozen_backbone(self):
        torch.manual_seed(0)
        model = ResNet18Classifier(pretrained=False, training_mode="classifier_only")
        images = torch.randn(2, 3, 224, 224)
        cams = gradcam(model, images, target_class=1)
        self.assertEqual(cams.shape, (2, 224, 224))
        self.assertTrue(np.isfinite(cams).all())
        self.assertGreaterEqual(float(cams.min()), 0.0)
        self.assertLessEqual(float(cams.max()), 1.0)

    def test_right_quarter_share_and_line_lift(self):
        cam = np.zeros((224, 224), dtype=float)
        cam[:, 168:] = 1.0
        self.assertAlmostEqual(right_quarter_share(cam), 1.0, places=12)

        uniform = np.ones((224, 224), dtype=float)
        self.assertAlmostEqual(right_quarter_share(uniform), 56.0 / 224.0, places=12)

        img = render_chart(np.linspace(100.0, 110.0, 20))
        img_arr = np.asarray(img)
        focused_cam = np.zeros((224, 224), dtype=float)
        focused_cam[np.any(img_arr < 128, axis=-1)] = 1.0
        lift = line_lift(focused_cam, img)
        self.assertGreater(lift, 1.0)
        self.assertAlmostEqual(line_lift(uniform, img), 1.0, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
