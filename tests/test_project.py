"""Integration checks for the package and the committed numerical artifacts."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from galaxy_mass_estimation.data import ACTClusterDataset
from galaxy_mass_estimation.models.fno import ModelFNO1, ModelFNO2
from galaxy_mass_estimation.models.spectral import SpectralConv2d

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    def test_regressors_accept_images_and_redshift(self):
        torch.set_num_threads(1)
        images = torch.randn(2, 3, 148, 148)
        redshift = torch.tensor([[0.3], [0.8]])
        for model_class in (ModelFNO1, ModelFNO2):
            with self.subTest(model=model_class.__name__):
                model = model_class(modes1=2, modes2=2, width=4, depth=1, hidden_dim=16)
                model.eval()
                with torch.no_grad():
                    predictions = model(images, redshift)
                self.assertEqual(predictions.shape, (2,))
                self.assertTrue(torch.isfinite(predictions).all())

    def test_spectral_layer_has_finite_gradients(self):
        layer = SpectralConv2d(3, 4, 3, 4)
        image = torch.randn(2, 3, 16, 20, requires_grad=True)
        output = layer(image)
        self.assertEqual(output.shape, (2, 4, 16, 20))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertTrue(torch.isfinite(layer.weights.grad).all())


class DatasetTests(unittest.TestCase):
    def test_all_stored_image_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = np.arange(9978, dtype=np.float32) / 1000
            for suffix in ("", "_augm"):
                np.save(root / f"redshifts{suffix}.npy", labels)
                np.save(root / f"logM500cUncorr{suffix}.npy", labels + 1)
            cases = [
                (False, False, 0, "images", 0, 1),
                (False, True, 0, "images_normalized", 0, 2),
                (True, False, 0, "images_augm", 0, 3),
                (True, True, 9976, "images_normalized", 9976, 4),
                (True, True, 9977, "images_norm_augm", 9978, 5),
            ]
            for augm, norm, index, folder, filename, value in cases:
                with self.subTest(augm=augm, norm=norm, index=index):
                    (root / folder).mkdir(exist_ok=True)
                    np.save(
                        root / folder / f"{filename}.npy", np.full((3, 148, 148), value)
                    )
                    dataset = ACTClusterDataset(root, [index], augm=augm, norm=norm)
                    image, scalars, target = dataset[0]
                    self.assertEqual(len(dataset), 1)
                    self.assertEqual(image.shape, (3, 148, 148))
                    self.assertEqual(image.dtype, torch.float32)
                    self.assertTrue(torch.all(image == value))
                    self.assertEqual(scalars.shape, (1,))
                    self.assertAlmostEqual(scalars.item(), float(labels[index]))
                    self.assertAlmostEqual(target.item(), float(labels[index] + 1))


class SavedArtifactTests(unittest.TestCase):
    def test_metrics_match_saved_pairs(self):
        records = json.loads((PROJECT_ROOT / "results/metrics.json").read_text())
        self.assertEqual(len(records), 10)
        for record in records:
            folder = PROJECT_ROOT / record["path"]
            with self.subTest(experiment=record["path"]):
                targets = np.load(folder / "targets.npy").astype(np.float64).reshape(-1)
                predictions = (
                    np.load(folder / "predictions.npy").astype(np.float64).reshape(-1)
                )
                losses = np.load(folder / "losses.npy")
                self.assertEqual(targets.shape, predictions.shape)
                self.assertEqual(losses.shape, (100,))
                self.assertTrue(np.isfinite(losses).all())
                self.assertTrue(
                    np.isfinite(targets).all() and np.isfinite(predictions).all()
                )
                errors = predictions - targets
                self.assertAlmostEqual(
                    record["mae_dex"], np.abs(errors).mean(), places=10
                )
                self.assertAlmostEqual(
                    record["rmse_dex"], np.sqrt(np.square(errors).mean()), places=10
                )
                self.assertEqual(record["n_test"], len(targets))

    def test_split_indices_partition_the_sample_ids(self):
        for variant, size in [("raw", 9977), ("augmented", 14474)]:
            folder = PROJECT_ROOT / "data/splits" / variant
            train = np.load(folder / "train_indices.npy")
            test = np.load(folder / "test_indices.npy")
            with self.subTest(variant=variant):
                self.assertEqual(len(np.intersect1d(train, test)), 0)
                np.testing.assert_array_equal(
                    np.sort(np.concatenate([train, test])), np.arange(size)
                )


if __name__ == "__main__":
    unittest.main()
