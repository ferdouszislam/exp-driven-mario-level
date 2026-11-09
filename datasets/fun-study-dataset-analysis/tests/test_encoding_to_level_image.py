"""Behavior checks for decoding, literal tiles, and the final path overlay."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from encoding_to_level_image import (
    BACKGROUND, decode_encoding, encoding_to_image, level_to_image, render_encoding_file,
)


class EncodingImageTests(unittest.TestCase):
    def setUp(self):
        # Explicit exported channel IDs: coin=6, empty=2, normal pipe=4,
        # solid alias=19, stalk=20, flower pipe=5, path=21.
        ids = np.array([[6, 2, 4], [19, 20, 5]])
        self.tile_only = np.eye(21, dtype=np.uint8)[ids].transpose(2, 0, 1)
        self.mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        self.multi = np.concatenate([self.tile_only, self.mask[None]], axis=0)

    def test_one_hot_and_multi_hot_decode_exact_characters(self):
        expected = np.array([["o", "-", "t"], ["#", "|", "T"]])
        tiles, path = decode_encoding(self.multi)
        np.testing.assert_array_equal(tiles, expected)
        np.testing.assert_array_equal(path, self.mask)
        tiles, path = decode_encoding(self.tile_only)
        np.testing.assert_array_equal(tiles, expected)
        self.assertFalse(path.any())

    def test_black_x_is_above_coin_and_unvisited_cells_are_unchanged(self):
        original = self.multi.copy()
        plain = np.asarray(encoding_to_image(self.tile_only))
        overlaid = np.asarray(encoding_to_image(self.multi))
        coin_before, coin_after = plain[:16, :16], overlaid[:16, :16]
        # Some original coin pixels are already black. Require the overlay to
        # cover at least one originally colored sprite pixel, beyond background.
        colored = np.any(coin_before != BACKGROUND, axis=2) & np.any(coin_before != 0, axis=2)
        self.assertTrue(np.any(colored & np.all(coin_after == 0, axis=2)))
        for row, column in zip(*np.nonzero(self.mask)):
            # Both diagonals and their crossing must be visible, irrespective of tile.
            for dx, dy in [(4, 4), (11, 11), (4, 11), (11, 4), (8, 8)]:
                np.testing.assert_array_equal(overlaid[row * 16 + dy, column * 16 + dx], [0, 0, 0])
        for row, column in zip(*np.where(self.mask == 0)):
            area = np.s_[row * 16:(row + 1) * 16, column * 16:(column + 1) * 16]
            np.testing.assert_array_equal(overlaid[area], plain[area])
        np.testing.assert_array_equal(self.multi, original)

    def test_hide_path_and_scale(self):
        plain = np.asarray(encoding_to_image(self.tile_only))
        np.testing.assert_array_equal(encoding_to_image(self.multi, show_path=False), plain)
        small = np.asarray(encoding_to_image(self.multi))
        large = np.asarray(encoding_to_image(self.multi, scale=2))
        np.testing.assert_array_equal(large, small.repeat(2, axis=0).repeat(2, axis=1))

    def test_asymmetric_pipe_does_not_paint_empty_right_column(self):
        tiles = np.array([["t", "t"], ["t", "-"], ["t", "-"]])
        image = np.asarray(level_to_image(tiles))
        for row in [1, 2]:
            np.testing.assert_array_equal(image[row * 16:(row + 1) * 16, 16:32],
                                          np.full((16, 16, 3), BACKGROUND, dtype=np.uint8))

    def test_stalks_are_rendered_only_when_present(self):
        without = np.asarray(level_to_image(np.array([["%"], ["-"]])))
        with_stalk = np.asarray(level_to_image(np.array([["%"], ["|"]])))
        np.testing.assert_array_equal(without[16:32], np.full((16, 16, 3), BACKGROUND, dtype=np.uint8))
        self.assertFalse(np.array_equal(with_stalk[16:32], without[16:32]))

    def test_invalid_encodings_are_rejected(self):
        invalid = self.multi.copy()
        invalid[0, 0, 0] = 1  # A second active tile, alongside coin.
        with self.assertRaisesRegex(ValueError, "exactly one"):
            decode_encoding(invalid)
        invalid = self.multi.astype(float)
        invalid[21, 0, 0] = 0.5
        with self.assertRaisesRegex(ValueError, "binary"):
            decode_encoding(invalid)
        with self.assertRaises(ValueError):
            decode_encoding(self.multi.transpose(1, 2, 0))

    def test_file_api_saves_png_and_never_replaces_input(self):
        with tempfile.TemporaryDirectory() as temp:
            source, destination = Path(temp) / "level.npy", Path(temp) / "level.png"
            np.save(source, self.multi)
            original = source.read_bytes()
            image = render_encoding_file(source, destination)
            self.assertEqual(image.size, (48, 32))
            self.assertTrue(destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            with self.assertRaisesRegex(ValueError, "must not replace"):
                render_encoding_file(source, source)
            self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
