"""Check validation grouping and safe migration of the source-derived export."""

import ast
import json
from pathlib import Path
import random
import tempfile
import unittest

from preprocess_dataset import (
    DEFAULT_SOURCE, PROJECT_ROOT, assign_splits, assign_validation_split,
    build_output, collect_pairs, migrate_output, split_training_groups, verify_output,
)


class ValidationSplitTests(unittest.TestCase):
    def test_validation_membership_matches_training_script(self):
        # Load the original policy without requiring the training dependencies.
        tree = ast.parse((PROJECT_ROOT.parent.parent / "model-training/train.py").read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "split_training_groups")
        namespace = {"random": random}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "train.py", "exec"), namespace)
        pairs, *_ = collect_pairs(DEFAULT_SOURCE)
        summary = assign_splits(pairs, 42)
        original_train = [pair for pair in pairs if pair["split"] == "train_set"]
        expected_train, expected_validation = namespace["split_training_groups"](original_train, 0.2, 42)
        original_test = [dict(pair) for pair in pairs if pair["split"] == "test_set"]
        assign_validation_split(pairs, summary, 0.2, 42)
        self.assertEqual([pair for pair in pairs if pair["split"] == "train_set"], expected_train)
        self.assertEqual([pair for pair in pairs if pair["split"] == "validation_set"], expected_validation)
        self.assertEqual([pair for pair in pairs if pair["split"] == "test_set"], original_test)
        self.assertEqual(summary["instances"], {"train_set": 232, "validation_set": 58, "test_set": 73})
        self.assertEqual(summary["original_train_test_split"]["instances"],
                         {"train_set": 290, "test_set": 73})
        for counts in summary["shared_between_sets"].values():
            for field in ("split_group_ids", "source_annotation_ids", "annotation_reuse_groups", "playthrough_keys"):
                self.assertEqual(counts[field], 0)

    def test_custom_fractions_preserve_groups_and_leave_training_data(self):
        entries = [{"split_group_id": group, "instance_id": f"{group}-{index}"}
                   for group, size in (("a", 3), ("b", 3), ("c", 6))
                   for index in range(size)]
        for fraction in (0.1, 0.3, 0.99):
            fitting, validation = split_training_groups(entries, fraction, 42)
            self.assertTrue(fitting)
            self.assertTrue(validation)
            self.assertEqual(len(fitting) + len(validation), len(entries))
            self.assertFalse({e["split_group_id"] for e in fitting}
                             & {e["split_group_id"] for e in validation})
            self.assertEqual((fitting, validation), split_training_groups(entries, fraction, 42))
        self.assertEqual(split_training_groups(entries, 0, 42), (entries, []))
        single_group = entries[:3]
        self.assertEqual(split_training_groups(single_group, 0.2, 42), (single_group, []))
        for invalid in (-0.1, 1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                split_training_groups(entries, invalid, 42)


class MigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy, _ = build_output(DEFAULT_SOURCE, validation_fraction=0)
        cls.expected, _ = build_output(DEFAULT_SOURCE)

    def write_legacy(self, root):
        for name, content in self.legacy.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

    def test_migration_preserves_contents_and_updates_all_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_legacy(root)
            migrate_output(root, self.expected, self.legacy)
            verify_output(root, self.expected)
            old_entries = {item["instance_id"]: item for item in
                           json.loads(self.legacy["manifest.json"])["instances"]}
            for entry in json.loads(self.expected["manifest.json"])["instances"]:
                old_folder = old_entries[entry["instance_id"]]["relative_directory"]
                new_folder = entry["relative_directory"]
                for name, content in self.expected.items():
                    if name.startswith(new_folder + "/"):
                        self.assertEqual(content, self.legacy[old_folder + name[len(new_folder):]])
            self.assertEqual(len(list((root / "validation_set").iterdir())), 58)

    def test_migration_rejects_changed_files_before_moving(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_legacy(root)
            name = next(name for name in self.legacy if name.endswith("ranking.txt"))
            (root / name).write_bytes(b"changed\n")
            with self.assertRaisesRegex(ValueError, "differs from source reconstruction"):
                migrate_output(root, self.expected, self.legacy)
            self.assertFalse((root / "validation_set").exists())
            self.assertEqual((root / "manifest.json").read_bytes(), self.legacy["manifest.json"])
            self.assertEqual((root / name).read_bytes(), b"changed\n")


if __name__ == "__main__":
    unittest.main()
