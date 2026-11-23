import csv
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_api_classifier as api

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'data_json' / 'four_batches'


class PackageTests(unittest.TestCase):
    def test_current_index_matches_packages(self):
        with (PACKAGE / 'index_current.csv').open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 260)
        self.assertEqual(len({row['experiment_id'] for row in rows}), len(rows))
        public = [row for row in rows if row['batch'] == 'batch_04_public']
        self.assertEqual(len(public), 80)
        self.assertEqual({row['dataset_id'] for row in public},
                         {'coffee', 'olive_oils', 'turmeric_authenticity', 'beef'})
        for row in rows:
            directory = PACKAGE / row['directory']
            meta = json.loads((directory / 'metadata.json').read_text(encoding='utf-8-sig'))
            self.assertEqual(row['experiment_id'], meta['experiment_id'])
            self.assertEqual(row['batch'], Path(row['directory']).parts[0])
            for name in ('prompt.txt', 'train.json', 'val.json', 'test.json', 'val_labels.json', 'test_labels.json'):
                self.assertTrue((directory / name).is_file(), directory / name)

    def test_api_request_does_not_read_labels_or_mapping(self):
        with (PACKAGE / 'index_current.csv').open(encoding='utf-8', newline='') as f:
            row = next(csv.DictReader(f))
        original = api.read_text
        accessed = []
        def record(path):
            accessed.append(path.name)
            return original(path)
        with patch.object(api, 'read_text', side_effect=record):
            messages, inputs = api.package_messages(PACKAGE / row['directory'])
        self.assertEqual(set(accessed), {'prompt.txt', 'train.json', 'val.json', 'test.json'})
        self.assertEqual(set(inputs), {'train', 'val', 'test'})
        self.assertEqual(len(messages), 2)


if __name__ == '__main__':
    unittest.main()
