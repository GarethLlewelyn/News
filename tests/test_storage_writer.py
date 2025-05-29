import unittest
import os
import json
import shutil # For rmtree
import datetime
import sys

# Add project root to sys.path to allow importing project modules
# This assumes tests are run from the project root or that PYTHONPATH is set up
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from storage.writer import save_processed_article, save_bulk_processed_articles, PROCESSED_DATA_DIR

class TestStorageWriter(unittest.TestCase):
    
    test_temp_dir = "temp_test_processed_data"
    original_processed_data_dir = None

    @classmethod
    def setUpClass(cls):
        """Set up a temporary directory for test outputs."""
        # Store original PROCESSED_DATA_DIR and replace it for tests
        cls.original_processed_data_dir = PROCESSED_DATA_DIR
        # Update the PROCESSED_DATA_DIR in the storage.writer module directly for the test scope
        # This is a common way to handle module-level configurations in tests.
        # Requires storage.writer to be imported as 'from storage import writer' and then 'writer.PROCESSED_DATA_DIR = ...'
        # Or, if PROCESSED_DATA_DIR is used directly from the import, we might need a different approach
        # For now, let's assume we can influence it or the functions allow specifying a dir (which they don't currently)
        # A simple workaround: create our test dir inside the existing PROCESSED_DATA_DIR if it must be fixed.
        # Better: Modify writer functions to accept output_dir or use a global test config.

        # For this test, we'll create a subdir within the default PROCESSED_DATA_DIR
        # to avoid modifying the module's global state too invasively if not designed for it.
        # However, the functions in storage.writer use PROCESSED_DATA_DIR globally.
        # Let's try to monkeypatch it for the test duration.
        cls.patch_target_module = __import__('storage.writer', fromlist=['PROCESSED_DATA_DIR'])
        setattr(cls.patch_target_module, 'PROCESSED_DATA_DIR', cls.test_temp_dir)

        if os.path.exists(cls.test_temp_dir):
            shutil.rmtree(cls.test_temp_dir)
        os.makedirs(cls.test_temp_dir)
        print(f"Created temp test directory: {os.path.abspath(cls.test_temp_dir)}")

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary directory after all tests."""
        if os.path.exists(cls.test_temp_dir):
            shutil.rmtree(cls.test_temp_dir)
            print(f"Removed temp test directory: {cls.test_temp_dir}")
        # Restore original PROCESSED_DATA_DIR
        if cls.original_processed_data_dir and cls.patch_target_module:
             setattr(cls.patch_target_module, 'PROCESSED_DATA_DIR', cls.original_processed_data_dir)

    def tearDown(self):
        """Clean up any files created within the temp_test_dir by a test."""
        for item in os.listdir(self.test_temp_dir):
            item_path = os.path.join(self.test_temp_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path) # Should not have subdirs per current writer logic

    def _get_expected_filename(self, prefix="test_features"):
        date_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        return os.path.join(self.test_temp_dir, f"{prefix}_{date_str}.jsonl")

    def test_save_single_article(self):
        print("Running test_save_single_article...")
        article = {"id": 1, "title": "Test Article 1", "content": "Hello"}
        filename_prefix = "single_save_test"
        expected_file = self._get_expected_filename(filename_prefix)

        self.assertTrue(save_processed_article(article, filename_prefix=filename_prefix))
        self.assertTrue(os.path.exists(expected_file))

        with open(expected_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.assertEqual(data["id"], 1)
            self.assertEqual(data["title"], "Test Article 1")

    def test_save_single_article_invalid_data(self):
        print("Running test_save_single_article_invalid_data...")
        article = "not a dict"
        self.assertFalse(save_processed_article(article, filename_prefix="invalid_test")) # type: ignore

    def test_save_bulk_articles(self):
        print("Running test_save_bulk_articles...")
        articles = [
            {"id": 10, "title": "Bulk Article 1"},
            {"id": 11, "title": "Bulk Article 2"}
        ]
        filename_prefix = "bulk_save_test"
        expected_file = self._get_expected_filename(filename_prefix)

        saved_count, failed_count = save_bulk_processed_articles(articles, filename_prefix=filename_prefix)
        self.assertEqual(saved_count, 2)
        self.assertEqual(failed_count, 0)
        self.assertTrue(os.path.exists(expected_file))

        with open(expected_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)
            data1 = json.loads(lines[0])
            data2 = json.loads(lines[1])
            self.assertEqual(data1["id"], 10)
            self.assertEqual(data2["id"], 11)

    def test_save_bulk_articles_with_some_invalid(self):
        print("Running test_save_bulk_articles_with_some_invalid...")
        articles = [
            {"id": 20, "title": "Valid Bulk 1"},
            "not a dict",
            {"id": 21, "title": "Valid Bulk 2"}
        ]
        filename_prefix = "bulk_invalid_test"
        expected_file = self._get_expected_filename(filename_prefix)

        saved_count, failed_count = save_bulk_processed_articles(articles, filename_prefix=filename_prefix) # type: ignore
        self.assertEqual(saved_count, 2)
        self.assertEqual(failed_count, 1)
        self.assertTrue(os.path.exists(expected_file))

        with open(expected_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)
            # Check if the valid articles are present
            titles_in_file = [json.loads(line)["title"] for line in lines]
            self.assertIn("Valid Bulk 1", titles_in_file)
            self.assertIn("Valid Bulk 2", titles_in_file)

    def test_save_bulk_no_articles(self):
        print("Running test_save_bulk_no_articles...")
        articles = []
        filename_prefix = "bulk_empty_test"
        saved_count, failed_count = save_bulk_processed_articles(articles, filename_prefix=filename_prefix)
        self.assertEqual(saved_count, 0)
        self.assertEqual(failed_count, 0)
        # File might or might not be created depending on handling of empty list, check writer logic.
        # Current writer logic: if not articles_list, prints "No articles to save" and returns 0,0.
        # It doesn't create an empty file, which is fine.
        # expected_file = self._get_expected_filename(filename_prefix)
        # self.assertFalse(os.path.exists(expected_file)) # or True if it creates an empty file

if __name__ == '__main__':
    # This allows running the tests directly from this file
    # For more complex projects, use a test runner like `python -m unittest discover tests`
    unittest.main() 