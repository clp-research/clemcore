import unittest

from backends import ModelSpec


class ModelSpecTestCase(unittest.TestCase):

    def test_dict_like_get(self):
        spec = ModelSpec(model_name="model_a")
        self.assertEqual(spec["model_name"], "model_a")

    def test_dict_like_contains(self):
        spec = ModelSpec(model_name="model_a")
        self.assertTrue("model_name" in spec)

    def test_dict_like_set_throws_error(self):
        spec = ModelSpec(model_name="model_a")
        with self.assertRaises(TypeError):
            spec["model_name"] = "model_b"

    def test_empty_query_unifies_with_empty_to_empty(self):
        query = ModelSpec()
        entry = ModelSpec()
        self.assertEqual(query.unify(entry), ModelSpec())

    def test_empty_query_unifies_with_entry_to_entry(self):
        query = ModelSpec()  # matches everything
        entry = ModelSpec(model_name="model_b")
        self.assertEqual(query.unify(entry), entry)

    def test_different_query_unifies_with_entry_fails(self):
        query = ModelSpec(model_name="model_a")
        entry = ModelSpec(model_name="model_b")
        with self.assertRaises(ValueError):
            query.unify(entry)

    def test_partial_query_unifies_with_entry_fails(self):
        query = ModelSpec(model_name="model_a", backend="backend_a")
        entry = ModelSpec(model_name="model_a", backend="backend_b")
        with self.assertRaises(ValueError):
            query.unify(entry)

    def test_query_unifies_with_entry_to_entry(self):
        query = ModelSpec(model_name="model_a")
        entry = ModelSpec(model_name="model_a", backend="backend_b")
        self.assertEqual(query.unify(entry), entry)

    def test_query_a_unifies_with_entry_to_union(self):
        query = ModelSpec(model_name="model_a", quantization="8bit")
        entry = ModelSpec(model_name="model_a", backend="backend_b")
        self.assertEqual(query.unify(entry), ModelSpec(model_name="model_a", backend="backend_b", quantization="8bit"))


if __name__ == '__main__':
    unittest.main()
