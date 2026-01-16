import unittest

from clemcore.backends import ModelSpec


class ModelSpecDictBehaviorTestCase(unittest.TestCase):
    """Tests for dict-like behavior of ModelSpec."""

    def test_getitem(self):
        spec = ModelSpec(model_name="model_a")
        self.assertEqual(spec["model_name"], "model_a")

    def test_contains(self):
        spec = ModelSpec(model_name="model_a")
        self.assertIn("model_name", spec)
        self.assertNotIn("nonexistent", spec)

    def test_setitem_raises(self):
        spec = ModelSpec(model_name="model_a")
        with self.assertRaises(TypeError):
            spec["model_name"] = "model_b"


class ModelSpecFactoryTestCase(unittest.TestCase):
    """Tests for ModelSpec factory methods."""

    def test_from_name(self):
        spec = ModelSpec.from_name("gpt-4")
        self.assertEqual(spec.model_name, "gpt-4")

    def test_from_name_none_raises(self):
        with self.assertRaises(ValueError):
            ModelSpec.from_name(None)

    def test_from_dict(self):
        spec = ModelSpec.from_dict({"model_name": "gpt-4", "backend": "openai"})
        self.assertEqual(spec.model_name, "gpt-4")
        self.assertEqual(spec.backend, "openai")

    def test_from_string_json(self):
        spec = ModelSpec.from_string('{"model_name": "gpt-4", "backend": "openai"}')
        self.assertEqual(spec.model_name, "gpt-4")
        self.assertEqual(spec.backend, "openai")

    def test_from_string_simple_name(self):
        spec = ModelSpec.from_string("gpt-4")
        self.assertEqual(spec.model_name, "gpt-4")

    def test_from_string_single_quotes(self):
        spec = ModelSpec.from_string("{'model_name': 'gpt-4'}")
        self.assertEqual(spec.model_name, "gpt-4")

    def test_from_strings(self):
        specs = ModelSpec.from_strings(["gpt-4", '{"model_name": "claude"}'])
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].model_name, "gpt-4")
        self.assertEqual(specs[1].model_name, "claude")


class ModelSpecSerializationTestCase(unittest.TestCase):
    """Tests for ModelSpec serialization methods."""

    def test_to_dict(self):
        spec = ModelSpec(model_name="gpt-4", backend="openai")
        d = spec.to_dict()
        self.assertEqual(d, {"model_name": "gpt-4", "backend": "openai"})

    def test_to_string(self):
        spec = ModelSpec(model_name="gpt-4")
        s = spec.to_string()
        self.assertIn("gpt-4", s)
        self.assertIn("model_name", s)

    def test_repr(self):
        spec = ModelSpec(model_name="gpt-4")
        self.assertIn("ModelSpec", repr(spec))
        self.assertIn("gpt-4", repr(spec))

    def test_str(self):
        spec = ModelSpec(model_name="gpt-4")
        self.assertIn("gpt-4", str(spec))


class ModelSpecAttributeTestCase(unittest.TestCase):
    """Tests for ModelSpec attribute checking methods."""

    def test_has_attr(self):
        spec = ModelSpec(model_name="gpt-4", temperature=0.7)
        self.assertTrue(spec.has_attr("model_name"))
        self.assertTrue(spec.has_attr("temperature"))
        self.assertFalse(spec.has_attr("nonexistent"))

    def test_has_temperature(self):
        spec_with = ModelSpec(model_name="gpt-4", temperature=0.7)
        spec_without = ModelSpec(model_name="gpt-4")
        self.assertTrue(spec_with.has_temperature())
        self.assertFalse(spec_without.has_temperature())

    def test_has_backend(self):
        spec_with = ModelSpec(model_name="gpt-4", backend="openai")
        spec_without = ModelSpec(model_name="gpt-4")
        self.assertTrue(spec_with.has_backend())
        self.assertFalse(spec_without.has_backend())


class ModelSpecTypeTestCase(unittest.TestCase):
    """Tests for ModelSpec type checking methods."""

    def test_is_programmatic(self):
        self.assertTrue(ModelSpec(model_name="mock").is_programmatic())
        self.assertTrue(ModelSpec(model_name="programmatic").is_programmatic())
        self.assertFalse(ModelSpec(model_name="gpt-4").is_programmatic())

    def test_is_human(self):
        self.assertTrue(ModelSpec(model_name="human").is_human())
        self.assertTrue(ModelSpec(model_name="terminal").is_human())
        self.assertFalse(ModelSpec(model_name="gpt-4").is_human())


class ModelSpecRenameTestCase(unittest.TestCase):
    """Tests for ModelSpec rename method."""

    def test_rename_creates_new_spec(self):
        original = ModelSpec(model_name="gpt-4", backend="openai")
        renamed = original.rename("gpt-4-turbo")
        self.assertEqual(renamed.model_name, "gpt-4-turbo")
        self.assertEqual(renamed.backend, "openai")
        # Original unchanged
        self.assertEqual(original.model_name, "gpt-4")

    def test_rename_preserves_attributes(self):
        original = ModelSpec(model_name="model", backend="openai", temperature=0.5)
        renamed = original.rename("new_model")
        self.assertEqual(renamed.temperature, 0.5)
        self.assertEqual(renamed.backend, "openai")


class ModelSpecUnificationTestCase(unittest.TestCase):
    """Tests for ModelSpec unification."""

    def test_empty_unifies_with_empty(self):
        self.assertEqual(ModelSpec().unify(ModelSpec()), ModelSpec())

    def test_empty_unifies_with_entry(self):
        entry = ModelSpec(model_name="model_b")
        self.assertEqual(ModelSpec().unify(entry), entry)

    def test_conflicting_values_raises(self):
        query = ModelSpec(model_name="model_a")
        entry = ModelSpec(model_name="model_b")
        with self.assertRaises(ValueError):
            query.unify(entry)

    def test_partial_conflict_raises(self):
        query = ModelSpec(model_name="model_a", backend="backend_a")
        entry = ModelSpec(model_name="model_a", backend="backend_b")
        with self.assertRaises(ValueError):
            query.unify(entry)

    def test_subset_unifies(self):
        query = ModelSpec(model_name="model_a")
        entry = ModelSpec(model_name="model_a", backend="backend_b")
        self.assertEqual(query.unify(entry), entry)

    def test_union_of_attributes(self):
        query = ModelSpec(model_name="model_a", quantization="8bit")
        entry = ModelSpec(model_name="model_a", backend="backend_b")
        unified = query.unify(entry)
        self.assertEqual(unified.model_name, "model_a")
        self.assertEqual(unified.backend, "backend_b")
        self.assertEqual(unified.quantization, "8bit")


if __name__ == '__main__':
    unittest.main()
