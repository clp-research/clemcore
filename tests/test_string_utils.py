import unittest

from clemcore.utils.string_utils import (
    to_pretty_json,
    remove_punctuation,
    str_to_bool,
    try_convert,
    read_query_string,
)


class ToPrettyJsonTestCase(unittest.TestCase):

    def test_simple_dict(self):
        """Test formatting simple dictionary."""
        data = {"key": "value", "number": 42}
        result = to_pretty_json(data)
        self.assertIn('"key":', result)
        self.assertIn('"value"', result)
        self.assertIn("42", result)

    def test_indentation(self):
        """Test that output is indented."""
        data = {"a": {"b": "c"}}
        result = to_pretty_json(data)
        self.assertIn("\n", result)  # Should have newlines
        self.assertIn("  ", result)  # Should have indentation

    def test_newlines_in_strings(self):
        """Test that escaped newlines become actual newlines."""
        data = {"text": "line1\\nline2"}
        result = to_pretty_json(data)
        # The \\n should be converted to actual newline
        self.assertIn("\n", result)

    def test_non_serializable_objects(self):
        """Test that non-serializable objects use str()."""
        class CustomObject:
            def __str__(self):
                return "custom_repr"

        data = {"obj": CustomObject()}
        result = to_pretty_json(data)
        self.assertIn("custom_repr", result)


class RemovePunctuationTestCase(unittest.TestCase):

    def test_removes_common_punctuation(self):
        """Test removal of common punctuation."""
        text = "Hello, world! How are you?"
        result = remove_punctuation(text)
        self.assertEqual(result, "Hello world How are you")

    def test_removes_all_punctuation_types(self):
        """Test removal of various punctuation types."""
        text = "Test: semi-colon; quotes 'single' \"double\""
        result = remove_punctuation(text)
        self.assertNotIn(":", result)
        self.assertNotIn(";", result)
        self.assertNotIn("'", result)
        self.assertNotIn('"', result)

    def test_preserves_alphanumeric(self):
        """Test that alphanumeric characters are preserved."""
        text = "ABC123 xyz789"
        result = remove_punctuation(text)
        self.assertEqual(result, "ABC123 xyz789")

    def test_empty_string(self):
        """Test with empty string."""
        result = remove_punctuation("")
        self.assertEqual(result, "")


class StrToBoolTestCase(unittest.TestCase):

    def test_true_values(self):
        """Test various true values."""
        for val in ["true", "True", "TRUE", "yes", "YES", "on", "ON", "1"]:
            self.assertTrue(str_to_bool(val), f"Failed for {val}")

    def test_false_values(self):
        """Test various false values."""
        for val in ["false", "False", "FALSE", "no", "NO", "off", "OFF", "0"]:
            self.assertFalse(str_to_bool(val), f"Failed for {val}")

    def test_invalid_value_raises(self):
        """Test that invalid values raise ValueError."""
        with self.assertRaises(ValueError):
            str_to_bool("maybe")
        with self.assertRaises(ValueError):
            str_to_bool("2")
        with self.assertRaises(ValueError):
            str_to_bool("")


class TryConvertTestCase(unittest.TestCase):

    def test_converts_to_int(self):
        """Test conversion to int."""
        result = try_convert("42", (int, float, str))
        self.assertEqual(result, 42)
        self.assertIsInstance(result, int)

    def test_converts_to_float(self):
        """Test conversion to float."""
        result = try_convert("3.14", (int, float, str))
        self.assertEqual(result, 3.14)
        self.assertIsInstance(result, float)

    def test_converts_to_bool(self):
        """Test conversion to bool."""
        result = try_convert("true", (str_to_bool, int, str))
        self.assertTrue(result)

    def test_fallback_to_string(self):
        """Test that unconvertible values remain strings."""
        result = try_convert("hello", (int, float))
        self.assertEqual(result, "hello")
        self.assertIsInstance(result, str)

    def test_first_matching_type_wins(self):
        """Test that first successful conversion is used."""
        # "42" can be both int and float, but int comes first
        result = try_convert("42", (int, float))
        self.assertIsInstance(result, int)


class ReadQueryStringTestCase(unittest.TestCase):

    def test_none_input(self):
        """Test that None returns None."""
        result = read_query_string(None)
        self.assertIsNone(result)

    def test_empty_string(self):
        """Test that empty string returns empty dict."""
        result = read_query_string("")
        self.assertEqual(result, {})

    def test_single_pair(self):
        """Test parsing single key-value pair."""
        result = read_query_string("key=value")
        self.assertEqual(result, {"key": "value"})

    def test_multiple_pairs(self):
        """Test parsing multiple key-value pairs."""
        result = read_query_string("a=1,b=2,c=3")
        self.assertEqual(result, {"a": 1, "b": 2, "c": 3})

    def test_type_conversion_int(self):
        """Test that integers are converted."""
        result = read_query_string("count=42")
        self.assertEqual(result["count"], 42)
        self.assertIsInstance(result["count"], int)

    def test_type_conversion_float(self):
        """Test that floats are converted."""
        result = read_query_string("rate=0.5")
        self.assertEqual(result["rate"], 0.5)
        self.assertIsInstance(result["rate"], float)

    def test_type_conversion_bool(self):
        """Test that booleans are converted."""
        result = read_query_string("enabled=true,disabled=false")
        self.assertTrue(result["enabled"])
        self.assertFalse(result["disabled"])

    def test_whitespace_trimming(self):
        """Test that whitespace is trimmed."""
        result = read_query_string("  key  =  value  ")
        self.assertEqual(result, {"key": "value"})

    def test_invalid_pair_raises(self):
        """Test that invalid pairs raise ValueError."""
        with self.assertRaises(ValueError):
            read_query_string("invalid_no_equals")

    def test_value_with_equals(self):
        """Test that values can contain equals signs."""
        result = read_query_string("url=http://example.com?a=b")
        self.assertEqual(result["url"], "http://example.com?a=b")


if __name__ == '__main__':
    unittest.main()
