import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import os
import json
from run_file_manager import Action, MarkdownCodeExtractor, CodeBlock

# base_dir should be actual file directory
file_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script's directory
os.chdir(file_dir)


class TestMarkdownCodeExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = MarkdownCodeExtractor()

    def test_extract_single_code_block(self):
        markdown = '''
File Path: `src/test.py`
```python
def hello():
    print("Hello")
```
'''
        blocks = self.extractor.extract_code_blocks(markdown)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["language"], "python")
        self.assertEqual(blocks[0]["file_path"], "src/test.py")
        self.assertEqual(blocks[0]["code"], 'def hello():\n    print("Hello")')

    def test_extract_multiple_code_blocks(self):
        markdown = '''
File Path: `src/test.py`
```python
def hello():
    print("Hello")
```
File Path: `src/test.js`
```javascript
console.log("Hello");
```
'''
        blocks = self.extractor.extract_code_blocks(markdown)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[1]["language"], "javascript")
        self.assertEqual(blocks[1]["file_path"], "src/test.js")

    def test_extract_no_file_path(self):
        markdown = '''
```python
print("Hello")
```
'''
        blocks = self.extractor.extract_code_blocks(markdown)
        self.assertEqual(len(blocks), 1)
        self.assertIsNone(blocks[0]["file_path"])


class TestAction(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.action = Action()
        self.test_dir = "test_generated"
        # Store original env var if it exists
        self.original_generated_dir = os.environ.get('GENERATED_DIR')
        # Set GENERATED_DIR for tests
        os.environ['GENERATED_DIR'] = self.test_dir

    def tearDown(self):
        # Clean up test directory after tests
        if os.path.exists(self.test_dir):
            for root, dirs, files in os.walk(self.test_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.test_dir)

        # Restore original env var
        if self.original_generated_dir:
            os.environ['GENERATED_DIR'] = self.original_generated_dir
        elif 'GENERATED_DIR' in os.environ:
            del os.environ['GENERATED_DIR']

    def test_save_file(self):
        code = "print('hello')"
        path = "test.py"
        result = self.action.save_file(code, path, "python", self.test_dir)

        self.assertTrue(result["success"])
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test.py")))

        with open(os.path.join(self.test_dir, "test.py"), 'r') as f:
            content = f.read()
        self.assertEqual(content, code)

    def test_save_file_with_nested_path(self):
        code = "console.log('hello')"
        path = "nested/dir/test.js"
        result = self.action.save_file(code, path, "javascript", self.test_dir)

        self.assertTrue(result["success"])
        self.assertTrue(os.path.exists(os.path.join(
            self.test_dir, "nested/dir/test.js")))

    async def test_action_with_valid_code_blocks(self):
        body = {
            "messages": [
                {"content": "Please create these files"},  # User message
                {"content": '''
File Path: `src/test.py`
```python
def test():
    pass
```
'''}  # Assistant message
            ]
        }

        mock_event_emitter = AsyncMock()
        mock_user = {"valves": self.action.UserValves()}

        result = await self.action.action(
            body,
            __user__=mock_user,
            __event_emitter__=mock_event_emitter,
            __event_call__=None
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "code_execution_result")
        self.assertTrue(result["data"]["done"])
        self.assertTrue(any(r["success"] for r in result["data"]["output"]))

    async def test_action_with_no_code_blocks(self):
        body = {
            "messages": [
                {"content": "Just a message"},  # User message
                {"content": "No code blocks here"}  # Assistant message
            ]
        }

        mock_event_emitter = AsyncMock()
        mock_user = {"valves": self.action.UserValves()}

        result = await self.action.action(
            body,
            __user__=mock_user,
            __event_emitter__=mock_event_emitter,
            __event_call__=None
        )

        self.assertIsNone(result)
        mock_event_emitter.assert_called_with({
            "type": "status",
            "data": {"description": "No valid code block detected", "done": True}
        })

    async def test_action_with_custom_generated_dir(self):
        # Set custom GENERATED_DIR
        custom_dir = os.path.join(self.test_dir, "custom_generated")
        os.environ['GENERATED_DIR'] = custom_dir

        body = {
            "messages": [
                {"content": "Create a test file"},
                {"content": '''
File Path: `test.py`
```python
print("test")
```
'''}
            ]
        }

        mock_event_emitter = AsyncMock()
        mock_user = {"valves": self.action.UserValves()}

        result = await self.action.action(
            body,
            __user__=mock_user,
            __event_emitter__=mock_event_emitter,
            __event_call__=None
        )

        # Verify the file was created in the custom directory
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "code_execution_result")

        # Check that the base directory starts with our custom directory
        output_files = result["data"]["output"]
        for file_result in output_files:
            if file_result["file"].name != "meta.json":  # Skip meta file
                self.assertTrue(
                    file_result["file"].name.startswith(custom_dir))


if __name__ == "__main__":
    unittest.main()
