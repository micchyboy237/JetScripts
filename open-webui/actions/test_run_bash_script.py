import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from run_bash_script import Action


class TestAction(unittest.IsolatedAsyncioTestCase):  # Updated to use async test case
    def setUp(self):
        self.action = Action()

    def test_execute_bash_code_success(self):
        """Test successful execution of a bash command."""
        result = self.action.execute_bash_code("echo 'Hello, World!'")
        self.assertEqual(result.strip(), "Hello, World!")

    def test_execute_bash_code_error(self):
        """Test bash command with an error."""
        result = self.action.execute_bash_code("invalid_command")
        self.assertTrue(result.startswith("Error:"))

    def test_execute_bash_code_empty(self):
        """Test execution with an empty bash command."""
        result = self.action.execute_bash_code("")
        self.assertEqual(result.strip(), "")

    def test_execute_bash_code_with_exception(self):
        """Test handling of exceptions in bash code execution."""
        with patch("subprocess.Popen", side_effect=Exception("Test Error")):
            result = self.action.execute_bash_code("echo 'test'")
            self.assertEqual(result, "test\n")

    async def test_action_valid_bash_code(self):
        """Test action method with valid bash code input."""
        body = {"messages": [
            {"content": "```bash\necho 'Hello, World!'\n```"}]}
        mock_event_emitter = AsyncMock()
        mock_user = MagicMock()

        result = await self.action.action(
            body, __user__=mock_user, __event_emitter__=mock_event_emitter
        )

        self.assertEqual(result["type"], "code_execution_result")
        self.assertEqual(result["data"]["output"].strip(), "Hello, World!")

    async def test_action_invalid_bash_code(self):
        """Test action method with invalid bash code input."""
        body = {"messages": [{"content": "```bash\ninvalid_command\n```"}]}
        mock_event_emitter = AsyncMock()
        mock_user = MagicMock()

        result = await self.action.action(
            body, __user__=mock_user, __event_emitter__=mock_event_emitter
        )

        self.assertEqual(result["type"], "code_execution_result")
        self.assertTrue(result["data"]["output"].startswith("Error:"))

    async def test_action_no_bash_code(self):
        """Test action method with no bash code input."""
        body = {"messages": [{"content": "Just a plain message"}]}
        mock_event_emitter = AsyncMock()
        mock_user = MagicMock()

        result = await self.action.action(
            body, __user__=mock_user, __event_emitter__=mock_event_emitter
        )

        self.assertIsNone(result)

    async def test_action_with_status_update(self):
        """Test action method emits status updates."""
        body = {"messages": [
            # Updated to invalid command
            {"content": "```bash\ninvalid_command\n```"}]}
        mock_event_emitter = AsyncMock()
        mock_user = MagicMock()
        result = await self.action.action(
            body, __user__=mock_user, __event_emitter__=mock_event_emitter
        )
        mock_event_emitter.assert_any_await(
            {"type": "status", "data": {
                "description": "Processing your input", "done": False}}
        )
        mock_event_emitter.assert_any_await(
            {"type": "status", "data": {
                "description": "No valid Bash code detected", "done": True}}
        )


if __name__ == "__main__":
    unittest.main()
