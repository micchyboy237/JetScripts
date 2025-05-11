import unittest
import asyncio
from _temp_test import wrap_await_code_multiline_args, format_json, CodeBlock, CancellationToken, DockerCommandLineCodeExecutor


class TestWrapAwaitCode(unittest.TestCase):
    def test_async_with_statement(self):
        code = """
async with DockerCommandLineCodeExecutor(work_dir='.') as executor:
    result = await executor.execute_code_blocks(
        code_blocks=[CodeBlock(language="python", code="print('Hello')")],
        cancellation_token=CancellationToken()
    )
    logger.debug(result)
"""
        expected = """
async def async_func_1():
    async with DockerCommandLineCodeExecutor(work_dir='.') as executor:
        result = await executor.execute_code_blocks(
            code_blocks=[CodeBlock(language="python", code="print('Hello')")],
            cancellation_token=CancellationToken()
        )
        logger.debug(result)
    return result

result = asyncio.run(async_func_1())
logger.success(format_json(result))
""".strip()
        result = wrap_await_code_multiline_args(code).strip()
        self.assertEqual(result, expected)

    def test_multiline_await(self):
        code = """
result = await executor.execute_code_blocks(
    code_blocks=[CodeBlock(language="python", code="print('Hello')")],
    cancellation_token=CancellationToken()
)
"""
        expected = """
async def async_func_1():
    result = await executor.execute_code_blocks(
        code_blocks=[CodeBlock(language="python", code="print('Hello')")],
        cancellation_token=CancellationToken()
    )
    return result
result = asyncio.run(async_func_1())
logger.success(format_json(result))
""".strip()
        result = wrap_await_code_multiline_args(code).strip()
        self.assertEqual(result, expected)

    def test_single_line_await(self):
        code = """
result = await some_function()
"""
        result = wrap_await_code_multiline_args(code).strip()
        self.assertIn("async def async_func_", result)
        self.assertIn("result = await some_function()", result)
        self.assertIn("result = asyncio.run(async_func_", result)
        self.assertIn("logger.success(format_json(result))", result)

    def test_mixed_code(self):
        code = """
print('Starting')
async with DockerCommandLineCodeExecutor(work_dir='.') as executor:
    result = await executor.execute_code_blocks(
        code_blocks=[CodeBlock(language="python", code="print('Hello')")],
        cancellation_token=CancellationToken()
    )
print('Done')
"""
        result = wrap_await_code_multiline_args(code).strip()
        lines = result.split('\n')
        self.assertEqual(lines[0], "print('Starting')")
        self.assertIn("async def async_func_", result)
        self.assertIn(
            "async with DockerCommandLineCodeExecutor(work_dir='.') as executor:", result)
        self.assertIn("result = await executor.execute_code_blocks(", result)
        self.assertEqual(lines[-1], "print('Done')")

    def test_no_await(self):
        code = """
print('Hello')
x = 42
"""
        result = wrap_await_code_multiline_args(code).strip()
        self.assertEqual(result, code.strip())

    def test_indented_async_with(self):
        code = """
def main():
    async with DockerCommandLineCodeExecutor(work_dir='.') as executor:
        result = await executor.execute_code_blocks(
            code_blocks=[CodeBlock(language="python", code="print('Hello')")],
            cancellation_token=CancellationToken()
        )
"""
        result = wrap_await_code_multiline_args(code).strip()
        self.assertIn("    async def async_func_", result)
        self.assertIn(
            "        async with DockerCommandLineCodeExecutor(work_dir='.') as executor:", result)
        self.assertIn("    result_2 = asyncio.run(async_func_", result)

    @classmethod
    def setUpClass(cls):
        cls.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(cls.loop)

    @classmethod
    def tearDownClass(cls):
        cls.loop.close()

    def run_async(self, coro):
        return self.loop.run_until_complete(coro)


if __name__ == '__main__':
    unittest.main()
