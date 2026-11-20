"""Test RPC selection without numerical libraries or a running Redis server."""
import ast
from collections import namedtuple
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


class RpcInvokerTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "squeak.py").read_text())
        selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                    and node.name in {"MergePromise", "get_rpc_invoker"}]
        namespace = {"namedtuple": namedtuple}
        exec(compile(ast.Module(body=selected, type_ignores=[]), "squeak.py", "exec"), namespace)
        self.invoker = namespace["get_rpc_invoker"]

    def test_local_sync_executes_without_redis(self):
        result = self.invoker(async_eval=False)(lambda a, b: a + b, 2, 3)
        self.assertTrue(result.is_finished)
        self.assertEqual(result.result, 5)

    def test_redis_receives_supported_async_option(self):
        redis_module, rq_module = ModuleType("redis"), ModuleType("rq")
        redis_module.Redis, rq_module.Queue = Mock(), Mock()
        for async_eval in (True, False):
            with self.subTest(async_eval=async_eval):
                with patch.dict(sys.modules, redis=redis_module, rq=rq_module):
                    invoke = self.invoker(url="remote.example", async_eval=async_eval)
                rq_module.Queue.assert_called_with(
                    connection=redis_module.Redis.return_value, is_async=async_eval)
                function = lambda value: value
                invoke(function, 7)
                rq_module.Queue.return_value.enqueue.assert_called_with(function, 7, timeout=86400)


if __name__ == "__main__":
    unittest.main()
