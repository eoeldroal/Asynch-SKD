import types
import unittest
from unittest import mock

import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.server_args import PortArgs, ServerArgs


class TestSglangNcclPortRetry(unittest.TestCase):
    def test_port_args_preserves_explicit_nccl_port_for_dp_workers(self):
        server_args = ServerArgs(model_path="dummy", dp_size=4, nccl_port=46006)

        port_args = PortArgs.init_new(server_args, dp_rank=2)

        self.assertEqual(port_args.nccl_port, 46006)

    def test_init_distributed_environment_surfaces_single_rank_tcp_eaddrinuse(self):
        init_methods = []

        dist_error = torch.distributed.DistNetworkError(
            "The server socket has failed to listen on any local network address. "
            "port: 45009, useIpv6: false, code: -98, name: EADDRINUSE, "
            "message: address already in use"
        )

        def fake_init_process_group(**kwargs):
            init_methods.append(kwargs["init_method"])
            if len(init_methods) == 1:
                raise dist_error

        with mock.patch.object(torch.distributed, "is_initialized", return_value=False):
            with mock.patch.object(parallel_state, "_WORLD", None):
                with mock.patch.object(
                    parallel_state,
                    "init_world_group",
                    return_value=types.SimpleNamespace(world_size=1),
                ):
                    with mock.patch.object(torch.distributed, "get_world_size", return_value=1):
                        with mock.patch.object(
                            torch.distributed,
                            "init_process_group",
                            side_effect=fake_init_process_group,
                        ):
                            with self.assertRaises(torch.distributed.DistNetworkError):
                                parallel_state.init_distributed_environment(
                                    backend="gloo",
                                    world_size=1,
                                    rank=0,
                                    local_rank=0,
                                    distributed_init_method="tcp://127.0.0.1:45009",
                                )

        self.assertEqual(
            init_methods,
            [
                "tcp://127.0.0.1:45009",
            ],
        )


if __name__ == "__main__":
    unittest.main()
