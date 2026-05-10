import numpy as np

from verl.model_merger.fsdp_model_merger import FSDPModelMerger


def test_calculate_shard_configuration_supports_fsdp_mesh_alias():
    merger = object.__new__(FSDPModelMerger)

    total_shards, mesh_shape = merger._calculate_shard_configuration(
        np.array([0, 1, 2, 3], dtype=np.int64),
        ("dp_shard_sp",),
    )

    assert total_shards == 4
    assert mesh_shape == (4,)
