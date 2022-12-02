from colossalai.device.device_mesh import DeviceMesh
from colossalai.initialize import launch
from colossalai.tensor.shape_consistency import CollectiveCommPattern, ShapeConsistencyManager
from colossalai.tensor.sharding_spec import ShardingSpec

import torch
import torch.distributed as dist
import os


def test_colo_tensor():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    world_size = dist.get_world_size()
    physical_mesh_id = torch.arange(0, 4)
    mesh_shape = torch.Size([4])
    
    tensor_shape = [8, 12]
    device_mesh = DeviceMesh(physical_mesh_id, mesh_shape, init_process_group=True)
    entire_shape = torch.Size(tensor_shape)
    shape_consistency_manager = ShapeConsistencyManager()

    # row shard
    row_sharding = {0: [0]}
    # column shard
    col_sharding = {1: [0]}

    sharding_spec_source = ShardingSpec(device_mesh, entire_shape, row_sharding)
    sharding_spec_target = ShardingSpec(device_mesh, entire_shape, col_sharding)

    # replicated tensor
    torch.manual_seed(0)
    tensor_to_comm = torch.randn(tensor_shape).cuda()

    tensor_to_comm.sharding_spec = sharding_spec_source
    tensor_to_comm = shape_consistency_manager.apply(tensor_to_comm, sharding_spec_target)

    print(tensor_to_comm)


if __name__ == "__main__":
    # torchrun --standalone --nproc_per_node=4 cai.py
    test_colo_tensor()