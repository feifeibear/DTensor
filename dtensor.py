import os
import torch
from torch.distributed._tensor import DTensor, DeviceMesh, Shard, Replicate, distribute_module, distribute_tensor
import torch.distributed as dist

def test_dtensor():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()

    # Create a mesh topology with the available devices.
    mesh = DeviceMesh("cuda", list(range(world_size)))
    big_tensor = torch.randn(100000, 88)
    # Shard this tensor over the mesh by sharding `big_tensor`'s 0th dimension over the 0th dimension of `mesh`.
    my_dtensor = distribute_tensor(big_tensor, mesh, [Shard(dim=0)])
    print(my_dtensor)

def test_4_gpu():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()

    # construct a device mesh with available devices (multi-host or single host)
    device_mesh = DeviceMesh("cuda", [0, 1, 2, 3])
    # if we want to do row-wise sharding
    rowwise_placement=[Shard(0)]
    # if we want to do col-wise sharding
    colwise_placement=[Shard(1)]
    # distributed tensor returned will be sharded across the dimension specified in placements
    dist_tensor = DTensor(
            torch.empty(8 // world_size, 12),
            device_mesh,
            rowwise_placement,
            size=(8, 12)
        )
    print('dist_tensor ', dist_tensor)
    # if we want to do replication across a certain device list
    replica_placement = [Replicate()]
    dist_tensor = DTensor(
            torch.empty(8 , 12),
            device_mesh,
            replica_placement,
            size=(8, 12)
        )

    # if we want to distributed a tensor with both replication and sharding
    device_mesh = DeviceMesh("cuda", [[0, 1], [2, 3]])
    # replicate across the first dimension of device mesh, then sharding on the second dimension of device mesh
    spec=[Replicate(), Shard(0)]
    dist_tensor = DTensor(
            torch.empty(8 , 12),
            device_mesh,
            replica_placement,
            size=(8, 12)
        )
    # create a DistributedTensor that shards on dim 0, from a local torch.Tensor
    local_tensor = torch.randn((8, 8), requires_grad=True)
    rowwise_tensor = DTensor.from_local(local_tensor, device_mesh, rowwise_placement)

    # reshard the current rowise tensor to a colwise tensor or replicate tensor
    colwise_tensor = rowwise_tensor.redistribute(device_mesh, colwise_placement)
    replica_tensor = colwise_tensor.redistribute(device_mesh, replica_placement)


if __name__ == "__main__":
    # torchrun --standalone --nproc_per_node=4 dtensor.py
    test_4_gpu()