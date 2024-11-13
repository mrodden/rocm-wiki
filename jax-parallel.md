

# Sample calculation

```python
import jax
from jax import numpy as jnp
from jax.sharding import PartitionSpec as P


arr = jnp.arange(24.0).reshape(3, 8)

#jax.debug.visualize_array_sharding(arr)
mesh = jax.make_mesh((3,), ("x",))
sharding = jax.sharding.NamedSharding(mesh, P("x"))
arr_sharded = jax.device_put(arr, sharding)
print(arr_sharded)

@jax.jit
def contract_2(x):
    out = x.sum(axis=0)
    mesh = jax.make_mesh((3,), ("x",))
    sharding = jax.sharding.NamedSharding(mesh, P("x"))
    return jax.lax.with_sharding_constraint(out, sharding)


result = contract_2(arr_sharded)
print(result)

```


HLO output
```
root@rocm-rome-3:~# cat xla_dump/module_0007.jit_contract_2.gpu_target_config.pbtxt
gpu_device_info {
  threads_per_block_limit: 1024
  threads_per_warp: 64
  shared_memory_per_block: 65536
  shared_memory_per_core: 65536
  threads_per_core_limit: 2560
  core_count: 120
  fpus_per_core: 128
  block_dim_limit_x: 2147483647
  block_dim_limit_y: 65536
  block_dim_limit_z: 65536
  memory_bandwidth: 1228800000000
  l2_cache_size: 8388608
  clock_rate_ghz: 1.502
  device_memory_size: 33806090240
  shared_memory_per_block_optin: -1
  rocm_compute_capability {
    gcn_arch_name: "gfx908:sramecc+:xnack-"
  }
  registers_per_core_limit: 65536
  registers_per_block_limit: 65536
}
platform_name: "ROCM"
dnn_version_info {
  major: 1
  minor: 3
}
device_description_str: "AMD Instinct MI100"
root@rocm-rome-3:~# cat xla_dump/module_0007.jit_contract_2.before_optimizations.txt
HloModule jit_contract_2, entry_computation_layout={(f32[3,8]{1,0})->f32[8]{0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=3

region_0.3 {
  Arg_0.4 = f32[] parameter(0), metadata={op_name="jit(contract_2)/jit(main)/reduce_sum"}
  Arg_1.5 = f32[] parameter(1), metadata={op_name="jit(contract_2)/jit(main)/reduce_sum"}
  ROOT add.6 = f32[] add(Arg_0.4, Arg_1.5), metadata={op_name="jit(contract_2)/jit(main)/reduce_sum" source_file="/root/parallel.py" source_line=18}
}

ENTRY main.9 {
  Arg_0.1 = f32[3,8]{1,0} parameter(0), sharding={devices=[3,1]<=[3]}, metadata={op_name="x"}
  constant.2 = f32[] constant(0)
  reduce.7 = f32[8]{0} reduce(Arg_0.1, constant.2), dimensions={0}, to_apply=region_0.3, metadata={op_name="jit(contract_2)/jit(main)/reduce_sum" source_file="/root/parallel.py" source_line=18}
  ROOT custom-call.8 = f32[8]{0} custom-call(reduce.7), custom_call_target="Sharding", sharding={devices=[3]<=[3]}, metadata={op_name="jit(contract_2)/jit(main)/sharding_constraint" source_file="/root/parallel.py" source_line=21}
}

root@rocm-rome-3:~# cat xla_dump/module_0007.jit_contract_2.gfx908_gpu_after_optimizations.txt
HloModule jit_contract_2, is_scheduled=true, entry_computation_layout={(f32[1,8]{1,0})->f32[8]{0}}, allow_spmd_sharding_propagation_to_output={true}, num_partitions=3, frontend_attributes={fingerprint_before_lhs="1ecf5621c56c9f3861571058e268ab5d"}

region_0.3.clone {
  Arg_1.0 = f32[] parameter(1), metadata={op_name="jit(contract_2)/jit(main)/reduce_sum"}
  Arg_0.0 = f32[] parameter(0), metadata={op_name="jit(contract_2)/jit(main)/reduce_sum"}
  ROOT add.2 = f32[] add(Arg_0.0, Arg_1.0), metadata={op_name="jit(contract_2)/jit(main)/reduce_sum" source_file="/root/parallel.py" source_line=18}
}

wrapped_copy_computation {
  param_0 = f32[1,8]{1,0} parameter(0)
  ROOT copy.4 = f32[1,8]{1,0} copy(param_0)
}

ENTRY main.9_spmd {
  param.1 = f32[1,8]{1,0} parameter(0), sharding={devices=[3,1]<=[3]}, metadata={op_name="x"}
  wrapped_copy = f32[1,8]{1,0} fusion(param.1), kind=kLoop, calls=wrapped_copy_computation
  bitcast.20.0 = f32[8]{0} bitcast(wrapped_copy)
  all-reduce-start = f32[8]{0} all-reduce-start(bitcast.20.0), channel_id=1, replica_groups=[1,3]<=[3], use_global_device_ids=true, to_apply=region_0.3.clone, metadata={op_name="jit(contract_2)/jit(main)/reduce_sum" source_file="/root/parallel.py" source_line=18}, backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"collective_backend_config":{"is_sync":true,"no_parallel_custom_call":false},"force_earliest_schedule":false}
  ROOT all-reduce-done = f32[8]{0} all-reduce-done(all-reduce-start), metadata={op_name="jit(contract_2)/jit(main)/reduce_sum" source_file="/root/parallel.py" source_line=18}
}

```
