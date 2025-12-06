import Lake
open System Lake DSL

package «bqp_np» where
  -- Start with Mathlib on stable toolchain
  -- QuantumInfo can be added later once compatibility is resolved

require "leanprover-community" / "mathlib"

@[default_target]
lean_lib BQP_NP where
  -- Library configuration
