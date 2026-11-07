# Attention notes (to be written by me)

## Why online softmax is exact
- 
- 

## Memory complexity: O(N^2) vs O(N)
- 
- 

## Why an educational PyTorch loop may not beat fused kernels
- Python overhead:
- Kernel launches:
- No SRAM fusion / HBM traffic:

## What the measured results show
- Time vs sequence length:
- Peak memory vs sequence length:
- Tiled block size effect:
- Comparison with SDPA backends (math / mem-efficient / flash):
- Errors vs fp32 baseline:
- Caveats (GPU, dtype, repeats):
