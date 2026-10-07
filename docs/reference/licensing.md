# License and third-party software

exo-tbccl is licensed under the Apache License, Version 2.0. The full text is in the `LICENSE` file at the repository root. Copyright 2026 Sandesh Bharadwaj.

Contributions are accepted under the same license (Apache-2.0, section 5).

## exo and MLX

exo-tbccl does not contain or redistribute exo source code. It is a separate package that exo imports at run time when the `MlxTbccl` instance type is selected. The benchmark and test files that import exo modules use them as installed libraries and contain no copy of them. exo is licensed Apache-2.0; at the audited revision it has a `LICENSE` file and no `NOTICE` file.

The exo-side changes that exo-tbccl relies on (the pipeline communication seam and the runner byte exchange) are not part of this repository. They are changes to exo itself and, wherever they are published, remain under exo's Apache-2.0 license and notices.

MLX is a run-time dependency supplied by exo's environment (MIT licensed); exo-tbccl does not declare or bundle it.

## Static linking of libtbccl

The compiled extension links the installed TBCCL C ABI static library (`TBCCL::tbccl_c`) into itself. A wheel therefore redistributes TBCCL code. TBCCL and exo-tbccl are both Apache-2.0 with the same copyright holder, so the wheel's `LICENSE` file covers both and no additional notice is required for the embedded TBCCL code.

## Notices

No third-party `NOTICE` file or license bundle is required for the package as built: it contains no third-party code. A `NOTICE` file is not provided.
