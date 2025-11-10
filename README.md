# `Paguro`: data validation using Polars

<p align="center">
  <img src="./docs/imgs/logo/logo-paguro.png" alt="Paguro Logo" width="50%">
</p>


`Paguro` is a Python library built on top Polars that provides efficient and rich tools for:

- Data **validation** and **models**
- Persistent custom **information**
- Customizable **summary statistics**
- And much more!

*All with informative and beautiful terminal output!*

---

- ➪ [Documentation](https://bernardodionisi.github.io/paguro/latest/)
    - ➪ [API reference](https://bernardodionisi.github.io/paguro/latest/pages/api/)

## Installation

`pip install paguro`

## Quick Start

### Data Validation

#### Expressive API

`Paguro` introduces a new expressive API for defining validation

Here is a simple example:

```python
import paguro as pg
import polars as pl

valid_frame = pg.vframe(
    pg.vcol("a", dtype=int, ge=1),
    pg.vcol("b", b_contains=pl.all().str.contains("z")),
)

valid_frame.validate({"a": [0, 1, 2], "b": ["z", "y", "x"]})
```

```text
══ ValidationError ═══════════════════════════════
 ━━━━━━━━━━━━━━━ valid_frame_list ━━━━━━━━━━━━━━━ 
  ╭─ > "" ─────────────────────────────────────╮  
  │ ----------------------------- validators - │  
  │   valid_column_list                        │  
  │     * 'a'                                  │  
  │       constraints                          │  
  │         ‣ ge                               │  
  │           errors                           │  
  │             ┌─────┐                        │  
  │             │ a   │                        │  
  │             │ --- │                        │  
  │             │ i64 │                        │  
  │             ╞═════╡                        │  
  │             │ 0   │                        │  
  │             └─────┘                        │  
  │             shape: (1, 1)                  │  
  │     * 'b'                                  │  
  │       constraints                          │  
  │         ‣ b_contains                       │  
  │           errors                           │  
  │             ┌─────┐                        │  
  │             │ b   │                        │  
  │             │ --- │                        │  
  │             │ str │                        │  
  │             ╞═════╡                        │  
  │             │ y   │                        │  
  │             │ x   │                        │  
  │             └─────┘                        │  
  │             shape: (2, 1)                  │  
  │                                            │  
  ╰────────────────────────────────────────────╯  
                                                  
══════════════════════════════════════════════════
```

Paguro contains many features, including a model-based API for validation and static typing of
columns, please visit the [Documentation](https://bernardodionisi.github.io/paguro/latest/) and
stay tuned for more examples!

---

**Paguro** is distributed under
the [Apache License, Version 2.0](https://spdx.org/licenses/Apache-2.0.html).  
&copy; 2025 Bernardo Dionisi | SPDX-License-Identifier: Apache-2.0