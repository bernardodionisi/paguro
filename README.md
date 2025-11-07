# `Paguro`: data validation using Polars

<p align="center">
  <img src="./docs/imgs/logo/logo-paguro.png" alt="Paguro Logo" width="50%">
</p>


`Paguro` is a Python library built on Polars that provides efficient and rich tools for:

- Data **validation** and **models**
- Custom **information management**
- Comprehensive **summary statistics**
- And much more!

---

- ➪ [Documentation](https://bernardodionisi.github.io/paguro/latest/)
    - ➪ [API reference](https://bernardodionisi.github.io/paguro/latest/)

## Installation

`pip install paguro`

## Quick Start

### Data Validation

#### Expressive API

`Paguro` introduces a new expressive API for defining validation

Here is a basic example of what it looks like:

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