# tidystl-simd

Experimental implementation of Rust/SIMD backend for [tidystl](https://github.com/midoriao/tidystl). 

Install this package alongside `tidystl`, then register it once to use the
`tidystl_simd` backend:

```python
import tidystl
import tidystl_simd

tidystl.use(tidystl_simd)
```

The extension is built with maturin.
