use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod algorithms;

fn check_interval(start: f64, end: f64) -> PyResult<()> {
    if start > end {
        return Err(PyValueError::new_err(format!(
            "invalid interval: start ({start}) > end ({end})"
        )));
    }
    Ok(())
}

fn check_inputs(times_len: usize, t_len: usize) -> PyResult<()> {
    if times_len == 0 {
        return Err(PyValueError::new_err("times array must be non-empty"));
    }
    if times_len != t_len {
        return Err(PyValueError::new_err(format!(
            "times length ({times_len}) does not match signal timesteps ({t_len})"
        )));
    }
    Ok(())
}

#[pyfunction]
fn eval_globally<'py>(
    py: Python<'py>,
    times: PyReadonlyArray1<'py, f64>,
    rho: PyReadonlyArray2<'py, f64>,
    start: f64,
    end: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    check_interval(start, end)?;
    check_inputs(times.as_array().len(), rho.as_array().dim().1)?;
    let result = algorithms::sliding_min(times.as_array(), rho.as_array(), start, end);
    Ok(result.into_pyarray_bound(py))
}

#[pyfunction]
fn eval_finally<'py>(
    py: Python<'py>,
    times: PyReadonlyArray1<'py, f64>,
    rho: PyReadonlyArray2<'py, f64>,
    start: f64,
    end: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    check_interval(start, end)?;
    check_inputs(times.as_array().len(), rho.as_array().dim().1)?;
    let result = algorithms::sliding_max(times.as_array(), rho.as_array(), start, end);
    Ok(result.into_pyarray_bound(py))
}

#[pyfunction]
fn eval_until<'py>(
    py: Python<'py>,
    times: PyReadonlyArray1<'py, f64>,
    p: PyReadonlyArray2<'py, f64>,
    q: PyReadonlyArray2<'py, f64>,
    start: f64,
    end: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    check_interval(start, end)?;
    check_inputs(times.as_array().len(), p.as_array().dim().1)?;
    let result = algorithms::eval_until(times.as_array(), p.as_array(), q.as_array(), start, end);
    Ok(result.into_pyarray_bound(py))
}

#[pymodule]
fn _ext(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eval_globally, m)?)?;
    m.add_function(wrap_pyfunction!(eval_finally, m)?)?;
    m.add_function(wrap_pyfunction!(eval_until, m)?)?;
    Ok(())
}
