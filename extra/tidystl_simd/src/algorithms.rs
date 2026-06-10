use ndarray::parallel::prelude::*;
use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use wide::f64x4;

fn interp_at_row(times: &[f64], rho: &[f64], t: f64) -> f64 {
    let t_first = times[0];
    let t_last = *times.last().unwrap();
    if t <= t_first { return rho[0]; }
    if t >= t_last { return *rho.last().unwrap(); }
    // searchsorted(times, t, side="right") - 1
    let pos = times.partition_point(|&x| x <= t);
    let idx = pos - 1;
    let (t0, t1) = (times[idx], times[idx + 1]);
    if t1 == t0 { return rho[idx]; }
    let alpha = (t - t0) / (t1 - t0);
    rho[idx] + alpha * (rho[idx + 1] - rho[idx])
}

fn simd_min_reduce(slice: &[f64]) -> f64 {
    // NOTE: wide::f64x4::min follows Intel MINPD semantics — when one operand is NaN the other
    // is returned (NaN is dropped). For NaN-propagating semantics use the native backend.
    let mut acc = f64x4::splat(f64::INFINITY);
    let chunks = slice.chunks_exact(4);
    let remainder = chunks.remainder();
    for chunk in chunks {
        acc = acc.min(f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let arr: [f64; 4] = acc.into();
    let mut result = arr[0].min(arr[1]).min(arr[2]).min(arr[3]);
    for &x in remainder {
        result = result.min(x);
    }
    result
}

fn simd_max_reduce(slice: &[f64]) -> f64 {
    // NOTE: wide::f64x4::max follows Intel MAXPD semantics — when one operand is NaN the other
    // is returned (NaN is dropped). For NaN-propagating semantics use the native backend.
    let mut acc = f64x4::splat(f64::NEG_INFINITY);
    let chunks = slice.chunks_exact(4);
    let remainder = chunks.remainder();
    for chunk in chunks {
        acc = acc.max(f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let arr: [f64; 4] = acc.into();
    let mut result = arr[0].max(arr[1]).max(arr[2]).max(arr[3]);
    for &x in remainder {
        result = result.max(x);
    }
    result
}

fn sliding_reduce_row(
    times: &[f64],
    rho: &[f64],
    start: f64,
    end: f64,
    reduce: fn(&[f64]) -> f64,
    init: f64,
    combine: fn(f64, f64) -> f64,
) -> Vec<f64> {
    let t_len = times.len();
    let t_first = times[0];
    let t_last = *times.last().unwrap();
    let eps = 1e-12_f64;
    let mut out = vec![0.0_f64; t_len];

    for i in 0..t_len {
        let t_lo = times[i] + start;
        let t_hi = times[i] + end;
        let t_lo_c = t_lo.max(t_first);
        let t_hi_c = t_hi.min(t_last);

        // searchsorted(times, t_lo_c, side="left")
        let j_start = times.partition_point(|&x| x < t_lo_c).min(t_len);
        // searchsorted(times, t_hi_c, side="right")
        let j_end = times.partition_point(|&x| x <= t_hi_c).min(t_len);

        let need_left = j_start < t_len
            && (times[j_start.min(t_len - 1)] - t_lo_c).abs() > eps
            && t_lo_c > t_first + eps;
        let need_right = j_end > 0
            && (times[(j_end - 1).min(t_len - 1)] - t_hi_c).abs() > eps
            && t_hi_c < t_last - eps;

        let mut acc = init;
        let mut has_candidate = false;

        if j_start < j_end {
            acc = combine(acc, reduce(&rho[j_start..j_end]));
            has_candidate = true;
        }
        if need_left {
            acc = combine(acc, interp_at_row(times, rho, t_lo_c));
            has_candidate = true;
        }
        if need_right {
            acc = combine(acc, interp_at_row(times, rho, t_hi_c));
            has_candidate = true;
        }
        if !has_candidate {
            // Unreachable for any valid (non-empty) signal: either the sample slice is non-empty
            // or one of the boundary interpolation branches fires. Kept as a safe fallback.
            let midpoint = (t_lo_c + t_hi_c) / 2.0;
            acc = interp_at_row(times, rho, midpoint);
        }

        out[i] = acc;
    }
    out
}

pub fn sliding_min(
    times: ArrayView1<f64>,
    rho: ArrayView2<f64>,
    start: f64,
    end: f64,
) -> Array2<f64> {
    let (n, t_len) = rho.dim();
    let _times_owned;
    let times_s: &[f64] = if let Some(s) = times.as_slice() {
        s
    } else {
        _times_owned = times.to_vec();
        &_times_owned
    };
    let mut out = Array2::<f64>::zeros((n, t_len));

    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(rho.axis_iter(Axis(0)))
        .for_each(|(mut out_row, rho_row)| {
            let _rho_owned;
            let rho_s: &[f64] = if let Some(s) = rho_row.as_slice() {
                s
            } else {
                _rho_owned = rho_row.to_vec();
                &_rho_owned
            };
            let row_result = sliding_reduce_row(times_s, rho_s, start, end,
                simd_min_reduce, f64::INFINITY, f64::min);
            out_row.as_slice_mut().unwrap().copy_from_slice(&row_result);
        });

    out
}

pub fn sliding_max(
    times: ArrayView1<f64>,
    rho: ArrayView2<f64>,
    start: f64,
    end: f64,
) -> Array2<f64> {
    let (n, t_len) = rho.dim();
    let _times_owned;
    let times_s: &[f64] = if let Some(s) = times.as_slice() {
        s
    } else {
        _times_owned = times.to_vec();
        &_times_owned
    };
    let mut out = Array2::<f64>::zeros((n, t_len));

    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(rho.axis_iter(Axis(0)))
        .for_each(|(mut out_row, rho_row)| {
            let _rho_owned;
            let rho_s: &[f64] = if let Some(s) = rho_row.as_slice() {
                s
            } else {
                _rho_owned = rho_row.to_vec();
                &_rho_owned
            };
            let row_result = sliding_reduce_row(times_s, rho_s, start, end,
                simd_max_reduce, f64::NEG_INFINITY, f64::max);
            out_row.as_slice_mut().unwrap().copy_from_slice(&row_result);
        });

    out
}

fn eval_until_row(
    times: &[f64],
    p: &[f64],
    q: &[f64],
    start: f64,
    end: f64,
) -> Vec<f64> {
    let t_len = times.len();
    let t_last = *times.last().unwrap();
    let mut out = vec![f64::NEG_INFINITY; t_len];

    for i in 0..t_len {
        let t_i = times[i];
        // searchsorted(times, t_i + start, side="left")
        let j_start = times.partition_point(|&x| x < t_i + start).min(t_len);
        // searchsorted(times, t_i + end, side="right")
        let j_end = times.partition_point(|&x| x <= t_i + end).min(t_len);

        // Prefix min over p[i..j_start] (the part before the witness window starts)
        let p_pre = if j_start > i {
            simd_min_reduce(&p[i..j_start])
        } else {
            f64::INFINITY
        };

        let mut p_prefix = p_pre;
        for j in j_start..j_end {
            p_prefix = p_prefix.min(p[j]);
            let candidate = q[j].min(p_prefix);
            if candidate > out[i] {
                out[i] = candidate;
            }
        }

        // When the window extends past the last sample the signal is held flat at q[t_len-1].
        // This branch fires even when j_start = j_end = t_len (witness window entirely past
        // the signal end, e.g. U[1,3] evaluated at the last timestep).
        if t_i + end > t_last {
            // j_end = t_len always here (partition_point returns t_len when all samples
            // satisfy x <= t_i+end), so the p[j_end..t_len] slice is empty — no inner guard.
            let candidate = q[t_len - 1].min(p_prefix);
            if candidate > out[i] {
                out[i] = candidate;
            }
        }
    }
    out
}

pub fn eval_until(
    times: ArrayView1<f64>,
    p: ArrayView2<f64>,
    q: ArrayView2<f64>,
    start: f64,
    end: f64,
) -> Array2<f64> {
    let (n, t_len) = p.dim();
    let _times_owned;
    let times_s: &[f64] = if let Some(s) = times.as_slice() {
        s
    } else {
        _times_owned = times.to_vec();
        &_times_owned
    };
    let mut out = Array2::<f64>::zeros((n, t_len));

    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(p.axis_iter(Axis(0)).into_par_iter())
        .zip(q.axis_iter(Axis(0)).into_par_iter())
        .for_each(|((mut out_row, p_row), q_row)| {
            let _p_owned;
            let p_s: &[f64] = if let Some(s) = p_row.as_slice() {
                s
            } else {
                _p_owned = p_row.to_vec();
                &_p_owned
            };
            let _q_owned;
            let q_s: &[f64] = if let Some(s) = q_row.as_slice() {
                s
            } else {
                _q_owned = q_row.to_vec();
                &_q_owned
            };
            let row_result = eval_until_row(times_s, p_s, q_s, start, end);
            out_row.as_slice_mut().unwrap().copy_from_slice(&row_result);
        });

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn assert_arr2_close(a: &Array2<f64>, b: &Array2<f64>, tol: f64) {
        assert_eq!(a.shape(), b.shape(), "shape mismatch");
        for (x, y) in a.iter().zip(b.iter()) {
            assert_abs_diff_eq!(x, y, epsilon = tol);
        }
    }

    #[test]
    fn test_sliding_min_aligned_window() {
        let times = array![0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let rho = array![[2.0_f64, -1.0, 3.0, -2.0, 1.0]];
        let result = sliding_min(times.view(), rho.view(), 0.0, 1.0);
        let expected = array![[-1.0_f64, -1.0, -2.0, -2.0, 1.0]];
        assert_arr2_close(&result, &expected, 1e-10);
    }

    #[test]
    fn test_sliding_min_interp_boundaries() {
        let times = array![0.0_f64, 2.0, 4.0];
        let rho = array![[-1.0_f64, 10.0, -1.0]];
        let result = sliding_min(times.view(), rho.view(), 0.5, 1.5);
        let expected = array![[1.75_f64, 1.75, -1.0]];
        assert_arr2_close(&result, &expected, 1e-10);
    }

    #[test]
    fn test_sliding_max_simple() {
        let times = array![0.0_f64, 1.0, 2.0];
        let rho = array![[2.0_f64, -1.0, 3.0]];
        let result = sliding_max(times.view(), rho.view(), 0.0, 1.0);
        let expected = array![[2.0_f64, 3.0, 3.0]];
        assert_arr2_close(&result, &expected, 1e-10);
    }

    #[test]
    fn test_eval_until_simple() {
        // U[0,2] on t=[0,1,2,3,4]
        // p=[1,-1,1,-1,1], q=[-1,1,-1,1,-1]
        // At t=0: j_start=searchsorted_left(0+0)=0, j_end=searchsorted_right(0+2)=3 => j in [0,1,2]
        //   p_pre = min(p[0..0]) = INFINITY (j_start==i==0)
        //   j=0: p_prefix=min(INF,p[0])=min(INF,1)=1;   min(q[0],1)=min(-1,1)=-1; out[0]=max(-INF,-1)=-1
        //   j=1: p_prefix=min(1,p[1])=min(1,-1)=-1;     min(q[1],-1)=min(1,-1)=-1; out[0]=max(-1,-1)=-1
        //   j=2: p_prefix=min(-1,p[2])=min(-1,1)=-1;    min(q[2],-1)=min(-1,-1)=-1; out[0]=max(-1,-1)=-1
        //   out[0] = -1
        // At t=1: j_start=searchsorted_left(1+0)=1, j_end=searchsorted_right(1+2)=4 => j in [1,2,3]
        //   p_pre = min(p[1..1]) = INFINITY (j_start==i==1)
        //   j=1: p_prefix=min(INF,-1)=-1; min(q[1],-1)=min(1,-1)=-1; out[1]=max(-INF,-1)=-1
        //   j=2: p_prefix=min(-1,1)=-1;  min(q[2],-1)=min(-1,-1)=-1; out[1]=max(-1,-1)=-1
        //   j=3: p_prefix=min(-1,-1)=-1; min(q[3],-1)=min(1,-1)=-1; out[1]=max(-1,-1)=-1
        //   out[1] = -1
        let times = array![0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let p = array![[1.0_f64, -1.0, 1.0, -1.0, 1.0]];
        let q = array![[-1.0_f64, 1.0, -1.0, 1.0, -1.0]];
        let result = eval_until(times.view(), p.view(), q.view(), 0.0, 2.0);
        assert_abs_diff_eq!(result[[0, 0]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 1]], -1.0, epsilon = 1e-10);
    }
}
