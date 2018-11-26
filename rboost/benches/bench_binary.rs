#![feature(core_intrinsics)]

/// This is a benchmark for optimising the binary logistic loss function.
///
/// The exact_naive function is quite slow: 50us per iteration, so 5s for 100k elements
/// Without any precision loss, we can push it to 17.193 us with exact_factored_ln1p.
///
/// Using Taylor series, we can push it to 3-4ms with an error < 1e-8.
/// What is important is to be sure to remove ALL divisions. eg. x*3./4. => x*(3./4.)
/// It's not clear if the optimisations for fast_mul and factorisation are worth it.
///
/// We can also use Shanks methods on the Taylor series. It does works, but is slower than raw
/// Taylor for the same precision. However it looks like more stable for reaching errors
/// from 1e-10...1e-11

#[macro_use]
extern crate criterion;
extern crate lazy_static;

use criterion::{Bencher, Criterion};
use lazy_static::*;
use rand::prelude::*;
use std::fmt::Debug;
use std::fmt::Error;
use std::fmt::Formatter;
use std::intrinsics::{fadd_fast, fdiv_fast, fmul_fast, fsub_fast};

type Float = f64;

lazy_static! {
    static ref LN_2: Float = (2. as Float).ln();
}

// Naive formula.
// f32: 26.157 us 26.194 us 26.237 us
// f64: 49.944 us 50.028 us 50.127 us
fn exact_naive(y: Float, x: Float) -> Float {
    let proba = 1. / (1. + (-x).exp());
    let loss_1 = -y * proba.max(1e-8).ln();
    let loss_2 = -(1. - y) * (1. - proba).max(1e-8).ln();
    let loss = loss_1 + loss_2;
    loss
}

// Same formula, but reformulated
// f32: 18.505 us 18.531 us 18.560 us. Precision VS exact_naive: 1e-5.7
// f64: 27.072 us 27.121 us 27.175 us. Precision VS exact_naive: 1e-14
fn exact_factored(y: Float, x: Float) -> Float {
    let a = (1. + (-x).exp()).ln();
    let loss_ = -(1. - y) * (-x) + a;
    loss_
}

// Use ln_1p instead of ln(1+...)
// f32: 14.217 us 14.238 us 14.261 us
// f64: 17.158 us 17.193 us 17.232 us
fn exact_factored_ln1p(y: Float, x: Float) -> Float {
    let a = ((-x).exp()).ln_1p();
    let loss_ = (1. - y) * x + a;
    loss_
}

/// Long taylor series expansion around x=0.
/// f64 with x**6:  2.1638 us 2.1728 us 2.1853 us. Precision 1e-4.6
/// f32 with x**8: 2.6404 us 2.6438 us 2.6476 us. Precision 1e-5.4
/// f64 with x**8:  3.4251 us 3.4326 us 3.4412 us. Precision: 1e-5.6
/// f32 with x**10: 3.1725 us 3.1769 us 3.1818 us. Precision 1e-5.7
/// f64 with x**10: 3.7915 us 3.7965 us 3.8024 us. Precision: 1e-6.7
/// f64 with x**12: 4.4420 us 4.4577 us 4.4883 us. Precision: 1e-7.8
/// f64 with x**14: 4.6436 us 4.7458 us 4.8832 us. Precision: 1e-8.9
/// f64 with x**16: 5.4314 us 5.4517 us 5.4787 us. Precision: 1e-9.9
/// f64 with x**18: 5.7089 us 5.7176 us 5.7275 us. Precision: 1e-9.6
///
/// Calculated using Sympy
/// >>> from sympy import *
/// >>> x, y, z, t = symbols('x y z t')
/// >>> loss = -y*log(1/(1+exp(-x))) - (1-y)*log(1-1/(1+exp(-x)))
/// >>>  dn(l, n):
/// >>>     if n==0: return l.subs(x, 0)
/// >>>     return dn(diff(l, x), n-1) / n
/// >>> dn(loss, 2)
/// 1 / 8
fn taylor_naive(y: Float, x: Float) -> Float {
    let loss: Float = *LN_2;
    let loss = loss + (0.5 - y) * x;
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x2 * x4;
    let x8 = x4 * x4;
    let x10 = x4 * x6;
    let x12 = x6 * x6;
    let x14 = x6 * x8;
    let x16 = x8 * x8;
    let x18 = x10 * x8;
    let loss = loss + x2 / 8.;
    let loss = loss - x4 / 192.;
    let loss = loss + x6 / 2880.;
    let loss = loss - x8 * (17. / 645120.);
    let loss = loss + x10 * (31. / 14515200.);
    let loss = loss - x12 * (691. / 3832012800.);
    let loss = loss + x14 * (5461. / 348713164800.);
    let loss = loss - x16 * (929569. / 669529276416000.);
    let loss = loss - x18 * (3202291. / 25609494822912000.);
    loss
}

/// Same as above, but optimised with less multiplications. Faster and more precise.
/// f32 with x**6: 1.6010 us 1.6033 us 1.6058 us. Precision 1e-4.7
/// f32 with x**8: 1.9207 us 1.9233 us 1.9264 us. Precision 1e-5.8
/// f64 with x**8: 1.9214 us 1.9237 us 1.9262 us. Precision: 1e-5.8
/// f64 with x**10: 2.1146 us 2.1175 us 2.1207 us. Precision: 1e-6.8
/// f64 with x**12: 3.3194 us 3.3236 us 3.3282 us. Precision: 1e-7.8
/// f64 with x**14: 3.7556 us 3.7814 us 3.8317 us. Precision: 1e-8.9
fn taylor_optimised(y: Float, x: Float) -> Float {
    let loss: Float = *LN_2;
    let loss = loss + (0.5 - y) * x;
    let x2 = x * x;
    let loss = loss
        + x2 * ((1. / 8.) // x**2
        + x2 * ((-1. / 192.) // x**4
        + x2 * ((1. / 2880. )// x**6
        + x2 * ((-17. / 645120.) // x**8
        + x2 * ((31. / 14515200.) // x**10
        + x2 * ((-691. / 3832012800.) // x**12
        + x2 * (5461. / 348713164800.) // x**14
    ))))));
    loss
}

// What if we use fmul_fast? It's faster, but not stable yet.
// f64 with x**14: 3.0418 us 3.0492 us 3.0603 us. Precision: 1e-8.9
fn taylor_fast_mul(y: Float, x: Float) -> Float {
    unsafe {
        let loss: Float = *LN_2;
        let loss = loss + fmul_fast(0.5 - y, x);
        let x2 = fmul_fast(x, x);
        loss + fmul_fast(
            x2,
            1. / 8. // x**2
         + fmul_fast(x2 , (-1. / 192.) // x**4
         + fmul_fast(x2 , (1. / 2880.) // x**6
         + fmul_fast(x2 , (-17. / 645120.) // x**8
         + fmul_fast(x2 , (31. / 14515200. )// x**10
         + fmul_fast(x2 , (-691. / 3832012800.) // x**12
         + fmul_fast(x2 , 5461. / 348713164800. // x**14
        )))))),
        )
    }
}

fn _shanks(prec: Float, current: Float, next: Float) -> Float {
    (next * prec - current.powi(2)) / (next + prec - 2. * current)
}

/// Shanks derivative
/// f64 with x**10: 5.8229 us 5.8437 us 5.8732 us. Precision: 1e-8.3
/// f64 with x**12: 6.6162 us 6.6332 us 6.6527 us. Precision: 1e-9.5
/// f64 with x**14: 8.1196 us 8.1410 us 8.1638 us. Precision: 1e-10.7
fn shanks_naive(y: Float, x: Float) -> Float {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x2 * x4;
    let x8 = x4 * x4;
    let x10 = x4 * x6;
    let x12 = x6 * x6;
    let x14 = x8 * x6;

    let a2 = x2 / 8.;
    let a4 = a2 - x4 / 192.;
    let a6 = a4 + x6 / 2880.;
    let a8 = a6 - x8 * 17. / 645120.;
    let a10 = a8 + x10 * 31. / 14515200.;
    let a12 = a10 - x12 * 691. / 3832012800.;
    let a14 = a12 + x14 * 5461. / 348713164800.;

    //let shank_val = shanks(a6, a8, a10);
    //let shank_val = shanks(a8, a10, a12);
    let shank_val = _shanks(a10, a12, a14);

    let loss: Float = *LN_2;
    let loss = loss + (0.5 - y) * x;
    let loss = loss + shank_val;
    loss
}

/// Optimised Shanks derivative
/// f64 with x**14: 7.8129 us 7.9778 us 8.2055 us. Precision: 1e-10.7
fn shanks_optimised(y: Float, x: Float) -> Float {
    let x2 = x * x;
    let x12 = x2.powi(6);
    let x14 = x2.powi(7);

    let a10 = x2
        * (1. / 8. // x**2
        + x2 * (-1. / 192. // x**4
        + x2 * (1. / 2880. // x**6
        + x2 * (-17. / 645120. // x**8
        + x2 * 31. / 14515200. // x**10
    ))));
    let a12 = a10 - x12 * 691. / 3832012800.;
    let a14 = a12 + x14 * 5461. / 348713164800.;
    let shank_val = _shanks(a10, a12, a14);
    let loss: Float = *LN_2;
    let loss = loss + (0.5 - y) * x;
    let loss = loss + shank_val;
    loss
}

fn _shanks_fast_mul(prec: Float, current: Float, next: Float) -> Float {
    unsafe {
        let top = fsub_fast(fmul_fast(next, prec), fmul_fast(current, current));
        let bottom = fadd_fast(next, fsub_fast(prec, fmul_fast(current, 2.)));
        fdiv_fast(top, bottom)
    }
}

/// f64 with x**14: 6.3917 us 6.4321 us 6.4819 us. Precision 1e-10.6
/// f64 with x**14: 7.3272 us 7.3686 us 7.4282 us. Precision 1e-11.1
fn shanks_fast_mul(y: Float, x: Float) -> Float {
    unsafe {
        let x2 = fmul_fast(x, x);
        let x4 = fmul_fast(x2, x2);
        let x8 = fmul_fast(x4, x4);
        let x12 = fmul_fast(x8, x4);
        let x14 = fmul_fast(x12, x2);

        let z12 = fmul_fast(x2, -691. / 3832012800.);
        let z10 = fmul_fast(x2, fadd_fast(31. / 14515200., z12));
        let z8 = fmul_fast(x2, fadd_fast(-17. / 645120., z10));
        let z6 = fmul_fast(x2, fadd_fast(1. / 2880., z8));
        let z4 = fmul_fast(x2, fadd_fast(-1. / 192., z6));
        let z2 = fmul_fast(x2, fadd_fast(1. / 8., z4));

        let a12 = z2;
        let a14 = fadd_fast(a12, fmul_fast(x12, 5461. / 348713164800.));
        let a16 = fsub_fast(a14, fmul_fast(x14, 929569. / 669529276416000.));
        let shank_val = _shanks_fast_mul(a12, a14, a16);

        let loss: Float = *LN_2;
        let loss = fadd_fast(loss, fmul_fast(0.5 - y, x));
        let loss = fadd_fast(loss, shank_val);
        loss
    }
}

/// Shanks derivative at order 2.
/// f64 with x**14: 14.127 us 14.145 us 14.167 us. Precision: 1e-9.7
fn shanks2_naive(y: Float, x: Float) -> Float {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x2 * x4;
    let x8 = x4 * x4;
    let x10 = x4 * x6;
    let x12 = x6 * x6;
    let x14 = x6 * x8;

    let a2 = x2 / 8.;
    let a4 = a2 - x4 / 192.;
    let a6 = a4 + x6 / 2880.;
    let a8 = a6 - x8 * 17. / 645120.;
    let a10 = a8 + x10 * 31. / 14515200.;
    let a12 = a10 - x12 * 691. / 3832012800.;
    let a14 = a12 + x14 * 5461. / 348713164800.;

    let shank_val_8 = _shanks(a6, a8, a10);
    let shank_val_10 = _shanks(a8, a10, a12);
    let shank_val_12 = _shanks(a10, a12, a14);

    let shank_val = _shanks(shank_val_8, shank_val_10, shank_val_12);

    let loss: Float = *LN_2;
    let loss = loss + (0.5 - y) * x;
    let loss = loss + shank_val;
    loss
}

// This function have a better precision than shanks2, which indicates some numerical errors
// f64 with x**14: 10.567 us 10.656 us 10.742 us. Precision: 1e-10.3
fn shanks2_fast_mul(y: Float, x: Float) -> Float {
    unsafe {
        let x2 = fmul_fast(x, x);
        let x4 = fmul_fast(x2, x2);
        let x6 = fmul_fast(x2, x4);
        let x8 = fmul_fast(x4, x4);
        let x10 = fmul_fast(x4, x6);
        let x12 = fmul_fast(x6, x6);
        let x14 = fmul_fast(x6, x8);

        let a2 = fmul_fast(x2, 1. / 8.);
        let a4 = fsub_fast(a2, fmul_fast(x4, 1. / 192.));
        let a6 = fadd_fast(a4, fmul_fast(x6, 1. / 2880.));
        let a8 = fsub_fast(a6, fmul_fast(x8, 17. / 645120.));
        let a10 = fadd_fast(a8, fmul_fast(x10, 31. / 14515200.));
        let a12 = fsub_fast(a10, fmul_fast(x12, 691. / 3832012800.));
        let a14 = fadd_fast(a12, fmul_fast(x14, 5461. / 348713164800.));

        let shank_val_8 = _shanks_fast_mul(a6, a8, a10);
        let shank_val_10 = _shanks_fast_mul(a8, a10, a12);
        let shank_val_12 = _shanks_fast_mul(a10, a12, a14);

        let shank_val = _shanks_fast_mul(shank_val_8, shank_val_10, shank_val_12);

        let loss: Float = *LN_2;
        let loss = fadd_fast(loss, fmul_fast(0.5 - y, x));
        let loss = fadd_fast(loss, shank_val);
        loss
    }
}

struct Fn_ {
    f: &'static (Fn(f64, f64) -> f64),
    name: &'static str,
}

impl Fn_ {
    fn new(name: &'static str, f: &'static Fn(f64, f64) -> f64) -> Self {
        Self { f, name }
    }
}

impl Debug for Fn_ {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        f.write_str(self.name)
    }
}

fn get_functions() -> Vec<Fn_> {
    vec![
        Fn_::new("exact_naive", &exact_naive),
        Fn_::new("exact_factored", &exact_factored),
        Fn_::new("exact_factored_ln1p", &exact_factored_ln1p),
        Fn_::new("taylor_naive", &taylor_naive),
        Fn_::new("taylor_optimised", &taylor_optimised),
        Fn_::new("taylor_fast_mul", &taylor_fast_mul),
        Fn_::new("shanks_naive", &shanks_naive),
        Fn_::new("shanks_optimised", &shanks_optimised),
        Fn_::new("shanks_fast_mul", &shanks_fast_mul),
        Fn_::new("shanks2_naive", &shanks2_naive),
        Fn_::new("shanks2_fast_mul", &shanks2_fast_mul),
    ]
}

fn get_proba_target() -> (Vec<Float>, Vec<Float>) {
    let seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]; // byte array
    let mut rng = SmallRng::from_seed(seed);

    let n = 1000;
    let probas: Vec<_> = (0..n)
        .map(|_| (rng.gen::<Float>() * 2. - 1.) * 20.)
        .collect();
    let target: Vec<_> = (0..n)
        .map(|_| if rng.gen::<Float>() > 0.5 { 1. } else { 0. })
        .collect();
    (probas, target)
}

fn bench_tree(b: &mut Bencher, f: &Fn_) {
    let (probas, target) = get_proba_target();
    b.iter(|| {
        let mut sum = 0.;
        for (&a, &b) in probas.iter().zip(&target) {
            sum += (f.f)(a, b);
        }
        sum
    })
}

fn criterion_benchmark(c: &mut Criterion) {
    let (probas, target) = get_proba_target();

    for f in get_functions() {
        let mut max_diff: Float = 0.;
        for (&a, &b) in probas.iter().zip(&target) {
            max_diff = max_diff.max(((f.f)(a, b) - exact_naive(a, b)).abs());
        }
        println!("max_diff for {} = e{:.1}", f.name, max_diff.log10());
    }

    c.bench_function_over_inputs("binary", bench_tree, get_functions());
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
