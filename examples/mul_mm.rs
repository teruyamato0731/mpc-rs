extern crate nalgebra as na;

use std::ops::{Index, Mul};

use na::{matrix, vector, zero, Matrix, Scalar};

// Matrix of Matrix × Matrix of Matrix
fn mul_mm<T, R11, C11, S11, R12, C12, S12, R21, C21, S21, R22, C22, S22>(
    m1: &Matrix<Matrix<T, R12, C12, S12>, R11, C11, S11>,
    m2: &Matrix<Matrix<T, R22, C22, S22>, R21, C21, S21>,
) -> na::OMatrix<na::OMatrix<T, R12, C22>, R11, C21>
where
    R11: na::Dim + na::DimName,
    C11: na::Dim + na::DimName,
    R12: na::Dim + na::DimName,
    C12: na::Dim,
    R21: na::Dim,
    C21: na::Dim + na::DimName,
    R22: na::Dim,
    C22: na::Dim + na::DimName,
    T: num::Zero + Clone + Scalar + std::ops::AddAssign,
    na::constraint::ShapeConstraint: na::constraint::AreMultipliable<R11, C11, R21, C21>,
    na::constraint::ShapeConstraint: na::constraint::AreMultipliable<R12, C12, R22, C22>,
    na::DefaultAllocator: na::allocator::Allocator<T, R12, C22>,
    na::DefaultAllocator: na::allocator::Allocator<
        na::Matrix<
            T,
            R12,
            C22,
            <na::DefaultAllocator as na::allocator::Allocator<T, R12, C22>>::Buffer,
        >,
        R11,
        C21,
    >,
    Matrix<Matrix<T, R12, C12, S12>, R11, C11, S11>: Index<(usize, usize)> + Sized + Clone,
    Matrix<Matrix<T, R22, C22, S22>, R21, C21, S21>: Index<(usize, usize)> + Sized + Clone,
    <Matrix<Matrix<T, R12, C12, S12>, R11, C11, S11> as Index<(usize, usize)>>::Output:
        core::ops::Mul<
            <Matrix<Matrix<T, R22, C22, S22>, R21, C21, S21> as Index<(usize, usize)>>::Output,
        >,
    <Matrix<Matrix<T, R22, C22, S22>, R21, C21, S21> as Index<(usize, usize)>>::Output: Sized,
    na::OMatrix<T, R12, C22>: core::ops::AddAssign<<<Matrix<Matrix<T, R12, C12, S12>, R11, C11, S11> as Index<(usize, usize)>>::Output as Mul<<Matrix<Matrix<T, R22, C22, S22>, R21, C21, S21> as Index<(usize, usize)>>::Output>>::Output>,
    for<'a> &'a <Matrix<Matrix<T, R12, C12, S12>, R11, C11, S11> as Index<(usize, usize)>>::Output: Mul<&'a <Matrix<Matrix<T, R22, C22, S22>, R21, C21, S21> as Index<(usize, usize)>>::Output, Output = <<Matrix<Matrix<T, R12, C12, S12>, R11, C11, S11> as Index<(usize, usize)>>::Output as Mul<<Matrix<Matrix<T, R22, C22, S22>, R21, C21, S21> as Index<(usize, usize)>>::Output>>::Output>,
{
    let mut result = na::OMatrix::<na::OMatrix<T, R12, C22>, R11, C21>::zeros();

    // m1とm2の積を計算する
    for i in 0..R11::dim() {
        for j in 0..C21::dim() {
            for k in 0..C11::dim() {
                result[(i, j)] += &m1[(i, k)] * &m2[(k, j)];
            }
        }
    }

    result
}

fn main() {
    let x = na::Vector4::<f32>::zeros();
    let u = vector![1.0, 0.0, 0.0, 0.0];
    println!("{}", 2.0 * x + 3.0 * u);

    let a = na::Matrix4::<f32>::new(
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    );
    let b = na::Matrix4::<f32>::new(
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    );

    let _ = a * b;
    let _ = a * 2.0;
    let _ = x.dot(&u);

    println!("{}", a * x + b * u);
    // println!("{}", x.dot(&a) + b.dot(&u));

    let _ = matrix![1.0f32, 2.0, 3.0; 4.0, 5.0, 6.0; 7.0, 8.0, 9.0];

    let a = na::Matrix4::<f32>::identity();
    let b = na::Vector4::<f32>::zeros();
    let c = matrix![
        1.0f32, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 1.0, 0.0;
        0.0, 0.0, 0.0, 0.0
    ];

    let _f = matrix![
        a;
        a * a;
        a * a * a;
        a * a * a * a;
    ];
    let _g = matrix![
        b, zero(), zero(), zero();
        a * b, b, zero(), zero();
        a * a * b, a * b, b, zero();
        a * a * a * b, a * a * b, a * b, b;
    ];
    let _h = matrix![
        c, zero(), zero(), zero();
        zero(), c, zero(), zero();
        zero(), zero(), c, zero();
        zero(), zero(), zero(), c;
    ];
    // println!("{:?}", f);
    // println!("{:?}", g);
    // println!("{:?}", h);

    let _q =
        matrix![1.0, 0.0, 0.0, 0.0; 0.0, 1.0, 0.0, 0.0; 0.0, 0.0, 1.0, 0.0; 0.0, 0.0, 0.0, 1.0];
    let _r =
        matrix![1.0, 0.0, 0.0, 0.0; 0.0, 1.0, 0.0, 0.0; 0.0, 0.0, 1.0, 0.0; 0.0, 0.0, 0.0, 1.0];

    // let U = -(g.transpose() * h.transpose() * q * h * g);

    let _g = matrix![b];
    let _h = matrix![c];
    println!("{:?}", b);
    println!("{:?}", c);
    println!("{:?}", c * b);
    println!("{:?}", _g);
    println!("{:?}", _h);
    let _u = _h[(0, 0)] * _g[(0, 0)];
    let _u = mul_mm(&_h, &_g);

    let a = matrix![1.0, 2.0; 3.0, 4.0];
    let _c = matrix![a, a; a, a];

    println!("{:?}", _c * _c);

    println!("{:?}", mul_mm(&_c, &_c));
}
