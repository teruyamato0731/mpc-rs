extern crate nalgebra as na;

use na::Matrix3;

use nalgebra::{matrix, vector};

fn main() {
    let m = Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    println!("A = {}", m);

    let m = Matrix3::from_fn(|r, c| 10 * r + c);
    println!("B = {}", m);

    let m = m.fixed_view::<2, 2>(1, 0).clone_owned();
    println!("C = {}", m);

    let m1 = matrix![1,2; 3,4];
    let m2 = matrix![m1, m1; m1, m1];

    // Multiplication works for matrices of the same type `T`.
    let _ = m2 * m2;

    // Multiplication fails because the types `T` are different.
    // However, the inner matrices (2x2 and 2x1) can be multiplied, so this should be allowed.
    let v = vector![5, 6];
    let m3 = matrix![v, v; v, v];
    // let _ = m2 * m3;

    let _ = m1 * v;
}
