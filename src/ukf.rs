// use nalgebra::{SMatrix, SVector};

// /// Type for state vectors of dimension S
// pub type State<const S: usize> = SVector<f64, S>;
// /// Type for covariance matricies of dimension SxS
// pub type Covariance<const S: usize> = SMatrix<f64, S, S>;
// /// Type for control vectors of dimension C
// pub type Control<const C: usize> = SVector<f64, C>;
// /// Type for measurement outputs of dimension Y
// pub type Output<const Y: usize> = SVector<f64, Y>;
// /// Type for cross covariance matrices of dimension SxY
// pub type CovarianceSY<const S: usize, const Y: usize> = SMatrix<f64, S, Y>;

// pub struct UnscentedKalmanFilter<const S: usize> {
//     pub state: State<S>,
//     pub covariance: Covariance<S>,
//     wm: [f64; 2 * S + 1],
//     wc: [f64; 2 * S + 1],
//     c: f64,
// }

// impl<const S: usize> UnscentedKalmanFilter<S> {
//     pub fn new(x0: State<S>, p0: Covariance<S>) -> Self {
//         Self {}
//     }
// }
