// fn cost(mut x: [f64; 4], u: &[f64]) -> f64 {
//     let mut c = 0.0;
//     for e in u.iter() {
//         dynamics(&mut x, *e);
//         let x_g = x[0] - x[2].sin() * L;
//         let x_g_dot = x[1] - x[3].sin() * L;
//         let x_g_dot_ref = -x_g;
//         let x_g_dot_err = x_g_dot - x_g_dot_ref;
//         let theta_ref = 0.0;
//         let theta_err = x[2] - theta_ref;
//         let theta_dot_ref = theta_err * 0.1;
//         let theta_dot_err = x[3] - theta_dot_ref;
//         let out_err = e - theta_dot_err;
//         c += GAIN[0] * x_g.powi(2)
//             + GAIN[1] * x_g_dot_err.powi(4)
//             + GAIN[2] * theta_err.powi(4)
//             + GAIN[3] * theta_dot_err.powi(4)
//             + GAIN[4] * out_err.powi(2);
//     }
//     // c.clamp(f64::MIN, f64::MAX)
//     c
// }

// 勾配
// fn grad_cost(mut x: [f64; 4], u: &[f64], grad: &mut [f64]) {
//     for i in 0..N - 1 {
//         let du = (u[i + 1] - u[i]) / DT;
//         let du = if du == 0.0 { 1e-6 } else { du };
//         grad[i] = (GAIN[0] * x[0] * x[1] + GAIN[2] * x[2] * x[3]) / du + GAIN[4] * u[i];
//         dynamics(&mut x, u[i]);
//     }
//     let du = (u[N - 1] - u[N - 2]) / DT;
//     let du = if du == 0.0 { 1e-6 } else { du };
//     grad[N - 1] = (GAIN[0] * x[0] * x[1] + GAIN[2] * x[2] * x[3]) / du + GAIN[4] * u[N - 1];
// }
// fn grad_cost(mut x: [f64; 4], u: &[f64], grad: &mut [f64]) {
//     const EPS: f64 = 0.001;
//     for i in 0..N {
//         let mut u_cpy: [f64; N] = u.try_into().expect("slice with incorrect length");
//         u_cpy[i] += EPS;
//         let posi = cost(x, &u_cpy);
//         u_cpy[i] -= 2.0 * EPS;
//         let nega = cost(x, &u_cpy);
//         grad[i] = (posi - nega) / (2.0 * EPS);
//         dynamics(&mut x, u[i]);
//     }
// }

fn main() {}
