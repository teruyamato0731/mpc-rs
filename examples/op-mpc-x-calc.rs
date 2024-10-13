use optimization_engine::{panoc::*, *};
extern crate nalgebra as na;
use na::{matrix, stack, vector};

// 予測ホライゾン
const T: f64 = 0.8;
const N: usize = 8;
const DT: f64 = T / N as f64;

const A: na::Matrix4<f64> = matrix![
    1.0, DT, 0.0, 0.0;
    0.0, 1.0, -M2 * M2 * G * L * L / D * DT, 0.0;
    0.0, 0.0, 1.0, DT;
    0.0, 0.0, (M1 + M2 + J1 / R * R) / D * M2 * G * L * DT, 1.0
];
const B: na::Vector4<f64> = matrix![
    0.0;
    (M2 * L * L + J2) / D / R * KT * DT;
    0.0;
    -M2 * L / D / R * KT * DT;
];
const C: na::Matrix4<f64> = matrix![
    5.0, 0.0, 0.0, 0.0;
    0.0, 5.0, 0.0, 0.0;
    0.0, 0.0, 1.0, 0.0;
    0.0, 0.0, 0.0, 1.0
];

fn gen_ref(x: &na::Vector4<f64>) -> na::SMatrix<f64, 4, N> {
    let mut r = na::SMatrix::<f64, 4, N>::zeros();
    for i in 0..N {
        let phase = std::f64::consts::PI * i as f64 / N as f64;
        let phase_2 = std::f64::consts::PI / 2.0 * i as f64 / N as f64;
        r[(0, i)] = x[0] * phase_2.cos();
        r[(1, i)] = -x[0] * phase.sin();
        r[(2, i)] = (-0.5 * x[0]).clamp(-0.45, 0.45) * phase_2.cos();
        r[(3, i)] = (-0.5 * x[0]).clamp(-0.45, 0.45) * phase.sin();
    }
    r
}

fn cost(mut x: na::Vector4<f64>, u: &[f64]) -> f64 {
    let mut c = 0.0;
    // for e in u.iter() {
    //     x = dynamics(&x, *e);
    //     c += x.dot(&(C * x));
    // }
    let x_ref = gen_ref(&x);
    for (e, r) in u.iter().zip(x_ref.column_iter()) {
        x = dynamics(&x, *e);
        c += (x - r).dot(&(C * (x - r)));
    }
    c
}

// コスト関数の勾配を数値微分によって求める(並列化版)
// 1. 数値微分の基礎となるコストを計算
// 2. 初期値xに運動方程式を適用し、kステップ先の状態を求める
// 3. kステップ先の状態と、入力に対して微小な変化を加えてコストを計算
// 4. 3.で求めたコストと1.のコストの差分を取り、微分を求める
fn grad_cost(x: na::Vector4<f64>, u: &na::SVector<f64, N>) -> na::SVector<f64, N> {
    let a: na::SMatrix<f64, 32, 4> = stack![
        A.pow(1); A.pow(2); A.pow(3); A.pow(4); A.pow(5); A.pow(6); A.pow(7); A.pow(8);
    ];
    let g: na::SMatrix<f64, 32, 8> = stack![
        B, 0, 0, 0, 0, 0, 0, 0;
        A * B, B, 0, 0, 0, 0, 0, 0;
        A.pow(2) * B, A * B, B, 0, 0, 0, 0, 0;
        A.pow(3) * B, A.pow(2) * B, A * B, B, 0, 0, 0, 0;
        A.pow(4) * B, A.pow(3) * B, A.pow(2) * B, A * B, B, 0, 0, 0;
        A.pow(5) * B, A.pow(4) * B, A.pow(3) * B, A.pow(2) * B, A * B, B, 0, 0;
        A.pow(6) * B, A.pow(5) * B, A.pow(4) * B, A.pow(3) * B, A.pow(2) * B, A * B, B, 0;
        A.pow(7) * B, A.pow(6) * B, A.pow(5) * B, A.pow(4) * B, A.pow(3) * B, A.pow(2) * B, A * B, B;
    ];
    let q = stack![
        C, 0, 0, 0, 0, 0, 0, 0;
        0, C, 0, 0, 0, 0, 0, 0;
        0, 0, C, 0, 0, 0, 0, 0;
        0, 0, 0, C, 0, 0, 0, 0;
        0, 0, 0, 0, C, 0, 0, 0;
        0, 0, 0, 0, 0, C, 0, 0;
        0, 0, 0, 0, 0, 0, C, 0;
        0, 0, 0, 0, 0, 0, 0, C;
    ];

    let x_ref = gen_ref(&x);
    let x_ref = na::SVector::from_iterator(x_ref.iter().copied());
    2.0 * g.transpose() * q * (g * u + a * x - x_ref)
}

fn main() -> anyhow::Result<()> {
    let file_path = "logs/op-mpc-x.csv";
    let mut wtr = csv::Writer::from_path(file_path)?;

    let tolerance = 1e-6;
    let lbfgs_memory = 20;
    let max_iters = usize::MAX;
    let max_dur = std::time::Duration::from_secs_f64(DT);
    let mut panoc_cache = PANOCCache::new(N, tolerance, lbfgs_memory);

    let mut u = [0.0; N];

    let mut x = vector![0.5, 0.0, 0.1, 0.0];

    const MAX_ITERS: usize = (5.0 / DT) as usize;
    for i in 0..MAX_ITERS + 1 {
        let f = |u: &[f64], c: &mut f64| -> Result<(), SolverError> {
            *c = cost(x, u);
            Ok(())
        };

        let df = |u: &[f64], grad: &mut [f64]| -> Result<(), SolverError> {
            let u = na::SVector::<f64, N>::from_iterator(u.iter().copied());
            let g = grad_cost(x, &u);
            // gradを更新
            for (i, e) in g.iter().enumerate() {
                grad[i] = *e;
            }
            Ok(())
        };

        // define the bounds at every iteration
        // let bounds = constraints::NoConstraints::new();
        let bounds = constraints::Rectangle::new(Some(&[-30.0]), Some(&[30.0]));

        // the problem definition is updated at every iteration
        let problem = Problem::new(&bounds, df, f);

        // updated instance of the solver
        let mut panoc = PANOCOptimizer::new(problem, &mut panoc_cache)
            .with_max_iter(max_iters)
            .with_max_duration(max_dur);

        let status = panoc.solve(&mut u).expect("Failed to solve");
        if status.iterations() == 0 || u[0].abs() >= 30.0 {
            println!("status is invalid, u[0]: {}, status: {:?}", u[0], status);
            break;
        }

        x = dynamics(&x, u[0]);

        let mut x_est = x;
        for e in u.iter() {
            x_est = dynamics(&x_est, *e);
        }

        let t = i as f64 * DT;
        print!("{t:4.2}, {:7.2}, ", u[0]);
        print!(
            "act: ({:7.2},{:7.2},{:7.2},{:7.2}) ",
            x[0], x[1], x[2], x[3]
        );
        println!();

        wtr.write_record(&[
            (i as f64 * DT).to_string(),
            u[0].to_string(),
            x[0].to_string(),
            x[1].to_string(),
            x[2].to_string(),
            x[3].to_string(),
            x_est[0].to_string(),
            x_est[1].to_string(),
            x_est[2].to_string(),
            x_est[3].to_string(),
        ])?;
        wtr.flush()?;

        // x[2]の絶対値がpi/2を超えればエラー
        if x[2].abs() > std::f64::consts::PI / 2.0 {
            println!("x[2] is over pi/2");
            break;
        }
    }

    Ok(())
}

const M1: f64 = 150e-3;
const R: f64 = 50e-3;
const M2: f64 = 2.3 - 2.0 * M1 + 2.0;
const L: f64 = 0.2474; // 重心までの距離
const J1: f64 = M1 * R * R;
const J2: f64 = 0.2;
const G: f64 = 9.81;
const KT: f64 = 0.15; // m2006
const D: f64 = (M1 + M2 + J1 / R * R) * (M2 * L * L + J2) - M2 * M2 * L * L;
// 逐次実行版
fn dynamics(x: &na::Vector4<f64>, u: f64) -> na::Vector4<f64> {
    A * x + B * u
}
