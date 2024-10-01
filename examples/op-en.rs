use optimization_engine::{constraints::*, panoc::*, *};

fn rosenbrock_cost(a: f64, b: f64, u: &[f64]) -> f64 {
    (a - u[0]).powi(2) + b * (u[1] - u[0].powi(2)).powi(2)
}

fn rosenbrock_grad(a: f64, b: f64, u: &[f64], grad: &mut [f64]) {
    grad[0] = 2.0 * u[0] - 2.0 * a - 4.0 * b * u[0] * (-u[0].powi(2) + u[1]);
    grad[1] = b * (-2.0 * u[0].powi(2) + 2.0 * u[1]);
}

fn main() {
    let tolerance = 1e-6;
    let mut a = 1.0;
    let mut b = 100.0;
    let n = 2;
    let lbfgs_memory = 10;
    let max_iters = 100;
    let mut u = [-1.5, 0.9];
    let mut radius = 1.0;

    // the cache is created only ONCE
    let mut panoc_cache = PANOCCache::new(n, tolerance, lbfgs_memory);

    let mut i = 0;
    while i < 100 {
        // update the values of `a`, `b` and `radius`
        b *= 1.01;
        a -= 1e-3;
        radius += 0.001;

        // define the cost function and its gradient
        let df = |u: &[f64], grad: &mut [f64]| -> Result<(), SolverError> {
            if a < 0.0 || b < 0.0 {
                Err(SolverError::Cost)
            } else {
                rosenbrock_grad(a, b, u, grad);
                Ok(())
            }
        };

        let f = |u: &[f64], c: &mut f64| -> Result<(), SolverError> {
            if a < 0.0 || b < 0.0 {
                Err(SolverError::Cost)
            } else {
                *c = rosenbrock_cost(a, b, u);
                Ok(())
            }
        };

        // define the bounds at every iteration
        let bounds = constraints::Ball2::new(None, radius);

        // the problem definition is updated at every iteration
        let problem = Problem::new(&bounds, df, f);

        // updated instance of the solver
        let mut panoc = PANOCOptimizer::new(problem, &mut panoc_cache).with_max_iter(max_iters);

        let status = panoc.solve(&mut u).unwrap();

        i += 1;

        // print useful information
        println!(
            "parameters: (a={:.4}, b={:.4}, r={:.4}), iters = {}",
            a,
            b,
            radius,
            status.iterations()
        );
        println!("u = {:#.6?}", u);
    }
}
