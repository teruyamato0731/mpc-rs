use optimization_engine::{panoc::*, *};

fn f(u: &[f64], c: &mut f64) -> Result<(), SolverError> {
    *c = u[0].powi(2) + u[1].powi(2);
    Ok(())
}

fn df(u: &[f64], grad: &mut [f64]) -> Result<(), SolverError> {
    grad[0] = 2.0 * u[0];
    grad[1] = 2.0 * u[1];
    Ok(())
}

fn main() {
    let tolerance = 1e-6;
    let n = 2;
    let lbfgs_memory = 10;
    let max_iters = 200;
    let mut u = [0.0, 0.0];
    let radius = 1.0;

    // the cache is created only ONCE
    let mut panoc_cache = PANOCCache::new(n, tolerance, lbfgs_memory);

    // define the bounds at every iteration
    let bounds = constraints::Ball2::new(None, radius);

    // the problem definition is updated at every iteration
    let problem = Problem::new(&bounds, df, f);

    // updated instance of the solver
    let mut panoc = PANOCOptimizer::new(problem, &mut panoc_cache).with_max_iter(max_iters);

    let status = panoc.solve(&mut u).unwrap();

    // print useful information
    println!(
        "parameters: (r={:.4}), iters = {}",
        radius,
        status.iterations()
    );
    println!("u = {:#.6?}", u);
}
