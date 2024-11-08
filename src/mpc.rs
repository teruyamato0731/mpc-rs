#[macro_export]
macro_rules! create_f_matrix {
    ($a:expr, $n:expr, $s:expr) => {{
        let mut f = nalgebra::SMatrix::<f64, { $s * $n }, $s>::zeros();
        for i in 0..$n {
            f.fixed_view_mut::<$s, $s>($s * i, 0)
                .copy_from(&$a.pow((i + 1) as u32));
        }
        f
    }};
}

#[macro_export]
macro_rules! create_g_matrix {
    ($a:expr, $b:expr, $n:expr, $s:expr) => {{
        let mut g = nalgebra::SMatrix::<f64, { $s * $n }, $n>::zeros();
        for i in 0..$n {
            for j in 0..=i {
                g.fixed_view_mut::<$s, 1>($s * i, j)
                    .copy_from(&($a.pow((i - j) as u32) * B));
            }
        }
        g
    }};
}

#[macro_export]
macro_rules! create_q_matrix {
    ($c:expr, $n:expr, $s:expr) => {{
        let mut q = nalgebra::SMatrix::<f64, { $s * $n }, { $s * $n }>::zeros();
        for i in 0..$n {
            q.fixed_view_mut::<$s, $s>($s * i, $s * i).copy_from(&$c);
        }
        q
    }};
}
