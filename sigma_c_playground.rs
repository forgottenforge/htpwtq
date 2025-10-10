// Rust Playground: Sigma_C live demo (no logging)
// Single-file, no external crates.
// License
// Copyright (c) 2025 ForgottenForge.xyz

// Dual Licensed under (see LICENSE):

// Creative Commons Attribution 4.0 International (CC BY 4.0)
// Elastic License 2.0 (ELv2)
// Commercial licensing available. Contact: nfo@forgottenforge.xyz

use std::io::{self, Write};

// I(eps) = F * C * sqrt(P) * (1 - eps)^k
fn info_function(f: f64, c: f64, p: f64, eps: f64, k: f64) -> f64 {
    if eps >= 1.0 { return 0.0; }
    f * c * p.sqrt() * (1.0 - eps).powf(k.max(0.0))
}

// Moving average smoothing; window must be odd (1 = no smoothing)
fn smooth(data: &Vec<f64>, window: usize) -> Vec<f64> {
    if window <= 1 || window % 2 == 0 || data.len() < window {
        return data.clone();
    }
    let half = window / 2;
    let mut out = vec![0.0; data.len()];
    for i in 0..data.len() {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(data.len());
        let mut sum = 0.0;
        let mut cnt = 0usize;
        for j in start..end {
            sum += data[j];
            cnt += 1;
        }
        out[i] = if cnt > 0 { sum / cnt as f64 } else { data[i] };
    }
    out
}

// Central finite difference (forward/backward at edges)
fn finite_diff(eps: &Vec<f64>, i: usize, vals: &Vec<f64>) -> f64 {
    if eps.len() < 2 { return 0.0; }
    if i == 0 {
        let de = (eps[1] - eps[0]).abs().max(1e-12);
        return (vals[1] - vals[0]) / de;
    }
    if i == eps.len() - 1 {
        let de = (eps[eps.len()-1] - eps[eps.len()-2]).abs().max(1e-12);
        return (vals[eps.len()-1] - vals[eps.len()-2]) / de;
    }
    let de = (eps[i+1] - eps[i-1]).abs().max(1e-12);
    (vals[i+1] - vals[i-1]) / de
}

// Find sigma_c as eps with maximal |dI/deps|
fn find_sigma_c(eps: &Vec<f64>, vals: &Vec<f64>) -> (f64, usize, f64) {
    let mut best_i = 0usize;
    let mut best_grad = -1.0f64;
    for i in 0..eps.len() {
        let g = finite_diff(eps, i, vals).abs();
        if g > best_grad {
            best_grad = g;
            best_i = i;
        }
    }
    (eps[best_i], best_i, best_grad)
}

fn recompute_sigma_c(
    f: f64, c: f64, p: f64, k: f64,
    n: usize, eps_min: f64, eps_max: f64,
    smoothing: bool, window: usize
) -> (Vec<f64>, Vec<f64>, f64, usize, f64) {
    // grid
    let mut eps = vec![0.0; n];
    for i in 0..n {
        eps[i] = eps_min + (eps_max - eps_min) * (i as f64) / ((n - 1) as f64);
    }
    // values
    let mut vals = vec![0.0; n];
    for (i, e) in eps.iter().enumerate() {
        vals[i] = info_function(f, c, p, *e, k);
    }
    let vals_proc = if smoothing { smooth(&vals, window) } else { vals.clone() };
    let (sigma_c, idx, grad) = find_sigma_c(&eps, &vals_proc);
    (eps, vals_proc, sigma_c, idx, grad)
}

fn print_help() {
    println!("Commands:");
    println!("  set k <val>         # change visibility exponent (e.g., 2.0)");
    println!("  set scale <s>       # scale F*C*sqrt(P) by s (e.g., 0.8)");
    println!("  set n <int>         # grid points (>= 5)");
    println!("  set range <min> <max>  # eps range, e.g., 0.0 0.3");
    println!("  set smooth <0|1>    # smoothing off/on");
    println!("  set window <odd>    # smoothing window (odd, >=1)");
    println!("  show                # recompute and print sigma_c and table");
    println!("  params              # print current parameters");
    println!("  help                # this help");
    println!("  quit                # exit");
}

fn print_params(f: f64, c: f64, p: f64, k: f64, scale: f64, n: usize, eps_min: f64, eps_max: f64, smoothing: bool, window: usize) {
    println!("Parameters:");
    println!("  F={:.3}, C={:.3}, P={:.3}, k={:.3}, scale={:.3}", f, c, p, k, scale);
    println!("  grid n={}, eps in [{:.3}, {:.3}]", n, eps_min, eps_max);
    println!("  smoothing={}, window={}", smoothing, window);
}

fn main() {
    // Base parameters (you can change F,C,P too if you like)
    let f0 = 0.95;
    let c0 = 0.60;
    let p0 = 0.90;

    let mut k: f64 = 2.0;          // visibility exponent
    let mut scale: f64 = 1.0;      // scales F*C*sqrt(P)
    let mut n: usize = 21;         // grid points
    let mut eps_min: f64 = 0.0;
    let mut eps_max: f64 = 0.30;
    let mut smoothing: bool = true;
    let mut window: usize = 3;     // odd

    println!("sigma_c live demo (Rust Playground) – no logging");
    print_help();
    print_params(f0, c0, p0, k, scale, n, eps_min, eps_max, smoothing, window);

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut line = String::new();
        if io::stdin().read_line(&mut line).is_err() {
            break;
        }
        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        if parts.is_empty() { continue; }

        match parts[0] {
            "set" if parts.len() >= 3 && parts[1] == "k" => {
                if let Ok(val) = parts[2].parse::<f64>() {
                    k = val;
                    println!("k = {}", k);
                } else { println!("invalid k"); }
            }
            "set" if parts.len() >= 3 && parts[1] == "scale" => {
                if let Ok(val) = parts[2].parse::<f64>() {
                    scale = val.max(0.0);
                    println!("scale = {}", scale);
                } else { println!("invalid scale"); }
            }
            "set" if parts.len() >= 3 && parts[1] == "n" => {
                if let Ok(val) = parts[2].parse::<usize>() {
                    if val >= 5 { n = val; println!("n = {}", n); } else { println!("n must be >= 5"); }
                } else { println!("invalid n"); }
            }
            "set" if parts.len() >= 4 && parts[1] == "range" => {
                if let (Ok(a), Ok(b)) = (parts[2].parse::<f64>(), parts[3].parse::<f64>()) {
                    if a < b && a >= 0.0 && b <= 1.0 {
                        eps_min = a; eps_max = b;
                        println!("range = [{:.3}, {:.3}]", eps_min, eps_max);
                    } else { println!("range must satisfy 0.0 <= min < max <= 1.0"); }
                } else { println!("invalid range"); }
            }
            "set" if parts.len() >= 3 && parts[1] == "smooth" => {
                match parts[2] {
                    "0" => { smoothing = false; println!("smoothing = off"); }
                    "1" => { smoothing = true;  println!("smoothing = on"); }
                    _ => println!("use 0 or 1"),
                }
            }
            "set" if parts.len() >= 3 && parts[1] == "window" => {
                if let Ok(val) = parts[2].parse::<usize>() {
                    if val >= 1 && val % 2 == 1 {
                        window = val; println!("window = {}", window);
                    } else { println!("window must be odd and >= 1"); }
                } else { println!("invalid window"); }
            }
            "params" => {
                print_params(f0, c0, p0, k, scale, n, eps_min, eps_max, smoothing, window);
            }
            "show" => {
                let f = f0 * scale;
                let c = c0 * scale;
                let p = p0 * scale;
                let (eps, vals, sigma_c, idx, grad) =
                    recompute_sigma_c(f, c, p, k, n, eps_min, eps_max, smoothing, window);

                println!("sigma_c ≈ {:.6} (idx={}, |dI/dε|≈{:.6})", sigma_c, idx, grad);
                println!("ε\tI(ε){}", if smoothing { "\tI_smooth(ε)" } else { "" });

                // Für Übersicht nur bis 60 Zeilen ausgeben
                let limit = eps.len().min(60);
                for i in 0..limit {
                    if smoothing {
                        println!("{:.3}\t{:.6}\t{:.6}", eps[i], info_function(f, c, p, eps[i], k), vals[i]);
                    } else {
                        println!("{:.3}\t{:.6}", eps[i], vals[i]);
                    }
                }
                if eps.len() > limit {
                    println!("... ({} rows total)", eps.len());
                }
            }
            "help" => print_help(),
            "quit" => break,
            _ => println!("unknown command (type 'help')"),
        }
    }
}
