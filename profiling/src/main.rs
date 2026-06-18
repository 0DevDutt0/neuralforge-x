//! # neuralforge_profile
//!
//! A small CLI that drives the NeuralForge-X kernels under a **sustained load**,
//! so an external profiler — `cargo flamegraph` / `samply` on the CPU, Nsight on
//! the GPU side of the stack — has a long, steady signal to sample. It profiles
//! the *real* `neuralforge_core` kernels (not a re-implementation), so a captured
//! flamegraph reflects exactly the shipped hot paths.
//!
//! ```text
//! neuralforge_profile <dot|batch|topk|all> [--seconds N] [--dim D]
//!                     [--corpus N] [--queries Q] [--k K]
//! ```
//!
//! Each workload runs in a tight loop until the time budget elapses, then prints
//! the iteration count and throughput. A running checksum is fed through
//! [`std::hint::black_box`] so the optimizer cannot elide the work being measured.

use std::hint::black_box;
use std::time::{Duration, Instant};

use neuralforge_core::{batch_similarity, cosine_similarity, top_k_search, MatrixView, Metric};

/// Parsed command-line configuration.
struct Config {
    workload: String,
    seconds: f64,
    dim: usize,
    corpus: usize,
    queries: usize,
    k: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            workload: "all".to_string(),
            seconds: 10.0,
            dim: 768,
            corpus: 100_000,
            queries: 32,
            k: 10,
        }
    }
}

const USAGE: &str = "\
neuralforge_profile — sustained kernel workloads for profiling

USAGE:
    neuralforge_profile <WORKLOAD> [OPTIONS]

WORKLOADS:
    dot      pairwise cosine similarity (light; mostly SIMD reduction)
    batch    batch_similarity over a Q x N corpus (rayon + SIMD)
    topk     top_k_search over an N x D corpus (rayon + bounded heaps)
    all      run dot, batch, then topk, each for seconds/3 (default)

OPTIONS:
    --seconds <F>   time budget per workload run        [default: 10]
    --dim <D>       vector dimensionality               [default: 768]
    --corpus <N>    number of corpus vectors            [default: 100000]
    --queries <Q>   number of query vectors (batch)     [default: 32]
    --k <K>         neighbours to retrieve (topk)       [default: 10]
    -h, --help      print this help";

fn main() {
    let config = match parse_args() {
        Ok(Some(config)) => config,
        Ok(None) => {
            println!("{USAGE}");
            return;
        }
        Err(msg) => {
            eprintln!("error: {msg}\n\n{USAGE}");
            std::process::exit(2);
        }
    };

    let corpus = synth(config.corpus, config.dim);
    let view = MatrixView::new(&corpus, config.corpus, config.dim).expect("valid corpus shape");

    match config.workload.as_str() {
        "dot" => run_dot(&config, view),
        "batch" => run_batch(&config, view),
        "topk" => run_topk(&config, view),
        "all" => {
            let each = Config {
                seconds: config.seconds / 3.0,
                ..clone_config(&config)
            };
            run_dot(&each, view);
            run_batch(&each, view);
            run_topk(&each, view);
        }
        other => {
            eprintln!("error: unknown workload '{other}'\n\n{USAGE}");
            std::process::exit(2);
        }
    }
}

/// Runs `body` in a loop until `seconds` elapse, returning `(iterations, elapsed)`.
fn drive(seconds: f64, mut body: impl FnMut()) -> (u64, Duration) {
    let budget = Duration::from_secs_f64(seconds);
    let start = Instant::now();
    let mut iters = 0u64;
    loop {
        body();
        iters += 1;
        if start.elapsed() >= budget {
            break;
        }
    }
    (iters, start.elapsed())
}

fn run_dot(config: &Config, view: MatrixView<'_>) {
    let a = view.row(0);
    let b = view.row(view.rows() / 2);
    let mut acc = 0.0f32;
    let (iters, elapsed) = drive(config.seconds, || {
        acc += cosine_similarity(black_box(a), black_box(b)).unwrap();
    });
    black_box(acc);
    let per_iter_elems = config.dim as f64;
    report("dot", iters, elapsed, iters as f64 * per_iter_elems);
}

fn run_batch(config: &Config, view: MatrixView<'_>) {
    let q_data = synth(config.queries, config.dim);
    let queries = MatrixView::new(&q_data, config.queries, config.dim).expect("valid query shape");
    let mut acc = 0.0f32;
    let (iters, elapsed) = drive(config.seconds, || {
        let out = batch_similarity(black_box(queries), black_box(view), Metric::Cosine).unwrap();
        acc += out[0];
    });
    black_box(acc);
    let elems = iters as f64 * (config.queries * config.corpus) as f64;
    report("batch", iters, elapsed, elems);
}

fn run_topk(config: &Config, view: MatrixView<'_>) {
    let query: Vec<f32> = view.row(0).to_vec();
    let mut acc = 0usize;
    let (iters, elapsed) = drive(config.seconds, || {
        let hits =
            top_k_search(black_box(&query), black_box(view), config.k, Metric::Cosine).unwrap();
        acc += hits.len();
    });
    black_box(acc);
    let elems = iters as f64 * config.corpus as f64;
    report("topk", iters, elapsed, elems);
}

/// Prints a throughput summary line for one workload run.
fn report(name: &str, iters: u64, elapsed: Duration, elements: f64) {
    let secs = elapsed.as_secs_f64();
    let per_iter_ms = secs / iters as f64 * 1e3;
    let gelem_s = elements / secs / 1e9;
    println!(
        "{name:<6} iters={iters:>8}  {secs:6.2}s  {per_iter_ms:9.4} ms/iter  {gelem_s:7.2} Gelem/s"
    );
}

/// Builds a deterministic `rows x cols` row-major corpus with a small LCG (no
/// `rand` dependency, fully reproducible).
fn synth(rows: usize, cols: usize) -> Vec<f32> {
    let mut state = 0x1234_5678u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
    };
    (0..rows * cols).map(|_| next()).collect()
}

fn clone_config(config: &Config) -> Config {
    Config {
        workload: config.workload.clone(),
        seconds: config.seconds,
        dim: config.dim,
        corpus: config.corpus,
        queries: config.queries,
        k: config.k,
    }
}

/// Parses argv. Returns `Ok(None)` when help was requested.
fn parse_args() -> Result<Option<Config>, String> {
    let mut config = Config::default();
    let mut args = std::env::args().skip(1).peekable();
    let mut saw_workload = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(None),
            "--seconds" => config.seconds = parse_next(&mut args, "--seconds")?,
            "--dim" => config.dim = parse_next(&mut args, "--dim")?,
            "--corpus" => config.corpus = parse_next(&mut args, "--corpus")?,
            "--queries" => config.queries = parse_next(&mut args, "--queries")?,
            "--k" => config.k = parse_next(&mut args, "--k")?,
            value if !value.starts_with('-') && !saw_workload => {
                config.workload = value.to_string();
                saw_workload = true;
            }
            other => return Err(format!("unexpected argument '{other}'")),
        }
    }

    if config.dim == 0 || config.corpus == 0 {
        return Err("--dim and --corpus must be positive".to_string());
    }
    if config.k == 0 || config.k > config.corpus {
        return Err("--k must satisfy 1 <= k <= corpus".to_string());
    }
    Ok(Some(config))
}

/// Parses the value following a flag into any `FromStr` type.
fn parse_next<T>(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let raw = args
        .next()
        .ok_or_else(|| format!("{flag} requires a value"))?;
    raw.parse::<T>()
        .map_err(|e| format!("invalid value for {flag}: {e}"))
}
