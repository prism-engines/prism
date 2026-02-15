#![allow(unused_variables)]

use pyo3::prelude::*;
use pyo3::types::PyModuleMethods;

mod individual;
mod dynamical;
mod pairwise;
mod matrix;
mod embedding;
mod information;
mod topology;
mod network;

/// Register a submodule and add it to sys.modules so `from pkg.sub import fn` works.
fn register_submodule(
    parent: &Bound<'_, PyModule>,
    name: &str,
    register_fn: fn(&Bound<'_, PyModule>) -> PyResult<()>,
) -> PyResult<()> {
    let py = parent.py();
    let sub = PyModule::new(py, name)?;
    register_fn(&sub)?;
    parent.add_submodule(&sub)?;

    // Critical: register in sys.modules so `from manifold_rs.X import Y` works
    let full_name = format!("manifold_rs.{}", name);
    py.import("sys")?
        .getattr("modules")?
        .set_item(full_name, &sub)?;

    Ok(())
}

/// Rust-accelerated primitives for the Manifold pipeline.
/// Drop-in replacements for manifold.primitives — same signatures, same outputs.
#[pymodule]
fn manifold_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_submodule(m, "individual", individual::register)?;
    register_submodule(m, "dynamical", dynamical::register)?;
    register_submodule(m, "pairwise", pairwise::register)?;
    register_submodule(m, "matrix", matrix::register)?;
    register_submodule(m, "embedding", embedding::register)?;
    register_submodule(m, "information", information::register)?;
    register_submodule(m, "topology", topology::register)?;
    register_submodule(m, "network", network::register)?;
    Ok(())
}
