//! One-process boundary-layout microbenchmark; merge trace is prepared before timing.
use efficient_bpe_rust::Bounds;
use efficient_bpe_rust::ablation::micro::run_micro;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let mut variant = None;
    let mut length = None;
    let mut positions = 131_072_usize;
    let mut pattern = "random".to_owned();
    let mut bounds = Bounds::Checked;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {arg}"))?;
        match arg.as_str() {
            "--variant" => variant = Some(value),
            "--length" => length = Some(value.parse::<usize>()?),
            "--positions" => positions = value.parse()?,
            "--pattern" => pattern = value,
            "--bounds" => {
                bounds = match value.as_str() {
                    "checked" => Bounds::Checked,
                    "unchecked" => Bounds::Unchecked,
                    _ => return Err("--bounds must be checked or unchecked".into()),
                }
            }
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    let variant = variant.ok_or("--variant is required")?;
    let length = length.ok_or("--length is required")?;
    let result = run_micro(&variant, length, positions, &pattern, bounds)?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
