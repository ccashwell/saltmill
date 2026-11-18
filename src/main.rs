use saltmill::cli::Config;
use std::process::ExitCode;

fn main() -> ExitCode {
    let outcome = Config::parse_args().and_then(saltmill::run);
    match outcome {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
