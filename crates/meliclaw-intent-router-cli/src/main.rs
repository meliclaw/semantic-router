//! Binary `meliclaw-intent-embed` — LocalAI Capa 1 embeddings probe.

use clap::Parser;
use meliclaw_intent_router_cli::{format_report, run, Cli};

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let output = cli.output;
    match run(cli).await {
        Ok(report) => {
            println!("{}", format_report(&report, output));
        }
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(e.exit_code());
        }
    }
}
