use std::io::{self, Write};
use tokio::time::{Duration, timeout};
use anyhow::{Context, Result};
use mini_ai_client::{Message, parse_stream, send_with_retry};
use clap::Parser;

#[derive(Parser)]
#[command(name = "mini-ai-client")]
struct Args {
    #[arg(short)]
    model: String,

    #[arg(long, default_value = "https://api.deepseek.com")]
    base_url: String,

    #[arg(long, default_value_t = 5)]
    max_rounds: usize,

    #[arg(long, default_value_t = 15)]
    timeout_secs: u64,
}
// openai api
#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let api_key = std::env::var("DEEPSEEK_API_KEY")
        .context("Please set envrionmental variable DEEPSEEK_API_KEY")?;
    let client = reqwest::Client::new();
    let mut history = Vec::<Message>::new();
    
    loop {
        print!("> ");
        io::stdout().flush()?;

        let mut line = String::new();
        let n = io::stdin().read_line(&mut line)?;
        if n == 0 {
            break;
        }
        
        let input = line.trim();

        match input {
            "exit" => break,
            "" => continue,
            _ => (),
        }

        history.push(Message { role: "user".into(), content: input.to_string()});

        let body = serde_json::json!({
            "model": args.model,
            "messages": &history,
            "stream": true,
            "stream_options": {"include_usage": true}
        });

        let url = format!("{}/chat/completions", args.base_url);
        let mut resp = send_with_retry(&client, &url, &api_key, &body).await?;

        let mut buff: Vec<u8> = Vec::new();
        let mut reply = String::new();

        let mut completed: bool = false;
        loop {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    println!("Interpreted");
                    history.pop();
                    break;
                }

                result = timeout(Duration::from_secs(args.timeout_secs), resp.chunk()) => {
                    match result {
                        Err(_elapsed) => {
                            eprintln!("Waiting for next token timed out");
                            history.pop();
                            break;
                        }
                        Ok(Err(e)) => {
                            eprintln!("{e}");
                            history.pop();
                            break;
                        }
                        Ok(Ok(None)) => {
                            completed = true;
                            break;
                        }
                        Ok(Ok(Some(bytes))) => {
                            buff.extend_from_slice(&bytes);
                            let done = parse_stream(&mut buff, |event| {
                                if let Some(usage) = &event.usage {
                                    println!("\n{usage}");
                                    return;
                                }
                                let Some(choice) = event.choices.first() else { return; };
                                let Some(delta) = choice.delta.as_ref() else { return; };
                                let Some(content) = delta.content.as_deref() else { return; };
                                print!("{content}");
                                let _ = io::stdout().flush();
                                reply.push_str(content);
                            });
                            if done {
                                completed = true;
                                break;
                            }
                        }

                    }
                }

            }
        }

        let max_msg = args.max_rounds * 2;
        if completed {
            history.push(Message { role: "assistant".into(), content: reply });
            if history.len() > max_msg {
                history.drain(0..history.len() - max_msg);
            }
        }

        println!();

    }

    Ok(())
}
