use std::io::{self, Write};
use tokio::time::{Duration, timeout};
use anyhow::{Context, Result};
use mini_ai_client::{Message, parse_stream, send_with_retry};

const MAX_ROUNDS: usize = 5;
const MAX_MSG: usize = MAX_ROUNDS * 2;

// openai api
#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("DEEPSEEK_API_KEY")
        .context("Please set envrionmental variable DEEPSEEK_API_KEY")?;
    let base_url = std::env::var("AI_BASE_URL")
        .unwrap_or_else(|_| "https://api.deepseek.com".to_string());
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
            "model": "deepseek-chat",
            "messages": &history,
            "stream": true,
            "stream_options": {"include_usage": true}
        });

        let url = format!("{base_url}/chat/completions");
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

                result = timeout(Duration::from_secs(15), resp.chunk()) => {
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

        if completed {
            history.push(Message { role: "assistant".into(), content: reply });
            if history.len() > MAX_MSG {
                history.drain(0..history.len() - MAX_MSG);
            }
        }

        println!();

    }

    Ok(())
}
