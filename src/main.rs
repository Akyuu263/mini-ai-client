use anyhow::{Context, Result, anyhow};
use tokio::time::{timeout, Duration, sleep};
use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::io::{self, Write};

// openai api
#[derive(Debug, Deserialize)]
struct ChatCompletion {
    id: String,
    object: String,
    created: u32,
    model: String,
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    index: u32,
    message: Option<Message>,
    delta: Option<Delta>,
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct Delta {
    role: Option<String>,
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "prompt: {}, completion: {}, total: {}", self.prompt_tokens, self.completion_tokens, self.total_tokens)
    }
}

fn parse_stream(buff: &mut Vec<u8>, mut process: impl FnMut(ChatCompletion)) -> bool {
    let mut done = false;
    while let Some(pos) = buff.iter().position(|&b| b == b'\n') {
        let line = buff.drain(..=pos).collect::<Vec<_>>();
        let trimmed_line = line.trim_ascii();
        if trimmed_line.is_empty() {
            continue;
        }
        if trimmed_line[0] == b':' {
            continue;
        }
        let Some(payload) = trimmed_line.strip_prefix(b"data:") else {
            continue;
        };

        let payload = payload.trim_ascii();
        if payload == b"[DONE]" {
            done = true;
            break;
        }

        let Ok(parse) = serde_json::from_slice::<ChatCompletion>(payload) else {
            eprintln!("Parsing failed");
            continue;
        };

        process(parse);
    }

    done

}

async fn send_with_retry(client: &reqwest::Client, url: &str, key: &str, body: &serde_json::Value) -> Result<reqwest::Response> {
    for attempt in 0..3 {
        match client.post(url).bearer_auth(key).json(body).send().await {
            Ok(resp) if resp.status().is_success() => {
                return Ok(resp);
            }
            Ok(resp) if resp.status().is_server_error() || resp.status().as_u16() == 429 => {
                eprintln!("Retrying: {}, error code: {}", attempt + 1, resp.status());
                sleep(Duration::from_secs(2u64.pow(attempt))).await;
            }
            Err(e) if e.is_timeout() || e.is_connect() => {
                eprintln!("Connection error, retrying: {}", attempt + 1);
                sleep(Duration::from_secs(2u64.pow(attempt))).await;
            }
            Err(e) => {
                return Err(e.into());
            }
            Ok(resp) => {
                return Err(anyhow!("Request refused (HTTP {})", resp.status()));
            }
        }
    }
    Err(anyhow!("Connection failed after 3 tries"))
}

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
        }

        println!();

    }

    Ok(())
}
