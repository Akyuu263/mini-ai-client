use serde::{Deserialize, Serialize};
use tokio::time::{Duration, sleep};
use anyhow::{Result, anyhow};
use std::fmt::Display;

#[derive(Debug, Deserialize)]
pub struct ChatCompletion {
    id: String,
    object: String,
    created: u32,
    model: String,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    index: u32,
    message: Option<Message>,
    pub delta: Option<Delta>,
    finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct Delta {
    role: Option<String>,
    pub content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

impl Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "prompt: {}, completion: {}, total: {}", self.prompt_tokens, self.completion_tokens, self.total_tokens)
    }
}

pub fn parse_stream(buff: &mut Vec<u8>, mut process: impl FnMut(ChatCompletion)) -> bool {
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

pub async fn send_with_retry(client: &reqwest::Client, url: &str, key: &str, body: &serde_json::Value) -> Result<reqwest::Response> {
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

