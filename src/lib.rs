use serde::{Deserialize, Serialize};
use tokio::{sync::Semaphore, task::JoinSet, time::{Duration, sleep}};
use anyhow::{Result, anyhow};
use std::{fmt::Display, sync::Arc};

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

pub struct BatchItem {
    pub index: usize,
    pub question: String,
    pub result: anyhow::Result<String>,
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

pub async fn send_with_retry(
    client: &reqwest::Client,
    url: &str,
    key: &str, 
    body: &serde_json::Value
) -> Result<reqwest::Response> {
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

async fn fetch_result(
    client: &reqwest::Client,
    url: &str,
    key: &str, 
    body: &serde_json::Value
) -> Result<String> {
    let resp = send_with_retry(&client, &url, &key, &body).await?;
    let completion: ChatCompletion = resp.json().await?;
    let answer = completion.choices
        .first()
        .and_then(|c| c.message.as_ref())
        .map(|m| m.content.clone())
        .unwrap_or_else(|| "(No response)".into());
    Ok(answer)
}

pub async fn run_batch(
    client: &reqwest::Client,
    url: &str,
    key: &str,
    model: &str,
    questions: Vec<String>,
    mut on_result: impl FnMut(BatchItem),
) -> Result<()> {
    let sem = Arc::new(Semaphore::new(3));
    let mut set = JoinSet::new();

    for (i, q) in questions.into_iter().enumerate() {
        let client = client.clone();
        let url = url.to_string();
        let key = key.to_string();
        let model = model.to_string();
        let sem = sem.clone();

        set.spawn(async move {
            let body = serde_json::json!({
                "model": model,
                "messages": [{"role": "user", "content":q}],
            });

            let result: Result<String> = async {
                let _permit = sem.acquire().await?;
                fetch_result(&client, &url, &key, &body).await
            }.await;

            BatchItem {
                index: i,
                question: q,
                result,
            }
        });
    }

    while let Some(joined) = set.join_next().await {
        match joined {
            Ok(item) => on_result(item),
            Err(e) if e.is_cancelled() => {
                eprintln!("The batch task was cancelled: {e}");
            },
            Err(e) => {
                eprintln!("The batch task panicked or failed: {e}");
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_a_split_line() {
        let mut buff = Vec::new();
        let mut received = Vec::new();

        buff.extend_from_slice(
            br#"data: {"id":"1","object":"chat.completion.chunk","created":0,"model":"test","choices":[{"index":0,"delta":{"content":"hi"#
            );
        let done = parse_stream(&mut buff, |c| received.push(c));
        assert!(!done);
        assert_eq!(received.len(), 0);

        buff.extend_from_slice(b"\"}}]}\n\n");
        let done = parse_stream(&mut buff, |c| received.push(c));
        assert!(!done);
        assert_eq!(received.len(), 1);

        let content = received[0].choices[0]
            .delta.as_ref().unwrap()
            .content.as_deref().unwrap();
        assert_eq!(content, "hi");
    }

    #[test]
    fn handles_a_merged_line() {
 let mut buff = Vec::new();
        let mut received = Vec::new();
        buff.extend_from_slice(
            br#"data: {"id":"1","object":"chat.completion.chunk","created":0,"model":"test","choices":[{"index":0,"delta":{"content":"hi"}}]}

            data: {"id":"1","object":"chat.completion.chu"#);
        let done = parse_stream(&mut buff, |c| received.push(c));
        assert!(!done);
        assert_eq!(received.len(), 1);

        buff.extend_from_slice(
            br#"nk","created":0,"model":"test","choices":[{"index":0,"delta":{"content":"hi"}}]}

            "#);
        let done = parse_stream(&mut buff, |c| received.push(c));
        assert!(!done);
        assert_eq!(received.len(), 2);

        let content = received[0].choices[0]
            .delta.as_ref().unwrap()
            .content.as_deref().unwrap();
        assert_eq!(content, "hi");

        let content = received[1].choices[0]
            .delta.as_ref().unwrap()
            .content.as_deref().unwrap();
        assert_eq!(content, "hi");
    }

    #[test]
    fn handles_heartbeat() {
        let mut received = Vec::new();
        let mut buff = Vec::new();

        buff.extend_from_slice(
            br#": heartbeat

            data: {"id":"1","object":"chat.completion.chunk","created":0,"model":"test","choices":[{"index":0,"delta":{"content":"hi"}}]}

            "#
            );
        let done = parse_stream(&mut buff, |c| received.push(c));
        assert!(!done);
        assert_eq!(received.len(), 1);

        let content = received[0].choices[0]
            .delta.as_ref().unwrap()
            .content.as_deref().unwrap();
        assert_eq!(content, "hi");
    }

    #[test]
    fn handles_done() {
        let mut received = Vec::new();
        let mut buff = Vec::new();

        buff.extend_from_slice(
            br#"data: {"id":"1","object":"chat.completion.chunk","created":0,"model":"test","choices":[{"index":0,"delta":{"content":"hi"}}]}

            data: [DONE]

            "#
        );
        let done = parse_stream(&mut buff, |c| received.push(c));
        assert!(done);
        assert_eq!(received.len(), 1);

        let content = received[0].choices[0]
            .delta.as_ref().unwrap()
            .content.as_deref().unwrap();
        assert_eq!(content, "hi");
    }

    #[test]
    fn handles_utf8_char_split() {
        let mut buff = Vec::new();
        let mut received = Vec::new();

        buff.extend_from_slice(
            br#"data: {"id":"1","object":"chat.completion.chunk","created":0,"model":"test","choices":[{"index":0,"delta":{"content":""#
        );
        buff.extend_from_slice(&[0xE4]);           
        let done = parse_stream(&mut buff, |c| received.push(c));
        assert!(!done);
        assert_eq!(received.len(), 0);             

        buff.extend_from_slice(&[0xBD, 0xA0, 0xE5, 0xA5, 0xBD]);
        buff.extend_from_slice(br#""}}]}"#);
        buff.extend_from_slice(b"\n\n");

        let done = parse_stream(&mut buff, |c| received.push(c));
        assert!(!done);
        assert_eq!(received.len(), 1);

        let content = received[0].choices[0]
            .delta.as_ref().unwrap()
            .content.as_deref().unwrap();
        assert_eq!(content, "你好");
    }
}
