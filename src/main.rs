use anyhow::{Context, Result};
use serde::Deserialize;
use std::fmt::Display;

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
    message: Msg,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Msg {
    role: String,
    content: String,
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

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("DEEPSEEK_API_KEY")
        .context("Please set envrionmental variable DEEPSEEK_API_KEY")?;
    let base_url = std::env::var("AI_BASE_URL")
        .unwrap_or_else(|_| "https://api.deepseek.com".to_string());
    let client = reqwest::Client::new();

    let body = serde_json::json!({
        "model": "deepseek-chat",
        "messages":[
            { "role": "user", "content": "一句话介绍你自己" }
        ]
    });

    let resp = client
        .post(format!("{base_url}/chat/completions"))
        .bearer_auth(&api_key)
        .json(&body)
        .send()
        .await?;

    let completion = resp.json::<ChatCompletion>().await?;

    match completion.choices.first() {
        Some(choice) => {
            println!("{}", choice.message.content);
        }
        None => {
            println!("No return from the model.")
        }
    }

    if let Some(usage) = completion.usage {
        println!("{}", usage);
    }

    Ok(())
}
