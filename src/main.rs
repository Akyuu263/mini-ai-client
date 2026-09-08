use anyhow::{Context, Result};

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

    let text = resp.text().await?;
    println!("{text}");

    Ok(())
}
