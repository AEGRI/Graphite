use std::{
    fs,
    io::{
        self,
        BufRead,
        BufReader,
        BufWriter,
        Write,
    },
    path::{Path, PathBuf},
    process::{
        Child,
        ChildStdin,
        ChildStdout,
        Command,
        Stdio,
    },
};

use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Debug, Deserialize)]
struct Config {
    model: ModelConfig,
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    name: String,
    directory: String,
}

pub struct Model {
    pub name: String,
    pub directory: PathBuf,
    pub root: PathBuf,
    pub model_config: PathBuf,
    pub inference_config: PathBuf,
    pub tokenizer: PathBuf,
    pub bridge: PathBuf,

    process: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl Model {
    pub fn load() -> io::Result<Self> {
        let config_path = Path::new("cli/config/config.toml");

        let config_text = fs::read_to_string(config_path).map_err(|error| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "could not read Graphite config '{}': {error}",
                    config_path.display()
                ),
            )
        })?;

        let config: Config = toml::from_str(&config_text).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid Graphite config: {error}"),
            )
        })?;

        let directory = PathBuf::from(&config.model.directory);
        let root = directory.join(&config.model.name);

        let model_config = root.join("config/model.json");
        let inference_config = root.join("config/inference.json");
        let tokenizer = root.join("tokenizer/files/tokenizer.json");
        let bridge = root.join("inference/bridge.py");

        if !root.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "configured Graphite model does not exist: '{}'",
                    root.display()
                ),
            ));
        }

        if !model_config.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "model configuration does not exist: '{}'",
                    model_config.display()
                ),
            ));
        }

        if !inference_config.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "inference configuration does not exist: '{}'",
                    inference_config.display()
                ),
            ));
        }

        if !tokenizer.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "tokenizer does not exist: '{}'",
                    tokenizer.display()
                ),
            ));
        }

        if !bridge.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "inference bridge does not exist: '{}'",
                    bridge.display()
                ),
            ));
        }

        let python = std::env::var("GRAPHITE_PYTHON")
            .unwrap_or_else(|_| "python".to_string());

        let mut process = Command::new(&python)
            .arg(&bridge)
            .current_dir(&root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!(
                        "could not start Graphite inference bridge using '{}': {error}",
                        python
                    ),
                )
            })?;

        let stdin = process.stdin.take().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "could not acquire Graphite inference bridge stdin",
            )
        })?;

        let stdout = process.stdout.take().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "could not acquire Graphite inference bridge stdout",
            )
        })?;

        Ok(Self {
            name: config.model.name,
            directory,
            root,
            model_config,
            inference_config,
            tokenizer,
            bridge,
            process,
            stdin: BufWriter::new(stdin),
            stdout: BufReader::new(stdout),
        })
    }

    pub fn generate(&mut self, prompt: &str) -> io::Result<String> {
        if prompt.trim().is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "prompt cannot be empty",
            ));
        }

        let request = json!({
            "prompt": prompt,
        });

        serde_json::to_writer(
            &mut self.stdin,
            &request,
        )
        .map_err(|error| {
            io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "could not send request to Graphite inference bridge: {error}"
                ),
            )
        })?;

        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;

        let mut response_line = String::new();

        let bytes_read = self
            .stdout
            .read_line(&mut response_line)?;

        if bytes_read == 0 {
            let status = self.process.try_wait()?.map(|status| {
                status.to_string()
            });

            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                match status {
                    Some(status) => format!(
                        "Graphite inference bridge exited unexpectedly: {status}"
                    ),
                    None => {
                        "Graphite inference bridge closed its output unexpectedly"
                            .to_string()
                    }
                },
            ));
        }

        let response: Value =
            serde_json::from_str(&response_line).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "invalid response from Graphite inference bridge: {error}"
                    ),
                )
            })?;

        if let Some(error) = response.get("error") {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "Graphite inference error: {}",
                    error.as_str().unwrap_or("unknown error")
                ),
            ));
        }

        response
            .get("text")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Graphite inference response did not contain 'text'",
                )
            })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn model_config(&self) -> &Path {
        &self.model_config
    }

    pub fn inference_config(&self) -> &Path {
        &self.inference_config
    }

    pub fn tokenizer(&self) -> &Path {
        &self.tokenizer
    }

    pub fn bridge(&self) -> &Path {
        &self.bridge
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}