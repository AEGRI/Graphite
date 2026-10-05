use std::{
    env, fs,
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
};

use serde::Deserialize;
use serde_json::json;

const CONFIG_PATH: &str = "cli/config/config.toml";

#[derive(Debug, Deserialize)]
struct Config {
    model: ModelConfig,
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    name: String,
    directory: PathBuf,
}

#[derive(Debug, Deserialize)]
struct InferenceConfig {
    #[serde(default)]
    model: InferenceModelConfig,
}

#[derive(Debug, Default, Deserialize)]
struct InferenceModelConfig {
    checkpoint: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct BridgeResponse {
    text: Option<String>,
    error: Option<String>,
}

pub struct Model {
    process: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl Model {
    pub fn load() -> io::Result<Self> {
        let project_root = find_project_root()?;
        let config_path = project_root.join(CONFIG_PATH);
        let config = read_config(&config_path)?;

        let directory = resolve_path(&project_root, &config.model.directory);
        let root = directory.join(&config.model.name);
        let model_config = root.join("config/model.json");
        let inference_config = root.join("config/inference.json");
        let tokenizer = root.join("tokenizer/files/tokenizer.json");
        let bridge = root.join("inference/bridge.py");

        require_directory(&root, "configured Graphite model")?;
        require_file(&model_config, "model configuration")?;
        require_file(&inference_config, "inference configuration")?;
        require_file(&tokenizer, "tokenizer")?;
        require_file(&bridge, "inference bridge")?;

        let inference = read_inference_config(&inference_config)?;
        let checkpoint = resolve_checkpoint(&root, &inference)?;
        require_file(&checkpoint, "Graphite checkpoint")?;

        let python = env::var("GRAPHITE_PYTHON").unwrap_or_else(|_| "python".to_string());

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
                "Graphite inference bridge did not provide stdin",
            )
        })?;

        let stdout = process.stdout.take().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "Graphite inference bridge did not provide stdout",
            )
        })?;

        Ok(Self {
            process,
            stdin: BufWriter::new(stdin),
            stdout: BufReader::new(stdout),
        })
    }

    pub fn generate(&mut self, prompt: &str) -> io::Result<String> {
        let prompt = prompt.trim();

        if prompt.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "prompt cannot be empty",
            ));
        }

        if let Some(status) = self.process.try_wait()? {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                format!("Graphite inference bridge is no longer running: {status}"),
            ));
        }

        let request = json!({ "prompt": prompt });

        serde_json::to_writer(&mut self.stdin, &request).map_err(|error| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                format!("could not send request to Graphite inference bridge: {error}"),
            )
        })?;

        self.stdin.write_all(b"\n").map_err(|error| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                format!("could not finish Graphite inference request: {error}"),
            )
        })?;

        self.stdin.flush().map_err(|error| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                format!("could not flush Graphite inference request: {error}"),
            )
        })?;

        let mut response_line = String::new();
        let bytes_read = self.stdout.read_line(&mut response_line)?;

        if bytes_read == 0 {
            let status = self.process.try_wait()?.map(|status| status.to_string());
            let message = match status {
                Some(status) => format!("Graphite inference bridge exited unexpectedly: {status}"),
                None => "Graphite inference bridge closed its output unexpectedly".to_string(),
            };

            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, message));
        }

        let response: BridgeResponse =
            serde_json::from_str(response_line.trim()).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid response from Graphite inference bridge: {error}"),
                )
            })?;

        if let Some(error) = response.error {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Graphite inference error: {error}"),
            ));
        }

        response.text.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Graphite inference response did not contain 'text'",
            )
        })
    }
}

fn read_config(path: &Path) -> io::Result<Config> {
    let text = fs::read_to_string(path).map_err(|error| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "could not read Graphite config '{}': {error}",
                path.display()
            ),
        )
    })?;

    toml::from_str(&text).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid Graphite config '{}': {error}", path.display()),
        )
    })
}

fn read_inference_config(path: &Path) -> io::Result<InferenceConfig> {
    let text = fs::read_to_string(path).map_err(|error| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "could not read inference config '{}': {error}",
                path.display()
            ),
        )
    })?;

    serde_json::from_str(&text).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid inference config '{}': {error}", path.display()),
        )
    })
}

fn resolve_checkpoint(model_root: &Path, config: &InferenceConfig) -> io::Result<PathBuf> {
    if let Some(checkpoint) = &config.model.checkpoint {
        return Ok(resolve_path(model_root, checkpoint));
    }

    if let Ok(checkpoint) = env::var("GRAPHITE_CHECKPOINT") {
        if !checkpoint.trim().is_empty() {
            return Ok(resolve_path(model_root, Path::new(&checkpoint)));
        }
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "no Graphite checkpoint is configured; set model.checkpoint in inference.json or GRAPHITE_CHECKPOINT",
    ))
}

fn resolve_path(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn require_directory(path: &Path, description: &str) -> io::Result<()> {
    if !path.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("{description} does not exist: '{}'", path.display()),
        ));
    }
    Ok(())
}

fn require_file(path: &Path, description: &str) -> io::Result<()> {
    if !path.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("{description} does not exist: '{}'", path.display()),
        ));
    }
    Ok(())
}

fn find_project_root() -> io::Result<PathBuf> {
    let mut current = env::current_dir()?;

    loop {
        if current.join(CONFIG_PATH).is_file() {
            return Ok(current);
        }

        if !current.pop() {
            break;
        }
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!("could not find Graphite project root containing '{CONFIG_PATH}'"),
    ))
}

impl Drop for Model {
    fn drop(&mut self) {
        let _ = self.process.kill();
        let _ = self.process.wait();
    }
}
