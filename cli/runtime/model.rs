use std::{
    fs,
    io,
    path::{Path, PathBuf},
};

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Config {
    model: ModelConfig,
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    name: String,
    directory: String,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub name: String,
    pub directory: PathBuf,
    pub root: PathBuf,
    pub model_config: PathBuf,
    pub inference_config: PathBuf,
    pub tokenizer: PathBuf,
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

        Ok(Self {
            name: config.model.name,
            directory,
            root,
            model_config,
            inference_config,
            tokenizer,
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
}