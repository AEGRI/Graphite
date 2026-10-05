use std::io;

#[path = "../runtime/mod.rs"]
mod runtime;

mod shell;

fn main() {
    if let Err(error) = run() {
        eprintln!("Graphite error: {error}");
    }
}

fn run() -> io::Result<()> {
    let mut model = runtime::model::Model::load()?;

    let mut input = shell::input::InputBox::new();
    let mut terminal = shell::input::InputBox::start()?;

    loop {
        match input.read(&mut terminal)? {
            Some(text) => {
                let text = text.trim();

                if text.is_empty() {
                    continue;
                }

                input.add_message(text);

                match model.generate(text) {
                    Ok(response) => {
                        input.add_graphite_message(&response);
                    }
                    Err(error) => {
                        input.add_graphite_message(&format!("Graphite error: {error}"));
                    }
                }
            }
            None => break,
        }
    }

    shell::input::InputBox::stop(terminal)?;
    Ok(())
}
