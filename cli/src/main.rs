use std::io;

mod shell;

fn main() {
    if let Err(error) = run() {
        eprintln!("Graphite error: {error}");
    }
}

fn run() -> io::Result<()> {
    let mut input = shell::input::InputBox::new();
    let mut terminal = shell::input::InputBox::start()?;

    loop {
        match input.read(&mut terminal)? {
            Some(text) => {
                let text = text.trim();

                if text == "/exit" || text == "/quit" {
                    break;
                }

                if !text.is_empty() {
                    input.add_message(text);
                }
            }

            None => break,
        }
    }

    shell::input::InputBox::stop(terminal)?;

    Ok(())
}