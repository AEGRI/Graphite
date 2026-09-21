use std::{
    fs,
    io::{self, stdout, Write},
    path::{Path, PathBuf},
};

use crossterm::{
    event::{
        self,
        DisableMouseCapture,
        EnableMouseCapture,
        Event,
        KeyCode,
        KeyEvent,
        KeyEventKind,
        KeyModifiers,
        MouseEventKind,
    },
    execute,
    terminal::{
        disable_raw_mode,
        enable_raw_mode,
        EnterAlternateScreen,
        LeaveAlternateScreen,
    },
};

use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};

use serde::Deserialize;

const LOGO: &str = r#"

 ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗██╗████████╗███████╗
██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║██║╚══██╔══╝██╔════╝
██║  ███╗██████╔╝███████║██████╔╝███████║██║   ██║   █████╗  
██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║██║   ██║   ██╔══╝  
╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║██║   ██║   ███████╗
 ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝   ╚═╝   ╚══════╝

"#;

#[derive(Debug, Deserialize)]
struct Config {
    #[serde(rename = "color-scheme")]
    color_scheme: ColorSchemeConfig,
}

#[derive(Debug, Deserialize)]
struct ColorSchemeConfig {
    directory: String,
    name: String,
}

#[derive(Debug, Deserialize)]
struct Theme {
    name: String,
    colors: ThemeColors,
    usage: ThemeUsage,
}

#[derive(Debug, Deserialize)]
struct ThemeColors {
    background: String,
    background_alt: String,
    background_highlight: String,
    foreground: String,
    foreground_alt: String,
    muted: String,

    black: String,
    red: String,
    orange: String,
    yellow: String,
    green: String,
    cyan: String,
    blue: String,
    purple: String,
    magenta: String,
    white: String,

    bright_black: String,
    bright_red: String,
    bright_orange: String,
    bright_yellow: String,
    bright_green: String,
    bright_cyan: String,
    bright_blue: String,
    bright_purple: String,
    bright_magenta: String,
    bright_white: String,
}

#[derive(Debug, Deserialize)]
struct ThemeUsage {
    background: String,
    foreground: String,
    logo: String,
    you: String,
    input: String,
    prompt: String,
    placeholder: String,
    cursor: String,
    separator: String,
    border: String,
    selection_background: String,
    selection_foreground: String,
    success: String,
    warning: String,
    error: String,
    info: String,
    muted: String,
}

impl Theme {
    fn load() -> io::Result<Self> {
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

        let theme_path = PathBuf::from("cli/config")
            .join(&config.color_scheme.directory)
            .join(format!("{}.json", config.color_scheme.name));

        let theme_text = fs::read_to_string(&theme_path).map_err(|error| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "could not read Graphite theme '{}': {error}",
                    theme_path.display()
                ),
            )
        })?;

        serde_json::from_str(&theme_text).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "invalid Graphite theme '{}': {error}",
                    theme_path.display()
                ),
            )
        })
    }

    fn color(&self, name: &str) -> Color {
        let value = match name {
            "background" => &self.colors.background,
            "background_alt" => &self.colors.background_alt,
            "background_highlight" => &self.colors.background_highlight,
            "foreground" => &self.colors.foreground,
            "foreground_alt" => &self.colors.foreground_alt,
            "muted" => &self.colors.muted,

            "black" => &self.colors.black,
            "red" => &self.colors.red,
            "orange" => &self.colors.orange,
            "yellow" => &self.colors.yellow,
            "green" => &self.colors.green,
            "cyan" => &self.colors.cyan,
            "blue" => &self.colors.blue,
            "purple" => &self.colors.purple,
            "magenta" => &self.colors.magenta,
            "white" => &self.colors.white,

            "bright_black" => &self.colors.bright_black,
            "bright_red" => &self.colors.bright_red,
            "bright_orange" => &self.colors.bright_orange,
            "bright_yellow" => &self.colors.bright_yellow,
            "bright_green" => &self.colors.bright_green,
            "bright_cyan" => &self.colors.bright_cyan,
            "bright_blue" => &self.colors.bright_blue,
            "bright_purple" => &self.colors.bright_purple,
            "bright_magenta" => &self.colors.bright_magenta,
            "bright_white" => &self.colors.bright_white,

            _ => return Color::Reset,
        };

        parse_color(value)
    }

    fn usage_color(&self, usage: &str) -> Color {
        let color_name = match usage {
            "background" => &self.usage.background,
            "foreground" => &self.usage.foreground,
            "logo" => &self.usage.logo,
            "you" => &self.usage.you,
            "input" => &self.usage.input,
            "prompt" => &self.usage.prompt,
            "placeholder" => &self.usage.placeholder,
            "cursor" => &self.usage.cursor,
            "separator" => &self.usage.separator,
            "border" => &self.usage.border,
            "selection_background" => &self.usage.selection_background,
            "selection_foreground" => &self.usage.selection_foreground,
            "success" => &self.usage.success,
            "warning" => &self.usage.warning,
            "error" => &self.usage.error,
            "info" => &self.usage.info,
            "muted" => &self.usage.muted,

            _ => return Color::Reset,
        };

        self.color(color_name)
    }
}

fn parse_color(value: &str) -> Color {
    let value = value.trim_start_matches('#');

    if value.len() != 6 {
        return Color::Reset;
    }

    let red = u8::from_str_radix(&value[0..2], 16);
    let green = u8::from_str_radix(&value[2..4], 16);
    let blue = u8::from_str_radix(&value[4..6], 16);

    match (red, green, blue) {
        (Ok(red), Ok(green), Ok(blue)) => Color::Rgb(red, green, blue),
        _ => Color::Reset,
    }
}

fn set_terminal_background(color: Color) -> io::Result<()> {
    let Color::Rgb(red, green, blue) = color else {
        return Ok(());
    };

    let mut stdout = stdout();

    write!(
        stdout,
        "\x1b]11;rgb:{red:02x}/{green:02x}/{blue:02x}\x07"
    )?;

    stdout.flush()
}

fn reset_terminal_background() -> io::Result<()> {
    let mut stdout = stdout();

    write!(stdout, "\x1b]111\x07")?;
    stdout.flush()
}

pub struct InputBox {
    pub text: String,
    messages: Vec<String>,
    chat_scroll: u16,
    follow_bottom: bool,
    dirty: bool,
    theme: Theme,
}

impl InputBox {
    pub fn new() -> Self {
        let theme = Theme::load().unwrap_or_else(|error| {
            eprintln!("Graphite theme error: {error}");

            Theme {
                name: "fallback".to_string(),
                colors: ThemeColors {
                    background: "#000000".to_string(),
                    background_alt: "#000000".to_string(),
                    background_highlight: "#444444".to_string(),
                    foreground: "#FFFFFF".to_string(),
                    foreground_alt: "#CCCCCC".to_string(),
                    muted: "#666666".to_string(),
                    black: "#000000".to_string(),
                    red: "#FF0000".to_string(),
                    orange: "#FF8800".to_string(),
                    yellow: "#FFFF00".to_string(),
                    green: "#00FF00".to_string(),
                    cyan: "#00FFFF".to_string(),
                    blue: "#0000FF".to_string(),
                    purple: "#8800FF".to_string(),
                    magenta: "#FF00FF".to_string(),
                    white: "#FFFFFF".to_string(),
                    bright_black: "#666666".to_string(),
                    bright_red: "#FF0000".to_string(),
                    bright_orange: "#FF8800".to_string(),
                    bright_yellow: "#FFFF00".to_string(),
                    bright_green: "#00FF00".to_string(),
                    bright_cyan: "#00FFFF".to_string(),
                    bright_blue: "#0000FF".to_string(),
                    bright_purple: "#8800FF".to_string(),
                    bright_magenta: "#FF00FF".to_string(),
                    bright_white: "#FFFFFF".to_string(),
                },
                usage: ThemeUsage {
                    background: "background".to_string(),
                    foreground: "foreground".to_string(),
                    logo: "red".to_string(),
                    you: "red".to_string(),
                    input: "foreground".to_string(),
                    prompt: "red".to_string(),
                    placeholder: "muted".to_string(),
                    cursor: "foreground".to_string(),
                    separator: "background_highlight".to_string(),
                    border: "background_highlight".to_string(),
                    selection_background: "background_highlight".to_string(),
                    selection_foreground: "foreground".to_string(),
                    success: "green".to_string(),
                    warning: "yellow".to_string(),
                    error: "red".to_string(),
                    info: "cyan".to_string(),
                    muted: "muted".to_string(),
                },
            }
        });

        Self {
            text: String::new(),
            messages: Vec::new(),
            chat_scroll: 0,
            follow_bottom: true,
            dirty: true,
            theme,
        }
    }

    pub fn start() -> io::Result<Terminal<CrosstermBackend<std::io::Stdout>>> {
        let theme = Theme::load().map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("could not load Graphite theme: {error}"),
            )
        })?;

        set_terminal_background(theme.usage_color("background"))?;

        enable_raw_mode()?;

        let mut stdout = stdout();

        execute!(
            stdout,
            EnterAlternateScreen,
            EnableMouseCapture
        )?;

        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        terminal.clear()?;
        terminal.show_cursor()?;

        Ok(terminal)
    }

    pub fn stop(
        mut terminal: Terminal<CrosstermBackend<std::io::Stdout>>,
    ) -> io::Result<()> {
        disable_raw_mode()?;

        execute!(
            terminal.backend_mut(),
            DisableMouseCapture,
            LeaveAlternateScreen
        )?;

        reset_terminal_background()?;

        terminal.show_cursor()?;

        Ok(())
    }

    pub fn add_message(&mut self, message: &str) {
        self.messages.push(message.to_string());
        self.text.clear();
        self.follow_bottom = true;
        self.chat_scroll = 0;
        self.dirty = true;
    }

    fn message_lines(&self) -> Vec<Line<'_>> {
        let you_color = self.theme.usage_color("you");
        let input_color = self.theme.usage_color("input");

        self.messages
            .iter()
            .flat_map(|message| {
                vec![
                    Line::from(vec![Span::styled(
                        "You",
                        Style::default()
                            .fg(you_color)
                            .add_modifier(Modifier::BOLD),
                    )]),
                    Line::from(Span::styled(
                        format!("  {message}"),
                        Style::default().fg(input_color),
                    )),
                    Line::from(""),
                ]
            })
            .collect()
    }

    fn max_chat_scroll(&self, visible_height: u16) -> u16 {
        let total_lines = self.messages.len() as u16 * 3;
        total_lines.saturating_sub(visible_height)
    }

    fn scroll_up(&mut self, amount: u16, visible_height: u16) {
        let max_scroll = self.max_chat_scroll(visible_height);

        self.chat_scroll = self
            .chat_scroll
            .saturating_add(amount)
            .min(max_scroll);

        self.follow_bottom = self.chat_scroll == max_scroll;
        self.dirty = true;
    }

    fn scroll_down(&mut self, amount: u16) {
        self.chat_scroll = self.chat_scroll.saturating_sub(amount);

        self.follow_bottom = self.chat_scroll == 0;
        self.dirty = true;
    }

    fn draw(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    ) -> io::Result<()> {
        let theme = &self.theme;

        terminal.draw(|frame| {
            let area = frame.area();

            let layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(9),
                    Constraint::Min(1),
                    Constraint::Length(3),
                ])
                .split(area);

            let logo_lines: Vec<Line> = LOGO
                .lines()
                .map(|line| {
                    Line::styled(
                        line,
                        Style::default()
                            .fg(theme.usage_color("logo")),
                    )
                })
                .collect();

            frame.render_widget(
                Paragraph::new(logo_lines),
                layout[0],
            );

            let divider_area = Rect {
                x: layout[0].x,
                y: layout[0].y + layout[0].height - 1,
                width: layout[0].width,
                height: 1,
            };

            frame.render_widget(
                Paragraph::new(
                    "─".repeat(divider_area.width as usize),
                )
                .style(
                    Style::default()
                        .fg(theme.usage_color("separator")),
                ),
                divider_area,
            );

            let visible_height = layout[1].height;
            let max_scroll = self.max_chat_scroll(visible_height);

            let scroll = if self.follow_bottom {
                max_scroll
            } else {
                self.chat_scroll.min(max_scroll)
            };

            frame.render_widget(
                Paragraph::new(self.message_lines())
                    .style(
                        Style::default()
                            .fg(theme.usage_color("foreground")),
                    )
                    .scroll((scroll, 0)),
                layout[1],
            );

            let input_area = layout[2];

            let content = if self.text.is_empty() {
                Line::from(vec![
                    Span::styled(
                        "❯  ",
                        Style::default()
                            .fg(theme.usage_color("prompt"))
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        "Ask Graphite anything...",
                        Style::default()
                            .fg(theme.usage_color("placeholder")),
                    ),
                ])
            } else {
                Line::from(vec![
                    Span::styled(
                        "❯  ",
                        Style::default()
                            .fg(theme.usage_color("prompt"))
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        &self.text,
                        Style::default()
                            .fg(theme.usage_color("input")),
                    ),
                ])
            };

            frame.render_widget(
                Paragraph::new(content).block(
                    Block::default()
                        .borders(
                            Borders::TOP | Borders::BOTTOM,
                        )
                        .border_style(
                            Style::default()
                                .fg(theme.usage_color("border")),
                        ),
                ),
                input_area,
            );

            frame.set_cursor_position((
                input_area.x
                    + 3
                    + self.text.chars().count() as u16,
                input_area.y + 1,
            ));
        })?;

        self.dirty = false;

        Ok(())
    }

    pub fn read(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    ) -> io::Result<Option<String>> {
        self.dirty = true;

        loop {
            if self.dirty {
                self.draw(terminal)?;
            }

            let event = event::read()?;

            match event {
                Event::Key(KeyEvent {
                    code,
                    modifiers,
                    kind,
                    ..
                }) => {
                    if kind != KeyEventKind::Press {
                        continue;
                    }

                    match code {
                        KeyCode::Enter => {
                            return Ok(Some(self.text.clone()));
                        }

                        KeyCode::Char('c')
                            if modifiers.contains(KeyModifiers::CONTROL) =>
                        {
                            return Ok(None);
                        }

                        KeyCode::Backspace => {
                            if self.text.pop().is_some() {
                                self.dirty = true;
                            }
                        }

                        KeyCode::Char(character) => {
                            self.text.push(character);
                            self.dirty = true;
                        }

                        KeyCode::PageUp => {
                            self.scroll_up(8, 10_000);
                        }

                        KeyCode::PageDown => {
                            self.scroll_down(8);
                        }

                        KeyCode::Up => {
                            self.scroll_up(1, 10_000);
                        }

                        KeyCode::Down => {
                            self.scroll_down(1);
                        }

                        _ => {}
                    }
                }

                Event::Mouse(mouse) => {
                    match mouse.kind {
                        MouseEventKind::ScrollUp => {
                            self.scroll_up(3, 10_000);
                        }

                        MouseEventKind::ScrollDown => {
                            self.scroll_down(3);
                        }

                        _ => {}
                    }
                }

                Event::Resize(_, _) => {
                    self.dirty = true;
                }

                _ => {}
            }
        }
    }
}