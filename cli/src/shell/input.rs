use std::{
    fs,
    io::{self, Write, stdout},
    path::{Path, PathBuf},
};

use arboard::Clipboard;
use crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
        KeyModifiers, MouseButton, MouseEventKind,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};
use serde::Deserialize;

const MAX_INPUT_LINES: usize = 5;

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

#[derive(Debug, Deserialize, Clone)]
struct Theme {
    #[allow(dead_code)]
    name: String,
    colors: ThemeColors,
    usage: ThemeUsage,
}

#[derive(Debug, Deserialize, Clone)]
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

#[derive(Debug, Deserialize, Clone)]
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
                format!("invalid Graphite theme '{}': {error}", theme_path.display()),
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
    write!(stdout, "\x1b]11;rgb:{red:02x}/{green:02x}/{blue:02x}\x07")?;
    stdout.flush()
}

fn reset_terminal_background() -> io::Result<()> {
    let mut stdout = stdout();
    write!(stdout, "\x1b]111\x07")?;
    stdout.flush()
}

#[derive(Debug, Clone, Copy)]
struct Selection {
    start: usize,
    end: usize,
}

impl Selection {
    fn new(a: usize, b: usize) -> Self {
        Self {
            start: a.min(b),
            end: a.max(b),
        }
    }

    fn is_empty(self) -> bool {
        self.start == self.end
    }
}

#[derive(Debug, Clone, Copy)]
enum MouseSelection {
    Input { anchor: usize },
    History { message_index: usize, anchor: usize },
}

#[derive(Debug, Clone)]
struct WrappedLine {
    text: String,
    start: usize,
    end: usize,
}

#[derive(Debug, Clone, Copy)]
struct MessageLayout {
    message_index: usize,
    top: u16,
    height: u16,
    width: u16,
    x: u16,
    text_top: u16,
    text_lines: usize,
    separator_width: u16,
    right_aligned: bool,
}

pub struct InputBox {
    pub text: String,
    messages: Vec<String>,
    cursor: usize,
    input_selection_anchor: Option<usize>,
    history_selection: Option<(usize, Selection)>,
    mouse_selection: Option<MouseSelection>,
    chat_scroll: u16,
    follow_bottom: bool,
    dirty: bool,
    theme: Theme,
}

impl InputBox {
    pub fn new() -> Self {
        let theme = Theme::load().unwrap_or_else(|error| {
            eprintln!("Graphite theme error: {error}");
            fallback_theme()
        });

        Self {
            text: String::new(),
            messages: Vec::new(),
            cursor: 0,
            input_selection_anchor: None,
            history_selection: None,
            mouse_selection: None,
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
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;

        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;
        terminal.clear()?;
        terminal.show_cursor()?;

        Ok(terminal)
    }

    pub fn stop(mut terminal: Terminal<CrosstermBackend<std::io::Stdout>>) -> io::Result<()> {
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
        self.cursor = 0;
        self.input_selection_anchor = None;
        self.history_selection = None;
        self.mouse_selection = None;
        self.follow_bottom = true;
        self.chat_scroll = 0;
        self.dirty = true;
    }

    fn terminal_layout(&self, area: Rect) -> [Rect; 3] {
        let height = area.height;

        let header_height = if height >= 24 {
            9
        } else if height >= 18 {
            7
        } else if height >= 12 {
            6
        } else {
            4
        }
        .min(height.saturating_sub(3));

        let input_lines = self.input_lines(area).min(MAX_INPUT_LINES);
        let desired_input_height = input_lines.saturating_add(2) as u16;
        let max_input_height = height.saturating_sub(header_height.saturating_add(1));
        let input_height = desired_input_height.min(max_input_height).max(1);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(header_height),
                Constraint::Min(1),
                Constraint::Length(input_height),
            ])
            .split(area);

        [chunks[0], chunks[1], chunks[2]]
    }

    fn input_wrap_width(&self, area: Rect) -> usize {
        area.width.saturating_sub(5).max(1) as usize
    }

    fn input_lines(&self, area: Rect) -> usize {
        let width = self.input_wrap_width(area);
        self.text.chars().count().div_ceil(width).max(1)
    }

    fn input_scroll_offset(&self, area: Rect) -> usize {
        let total_lines = self.input_lines(area);
        let visible_lines = total_lines.min(MAX_INPUT_LINES);
        let width = self.input_wrap_width(area);
        let cursor_line = self.cursor / width;

        if total_lines <= visible_lines {
            return 0;
        }

        cursor_line
            .saturating_sub(visible_lines.saturating_sub(1))
            .min(total_lines.saturating_sub(visible_lines))
    }

    fn wrapped_input(&self, area: Rect) -> Vec<WrappedLine> {
        let width = self.input_wrap_width(area);
        let chars: Vec<char> = self.text.chars().collect();

        if chars.is_empty() {
            return vec![WrappedLine {
                text: String::new(),
                start: 0,
                end: 0,
            }];
        }

        chars
            .chunks(width)
            .enumerate()
            .map(|(line, chunk)| {
                let start = line * width;
                let end = start + chunk.len();
                WrappedLine {
                    text: chunk.iter().collect(),
                    start,
                    end,
                }
            })
            .collect()
    }

    fn move_cursor_vertical(&mut self, direction: i32, area: Rect, selecting: bool) {
        let width = self.input_wrap_width(area);
        let current_line = self.cursor / width;
        let current_column = self.cursor % width;
        let line_count = self.input_lines(area);

        let target_line = if direction < 0 {
            current_line.saturating_sub(1)
        } else {
            (current_line + 1).min(line_count.saturating_sub(1))
        };

        let target_start = target_line * width;
        let target_end = ((target_line + 1) * width).min(self.text.chars().count());
        let target = (target_start + current_column).min(target_end);

        self.move_cursor(target, selecting);
    }

    fn clear_selection(&mut self) {
        self.input_selection_anchor = None;
        self.history_selection = None;
    }

    fn input_selection(&self) -> Option<Selection> {
        self.input_selection_anchor
            .map(|anchor| Selection::new(anchor, self.cursor))
            .filter(|selection| !selection.is_empty())
    }

    fn selected_input_text(&self) -> Option<String> {
        let selection = self.input_selection()?;

        Some(
            self.text
                .chars()
                .skip(selection.start)
                .take(selection.end - selection.start)
                .collect(),
        )
    }

    fn selected_history_text(&self) -> Option<String> {
        let (message_index, selection) = self.history_selection?;
        let message = self.messages.get(message_index)?;

        Some(
            message
                .chars()
                .skip(selection.start)
                .take(selection.end - selection.start)
                .collect(),
        )
    }

    fn copy_selection(&mut self) -> io::Result<bool> {
        let text = if let Some(text) = self.selected_input_text() {
            text
        } else if let Some(text) = self.selected_history_text() {
            text
        } else {
            return Ok(false);
        };

        let mut clipboard = Clipboard::new()
            .map_err(|error| io::Error::other(format!("could not access clipboard: {error}")))?;

        clipboard
            .set_text(text)
            .map_err(|error| io::Error::other(format!("could not copy text: {error}")))?;

        Ok(true)
    }

    fn cut_input_selection(&mut self) -> io::Result<bool> {
        let Some(selection) = self.input_selection() else {
            return Ok(false);
        };

        let start = char_to_byte_index(&self.text, selection.start);
        let end = char_to_byte_index(&self.text, selection.end);
        let selected = self.text[start..end].to_string();

        let mut clipboard = Clipboard::new()
            .map_err(|error| io::Error::other(format!("could not access clipboard: {error}")))?;

        clipboard
            .set_text(selected)
            .map_err(|error| io::Error::other(format!("could not cut text: {error}")))?;

        self.text.replace_range(start..end, "");
        self.cursor = selection.start;
        self.input_selection_anchor = None;
        self.mouse_selection = None;
        self.dirty = true;

        Ok(true)
    }

    fn paste_clipboard(&mut self) -> io::Result<()> {
        let mut clipboard = Clipboard::new()
            .map_err(|error| io::Error::other(format!("could not access clipboard: {error}")))?;

        let text = clipboard
            .get_text()
            .map_err(|error| io::Error::other(format!("could not read clipboard: {error}")))?;

        if text.is_empty() {
            return Ok(());
        }

        self.delete_selected_input();

        let byte_index = char_to_byte_index(&self.text, self.cursor);
        self.text.insert_str(byte_index, &text);
        self.cursor += text.chars().count();

        self.clear_selection();
        self.dirty = true;

        Ok(())
    }

    fn delete_selected_input(&mut self) -> bool {
        let Some(selection) = self.input_selection() else {
            return false;
        };

        let start = char_to_byte_index(&self.text, selection.start);
        let end = char_to_byte_index(&self.text, selection.end);

        self.text.replace_range(start..end, "");
        self.cursor = selection.start;
        self.input_selection_anchor = None;

        true
    }

    fn insert_character(&mut self, character: char) {
        self.delete_selected_input();

        let byte_index = char_to_byte_index(&self.text, self.cursor);
        self.text.insert(byte_index, character);
        self.cursor += 1;

        self.clear_selection();
        self.dirty = true;
    }

    fn backspace(&mut self) {
        if self.delete_selected_input() {
            self.dirty = true;
            return;
        }

        if self.cursor == 0 {
            return;
        }

        let start = char_to_byte_index(&self.text, self.cursor - 1);
        let end = char_to_byte_index(&self.text, self.cursor);

        self.text.replace_range(start..end, "");
        self.cursor -= 1;
        self.dirty = true;
    }

    fn delete(&mut self) {
        if self.delete_selected_input() {
            self.dirty = true;
            return;
        }

        let char_count = self.text.chars().count();

        if self.cursor >= char_count {
            return;
        }

        let start = char_to_byte_index(&self.text, self.cursor);
        let end = char_to_byte_index(&self.text, self.cursor + 1);

        self.text.replace_range(start..end, "");
        self.dirty = true;
    }

    fn move_cursor(&mut self, position: usize, selecting: bool) {
        let max = self.text.chars().count();
        let position = position.min(max);

        if selecting {
            if self.input_selection_anchor.is_none() {
                self.input_selection_anchor = Some(self.cursor);
            }
        } else {
            self.input_selection_anchor = None;
        }

        self.cursor = position;
        self.dirty = true;
    }

    fn move_cursor_left(&mut self, selecting: bool) {
        if self.cursor == 0 {
            return;
        }

        self.move_cursor(self.cursor - 1, selecting);
    }

    fn move_cursor_right(&mut self, selecting: bool) {
        let max = self.text.chars().count();

        if self.cursor >= max {
            return;
        }

        self.move_cursor(self.cursor + 1, selecting);
    }

    fn move_word_left(&mut self, selecting: bool) {
        if self.cursor == 0 {
            return;
        }

        let chars: Vec<char> = self.text.chars().collect();
        let mut position = self.cursor;

        while position > 0 && chars[position - 1].is_whitespace() {
            position -= 1;
        }

        while position > 0 && !chars[position - 1].is_whitespace() {
            position -= 1;
        }

        self.move_cursor(position, selecting);
    }

    fn move_word_right(&mut self, selecting: bool) {
        let chars: Vec<char> = self.text.chars().collect();
        let mut position = self.cursor;

        while position < chars.len() && chars[position].is_whitespace() {
            position += 1;
        }

        while position < chars.len() && !chars[position].is_whitespace() {
            position += 1;
        }

        self.move_cursor(position, selecting);
    }

    fn select_all_input(&mut self) {
        self.input_selection_anchor = Some(0);
        self.cursor = self.text.chars().count();
        self.history_selection = None;
        self.mouse_selection = None;
        self.dirty = true;
    }

    fn message_width(&self, chat_area: Rect) -> u16 {
        let width = chat_area.width;

        if width <= 4 {
            return width;
        }

        ((width as u32 * 62) / 100)
            .max(12)
            .min(width.saturating_sub(2) as u32) as u16
    }

    fn wrap_message(&self, message: &str, width: usize) -> Vec<WrappedLine> {
        let width = width.max(1);
        let chars: Vec<char> = message.chars().collect();

        if chars.is_empty() {
            return vec![WrappedLine {
                text: String::new(),
                start: 0,
                end: 0,
            }];
        }

        let mut lines = Vec::new();
        let mut line_start = 0usize;
        let mut position = 0usize;

        while line_start < chars.len() {
            if chars[line_start] == '\n' {
                lines.push(WrappedLine {
                    text: String::new(),
                    start: line_start,
                    end: line_start,
                });
                line_start += 1;
                position = line_start;
                continue;
            }

            let mut line_end = line_start;
            let mut last_space = None;

            while position < chars.len() && position < line_start + width {
                if chars[position] == '\n' {
                    break;
                }

                if chars[position].is_whitespace() {
                    last_space = Some(position);
                }

                position += 1;
                line_end = position;
            }

            if position >= chars.len() || chars.get(position) == Some(&'\n') {
                let end = if line_end > line_start && chars[line_end - 1].is_whitespace() {
                    line_end - 1
                } else {
                    line_end
                };

                lines.push(WrappedLine {
                    text: chars[line_start..end].iter().collect(),
                    start: line_start,
                    end,
                });

                if position < chars.len() && chars[position] == '\n' {
                    line_start = position + 1;
                    position = line_start;
                } else {
                    line_start = chars.len();
                }

                continue;
            }

            if let Some(space) = last_space.filter(|space| *space > line_start) {
                let end = space;
                lines.push(WrappedLine {
                    text: chars[line_start..end].iter().collect(),
                    start: line_start,
                    end,
                });

                line_start = space + 1;
                while line_start < chars.len()
                    && chars[line_start].is_whitespace()
                    && chars[line_start] != '\n'
                {
                    line_start += 1;
                }
                position = line_start;
            } else {
                let end = line_start + width;
                lines.push(WrappedLine {
                    text: chars[line_start..end].iter().collect(),
                    start: line_start,
                    end,
                });

                line_start = end;
                position = line_start;
            }
        }

        if lines.is_empty() {
            lines.push(WrappedLine {
                text: String::new(),
                start: 0,
                end: 0,
            });
        }

        lines
    }

    fn message_layouts(&self, chat_area: Rect) -> Vec<MessageLayout> {
        let message_width = self.message_width(chat_area) as usize;
        let mut layouts = Vec::with_capacity(self.messages.len());
        let mut top = 0u16;

        for (message_index, message) in self.messages.iter().enumerate() {
            let wrapped = self.wrap_message(message, message_width);
            let text_lines = wrapped.len();
            let separator_width = wrapped
                .iter()
                .map(|line| line.text.chars().count())
                .max()
                .unwrap_or(0)
                .min(message_width) as u16;

            let width = message_width as u16;
            let height = (text_lines + 3).min(u16::MAX as usize) as u16;
            let right_aligned = true;
            let x = if right_aligned {
                chat_area
                    .x
                    .saturating_add(chat_area.width.saturating_sub(width))
            } else {
                chat_area.x
            };

            layouts.push(MessageLayout {
                message_index,
                top,
                height,
                width,
                x,
                text_top: top + 1,
                text_lines,
                separator_width,
                right_aligned,
            });

            top = top.saturating_add(height);
        }

        layouts
    }

    fn chat_line_count(&self, chat_area: Rect) -> u16 {
        self.message_layouts(chat_area)
            .last()
            .map(|layout| layout.top.saturating_add(layout.height))
            .unwrap_or(0)
    }

    fn max_chat_scroll(&self, chat_area: Rect) -> u16 {
        self.chat_line_count(chat_area)
            .saturating_sub(chat_area.height)
    }

    // chat_scroll is the distance from the bottom of the history.
    // 0 means the newest messages are visible; increasing it moves upward.
    fn chat_scroll_offset(&self, chat_area: Rect) -> u16 {
        let max_scroll = self.max_chat_scroll(chat_area);

        if self.follow_bottom {
            max_scroll
        } else {
            max_scroll.saturating_sub(self.chat_scroll.min(max_scroll))
        }
    }

    fn scroll_up(&mut self, amount: u16, chat_area: Rect) {
        let max_scroll = self.max_chat_scroll(chat_area);

        if max_scroll == 0 {
            self.follow_bottom = true;
            self.chat_scroll = 0;
            self.dirty = true;
            return;
        }

        // Start from the current distance from the bottom. If we're following
        // the newest messages, that's zero. Never recreate or discard history;
        // this only changes which part of the existing history is visible.
        self.chat_scroll = if self.follow_bottom {
            amount.min(max_scroll)
        } else {
            self.chat_scroll.saturating_add(amount).min(max_scroll)
        };

        self.follow_bottom = false;
        self.dirty = true;
    }

    fn scroll_down(&mut self, amount: u16) {
        self.chat_scroll = self.chat_scroll.saturating_sub(amount);

        if self.chat_scroll == 0 {
            self.follow_bottom = true;
        } else {
            self.follow_bottom = false;
        }

        self.dirty = true;
    }

    fn scroll_to_top(&mut self, chat_area: Rect) {
        let max_scroll = self.max_chat_scroll(chat_area);
        self.chat_scroll = max_scroll;
        self.follow_bottom = false;
        self.dirty = true;
    }

    fn scroll_to_bottom(&mut self) {
        self.chat_scroll = 0;
        self.follow_bottom = true;
        self.dirty = true;
    }

    fn input_index_from_mouse(&self, column: u16, row: u16, input_area: Rect) -> usize {
        let width = self.input_wrap_width(input_area);
        let visible_line = row
            .saturating_sub(input_area.y.saturating_add(1))
            .min(MAX_INPUT_LINES.saturating_sub(1) as u16) as usize;
        let line = self.input_scroll_offset(input_area) + visible_line;
        let start_x = input_area.x + 3;

        let column = if column <= start_x {
            0
        } else {
            column.saturating_sub(start_x) as usize
        };

        let position = line * width + column.min(width);
        position.min(self.text.chars().count())
    }

    fn history_index_from_mouse(
        &self,
        column: u16,
        row: u16,
        chat_area: Rect,
    ) -> Option<(usize, usize)> {
        if row < chat_area.y || row >= chat_area.y.saturating_add(chat_area.height) {
            return None;
        }

        let layouts = self.message_layouts(chat_area);
        let scroll = self.chat_scroll_offset(chat_area);

        let absolute_row = row.saturating_sub(chat_area.y).saturating_add(scroll);

        for layout in layouts {
            if absolute_row < layout.top || absolute_row >= layout.top.saturating_add(layout.height)
            {
                continue;
            }

            let relative_row = absolute_row.saturating_sub(layout.top);

            if relative_row == 0 {
                return None;
            }

            if relative_row >= 1 && relative_row <= layout.text_lines as u16 {
                let message = self.messages.get(layout.message_index)?;
                let wrapped = self.wrap_message(message, layout.width as usize);
                let line = wrapped.get(relative_row as usize - 1)?;

                let relative_column = if layout.right_aligned {
                    let right_edge = layout.x.saturating_add(line.text.chars().count() as u16);

                    if column >= right_edge {
                        line.text.chars().count()
                    } else if column <= layout.x {
                        0
                    } else {
                        column.saturating_sub(layout.x) as usize
                    }
                } else {
                    if column <= layout.x {
                        0
                    } else {
                        column
                            .saturating_sub(layout.x)
                            .min(line.text.chars().count() as u16) as usize
                    }
                };

                return Some((
                    layout.message_index,
                    (line.start + relative_column).min(line.end),
                ));
            }

            return None;
        }

        None
    }

    fn begin_mouse_selection(&mut self, column: u16, row: u16, chat_area: Rect, input_area: Rect) {
        if row >= input_area.y && row < input_area.y.saturating_add(input_area.height) {
            let position = self.input_index_from_mouse(column, row, input_area);
            self.cursor = position;
            self.input_selection_anchor = Some(position);
            self.history_selection = None;
            self.mouse_selection = Some(MouseSelection::Input { anchor: position });
            self.dirty = true;
            return;
        }

        if let Some((message_index, position)) =
            self.history_index_from_mouse(column, row, chat_area)
        {
            self.history_selection = Some((message_index, Selection::new(position, position)));
            self.input_selection_anchor = None;
            self.mouse_selection = Some(MouseSelection::History {
                message_index,
                anchor: position,
            });
            self.dirty = true;
        }
    }

    fn update_mouse_selection(&mut self, column: u16, row: u16, chat_area: Rect, input_area: Rect) {
        let Some(selection) = self.mouse_selection else {
            return;
        };

        match selection {
            MouseSelection::Input { anchor } => {
                if row >= input_area.y && row < input_area.y.saturating_add(input_area.height) {
                    let position = self.input_index_from_mouse(column, row, input_area);
                    self.cursor = position;
                    self.input_selection_anchor = Some(anchor);
                    self.history_selection = None;
                    self.dirty = true;
                }
            }
            MouseSelection::History {
                message_index,
                anchor,
            } => {
                if let Some((index, position)) =
                    self.history_index_from_mouse(column, row, chat_area)
                {
                    if index == message_index {
                        self.history_selection =
                            Some((message_index, Selection::new(anchor, position)));
                        self.input_selection_anchor = None;
                        self.dirty = true;
                    }
                }
            }
        }
    }

    fn draw_logo(&self, frame: &mut ratatui::Frame, area: Rect) {
        let lines: Vec<Line> = LOGO
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| Line::styled(line, Style::default().fg(self.theme.usage_color("logo"))))
            .collect();

        if lines.is_empty() || area.width == 0 || area.height == 0 {
            return;
        }

        let logo_height = lines.len().min(area.height as usize) as u16;
        let logo_area = Rect {
            x: area.x,
            y: area.y,
            width: area.width,
            height: logo_height,
        };

        frame.render_widget(Paragraph::new(lines).alignment(Alignment::Left), logo_area);
    }

    fn draw_message(&self, frame: &mut ratatui::Frame, chat_area: Rect, layout: MessageLayout) {
        let message = match self.messages.get(layout.message_index) {
            Some(message) => message,
            None => return,
        };

        let wrapped = self.wrap_message(message, layout.width as usize);
        let selection = self
            .history_selection
            .filter(|(index, _)| *index == layout.message_index)
            .map(|(_, selection)| selection);

        let text_color = self.theme.usage_color("input");
        let you_color = self.theme.usage_color("you");
        let separator_color = self.theme.usage_color("separator");
        let selection_background = self.theme.usage_color("selection_background");
        let selection_foreground = self.theme.usage_color("selection_foreground");

        let scroll = self.chat_scroll_offset(chat_area);
        let message_y = chat_area
            .y
            .saturating_add(layout.top)
            .saturating_sub(scroll);

        // The chat is the background layer. Everything is explicitly clipped to
        // chat_area so it can never draw over the header or input foreground.
        let chat_bottom = chat_area.y.saturating_add(chat_area.height);
        let message_bottom = message_y.saturating_add(layout.height);

        if message_y >= chat_bottom || message_bottom <= chat_area.y {
            return;
        }

        let visible_top = chat_area.y.max(message_y);
        let visible_bottom = chat_bottom.min(message_bottom);

        let message_area = Rect {
            x: layout.x,
            y: message_y,
            width: layout.width,
            height: layout.height,
        };

        // Header / label.
        if message_y >= visible_top && message_y < visible_bottom {
            frame.render_widget(
                Paragraph::new(Line::styled(
                    "You",
                    Style::default().fg(you_color).add_modifier(Modifier::BOLD),
                ))
                .alignment(if layout.right_aligned {
                    Alignment::Right
                } else {
                    Alignment::Left
                }),
                Rect {
                    x: message_area.x,
                    y: message_y,
                    width: message_area.width,
                    height: 1,
                },
            );
        }

        // Render only the message rows that actually intersect the chat area.
        // This is the important clipping step: even when a message is scrolled
        // upward, its lines cannot be painted into the header.
        let text_start_row = visible_top.saturating_sub(message_y).saturating_sub(1) as usize;
        let text_end_row = visible_bottom
            .saturating_sub(message_y)
            .saturating_sub(1)
            .min(wrapped.len() as u16) as usize;

        if text_start_row < text_end_row && text_start_row < wrapped.len() {
            let visible_wrapped = &wrapped[text_start_row..text_end_row];
            let text_y = message_y
                .saturating_add(1)
                .saturating_add(text_start_row as u16);

            let lines = visible_wrapped
                .iter()
                .map(|wrapped_line| {
                    let mut spans = Vec::with_capacity(wrapped_line.text.chars().count());

                    for (offset, character) in wrapped_line.text.chars().enumerate() {
                        let absolute_index = wrapped_line.start + offset;
                        let mut style = Style::default().fg(text_color);

                        if let Some(selection) = selection {
                            if absolute_index >= selection.start && absolute_index < selection.end {
                                style = style.fg(selection_foreground).bg(selection_background);
                            }
                        }

                        spans.push(Span::styled(character.to_string(), style));
                    }

                    Line::from(spans)
                })
                .collect::<Vec<_>>();

            frame.render_widget(
                Paragraph::new(lines).alignment(if layout.right_aligned {
                    Alignment::Right
                } else {
                    Alignment::Left
                }),
                Rect {
                    x: message_area.x,
                    y: text_y,
                    width: message_area.width,
                    height: (text_end_row - text_start_row) as u16,
                },
            );
        }

        // Separator.
        let separator_y = message_y.saturating_add(1 + wrapped.len() as u16);

        if separator_y >= chat_area.y && separator_y < chat_bottom {
            let separator_width = layout.separator_width.max(1);
            let separator_x = if layout.right_aligned {
                message_area
                    .x
                    .saturating_add(message_area.width.saturating_sub(separator_width))
            } else {
                message_area.x
            };

            frame.render_widget(
                Paragraph::new("─".repeat(separator_width as usize))
                    .style(Style::default().fg(separator_color)),
                Rect {
                    x: separator_x,
                    y: separator_y,
                    width: separator_width.min(chat_area.width),
                    height: 1,
                },
            );
        }
    }

    fn draw(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    ) -> io::Result<()> {
        terminal.draw(|frame| {
            let area = frame.area();
            let layout = self.terminal_layout(area);

            let chat_area = layout[1];
            let layouts = self.message_layouts(chat_area);

            // Chat is rendered behind the fixed header and input regions.
            // Message rendering clips itself to chat_area, so scrolled text
            // remains in history without painting across those boundaries.
            for message_layout in layouts {
                self.draw_message(frame, chat_area, message_layout);
            }

            self.draw_logo(frame, layout[0]);

            let divider_area = Rect {
                x: layout[0].x,
                y: layout[0]
                    .y
                    .saturating_add(layout[0].height.saturating_sub(1)),
                width: layout[0].width,
                height: 1,
            };

            frame.render_widget(
                Paragraph::new("─".repeat(divider_area.width as usize))
                    .style(Style::default().fg(self.theme.usage_color("separator"))),
                divider_area,
            );

            let input_area = layout[2];
            let selection = self.input_selection();
            let wrapped = self.wrapped_input(input_area);
            let visible_lines = wrapped.len().min(MAX_INPUT_LINES);
            let scroll_offset = self.input_scroll_offset(input_area);
            let visible_end = (scroll_offset + visible_lines).min(wrapped.len());
            let visible_wrapped = &wrapped[scroll_offset..visible_end];
            let mut lines = Vec::with_capacity(visible_wrapped.len());

            for (line_index, wrapped_line) in visible_wrapped.iter().enumerate() {
                let line_index = line_index;
                let mut spans = Vec::new();

                spans.push(Span::styled(
                    if line_index == 0 { "❯  " } else { "   " },
                    Style::default()
                        .fg(self.theme.usage_color("prompt"))
                        .add_modifier(Modifier::BOLD),
                ));

                if self.text.is_empty() && line_index == 0 {
                    spans.push(Span::styled(
                        "Ask anything...",
                        Style::default().fg(self.theme.usage_color("placeholder")),
                    ));
                } else {
                    for (offset, character) in wrapped_line.text.chars().enumerate() {
                        let index = wrapped_line.start + offset;
                        let mut style = Style::default().fg(self.theme.usage_color("input"));

                        if let Some(selection) = selection {
                            if index >= selection.start && index < selection.end {
                                style = style
                                    .fg(self.theme.usage_color("selection_foreground"))
                                    .bg(self.theme.usage_color("selection_background"));
                            }
                        }

                        spans.push(Span::styled(character.to_string(), style));
                    }
                }

                lines.push(Line::from(spans));
            }

            frame.render_widget(
                Paragraph::new(lines).block(
                    Block::default()
                        .borders(Borders::TOP | Borders::BOTTOM)
                        .border_style(Style::default().fg(self.theme.usage_color("border"))),
                ),
                input_area,
            );

            let width = self.input_wrap_width(input_area);
            let cursor_line = self.cursor / width;
            let cursor_column = self.cursor % width;
            let cursor_visible_line = cursor_line.saturating_sub(scroll_offset);
            let cursor_x = input_area
                .x
                .saturating_add(3)
                .saturating_add(cursor_column as u16)
                .min(
                    input_area
                        .x
                        .saturating_add(input_area.width.saturating_sub(1)),
                );
            let cursor_y = input_area
                .y
                .saturating_add(1)
                .saturating_add(cursor_visible_line as u16)
                .min(
                    input_area
                        .y
                        .saturating_add(input_area.height.saturating_sub(1)),
                );

            frame.set_cursor_position((cursor_x, cursor_y));
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

                    let selecting = modifiers.contains(KeyModifiers::SHIFT);

                    match code {
                        KeyCode::Enter => {
                            if !self.text.is_empty() {
                                return Ok(Some(self.text.clone()));
                            }
                        }

                        // Copy selected input OR selected history.
                        KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                            if self.copy_selection()? {
                                self.clear_selection();
                                self.dirty = true;
                            } else {
                                return Ok(None);
                            }
                        }

                        // Cut selected text from the Ask Anything box only.
                        KeyCode::Char('x') if modifiers.contains(KeyModifiers::CONTROL) => {
                            self.cut_input_selection()?;
                        }

                        KeyCode::Char('v') if modifiers.contains(KeyModifiers::CONTROL) => {
                            self.paste_clipboard()?;
                        }

                        KeyCode::Char('a') if modifiers.contains(KeyModifiers::CONTROL) => {
                            self.select_all_input();
                        }

                        KeyCode::Left => {
                            if modifiers.contains(KeyModifiers::CONTROL) {
                                self.move_word_left(selecting);
                            } else {
                                self.move_cursor_left(selecting);
                            }
                        }

                        KeyCode::Right => {
                            if modifiers.contains(KeyModifiers::CONTROL) {
                                self.move_word_right(selecting);
                            } else {
                                self.move_cursor_right(selecting);
                            }
                        }

                        KeyCode::Home => {
                            self.move_cursor(0, selecting);
                        }

                        KeyCode::End => {
                            self.move_cursor(self.text.chars().count(), selecting);
                        }

                        KeyCode::Backspace => {
                            self.backspace();
                        }

                        KeyCode::Delete => {
                            self.delete();
                        }

                        KeyCode::Char(character) if !modifiers.contains(KeyModifiers::CONTROL) => {
                            self.insert_character(character);
                        }

                        KeyCode::PageUp => {
                            let size = terminal.size()?;
                            let area = Rect::new(0, 0, size.width, size.height);
                            let layout = self.terminal_layout(area);
                            self.scroll_up(2, layout[1]);
                        }

                        KeyCode::PageDown => {
                            let size = terminal.size()?;
                            let area = Rect::new(0, 0, size.width, size.height);
                            let layout = self.terminal_layout(area);
                            self.scroll_down(2);
                        }

                        KeyCode::Up => {
                            let size = terminal.size()?;
                            let area = Rect::new(0, 0, size.width, size.height);

                            if self.input_lines(area) > 1 {
                                self.move_cursor_vertical(-1, area, selecting);
                            } else {
                                let layout = self.terminal_layout(area);
                                self.scroll_up(1, layout[1]);
                            }
                        }

                        KeyCode::Down => {
                            let size = terminal.size()?;
                            let area = Rect::new(0, 0, size.width, size.height);

                            if self.input_lines(area) > 1 {
                                self.move_cursor_vertical(1, area, selecting);
                            } else {
                                self.scroll_down(1);
                            }
                        }

                        _ => {}
                    }
                }

                Event::Mouse(mouse) => {
                    let size = terminal.size()?;
                    let area = Rect::new(0, 0, size.width, size.height);
                    let layout = self.terminal_layout(area);

                    match mouse.kind {
                        MouseEventKind::ScrollUp => {
                            // Mouse wheel: move one terminal row at a time for smooth scrolling.
                            self.scroll_up(1, layout[1]);
                        }

                        MouseEventKind::ScrollDown => {
                            self.scroll_down(1);
                        }

                        MouseEventKind::Down(MouseButton::Left) => {
                            self.begin_mouse_selection(
                                mouse.column,
                                mouse.row,
                                layout[1],
                                layout[2],
                            );
                        }

                        MouseEventKind::Drag(MouseButton::Left) => {
                            self.update_mouse_selection(
                                mouse.column,
                                mouse.row,
                                layout[1],
                                layout[2],
                            );
                        }

                        MouseEventKind::Up(MouseButton::Left) => {
                            self.mouse_selection = None;

                            let empty_input_selection = self.input_selection().is_none();
                            let empty_history_selection = self
                                .history_selection
                                .map(|(_, selection)| selection.is_empty())
                                .unwrap_or(true);

                            if empty_input_selection && empty_history_selection {
                                self.history_selection = None;
                                self.dirty = true;
                            }
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

fn char_to_byte_index(text: &str, character_index: usize) -> usize {
    text.char_indices()
        .nth(character_index)
        .map(|(index, _)| index)
        .unwrap_or(text.len())
}

fn fallback_theme() -> Theme {
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
}

// TODO: Add alternate compact Graphite ASCII/logo assets once the responsive
// layout reaches the minimum useful size for the current block-art logo.
// TODO: Add model-role messages (Graphite/User) once model output is wired in.
// TODO: Add a dedicated responsive input editor if multiline user input is desired.
