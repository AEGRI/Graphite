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
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
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

    fn terminal_layout(area: Rect) -> [Rect; 3] {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(9),
                Constraint::Min(1),
                Constraint::Length(3),
            ])
            .split(area);

        [chunks[0], chunks[1], chunks[2]]
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

    fn message_lines(&self) -> Vec<Line<'static>> {
        let you_color = self.theme.usage_color("you");
        let input_color = self.theme.usage_color("input");
        let selection_background = self.theme.usage_color("selection_background");
        let selection_foreground = self.theme.usage_color("selection_foreground");

        let mut lines = Vec::new();

        for (message_index, message) in self.messages.iter().enumerate() {
            lines.push(Line::from(vec![Span::styled(
                "You",
                Style::default().fg(you_color).add_modifier(Modifier::BOLD),
            )]));

            let selection = self
                .history_selection
                .filter(|(index, _)| *index == message_index)
                .map(|(_, selection)| selection);

            let mut spans = Vec::new();

            for (index, character) in message.chars().enumerate() {
                let mut style = Style::default().fg(input_color);

                if let Some(selection) = selection {
                    if index >= selection.start && index < selection.end {
                        style = style.fg(selection_foreground).bg(selection_background);
                    }
                }

                spans.push(Span::styled(character.to_string(), style));
            }

            lines.push(Line::from(spans));
            lines.push(Line::from(""));
        }

        lines
    }

    fn chat_line_count(&self) -> u16 {
        self.messages.len().saturating_mul(3).min(u16::MAX as usize) as u16
    }

    fn max_chat_scroll(&self, visible_height: u16) -> u16 {
        self.chat_line_count().saturating_sub(visible_height)
    }

    fn scroll_up(&mut self, amount: u16, visible_height: u16) {
        let max_scroll = self.max_chat_scroll(visible_height);

        self.chat_scroll = self.chat_scroll.saturating_add(amount).min(max_scroll);
        self.follow_bottom = self.chat_scroll == 0;

        if max_scroll > 0 && self.chat_scroll < max_scroll {
            self.follow_bottom = false;
        }

        self.dirty = true;
    }

    fn scroll_down(&mut self, amount: u16) {
        self.chat_scroll = self.chat_scroll.saturating_sub(amount);

        if self.chat_scroll == 0 {
            self.follow_bottom = true;
        }

        self.dirty = true;
    }

    fn input_index_from_mouse(&self, column: u16, input_area: Rect) -> usize {
        let start_x = input_area.x + 3;

        if column <= start_x {
            return 0;
        }

        let position = column.saturating_sub(start_x) as usize;

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

        let max_scroll = self.max_chat_scroll(chat_area.height);

        let scroll = if self.follow_bottom {
            max_scroll
        } else {
            self.chat_scroll.min(max_scroll)
        };

        let relative_row = row.saturating_sub(chat_area.y) as usize + scroll as usize;

        let message_index = relative_row / 3;
        let line_index = relative_row % 3;

        // Each message is:
        // 0 = "You"
        // 1 = message
        // 2 = blank
        if line_index != 1 {
            return None;
        }

        let message = self.messages.get(message_index)?;

        let index = (column.saturating_sub(chat_area.x) as usize).min(message.chars().count());

        Some((message_index, index))
    }

    fn begin_mouse_selection(&mut self, column: u16, row: u16, chat_area: Rect, input_area: Rect) {
        if row >= input_area.y && row < input_area.y.saturating_add(input_area.height) {
            let position = self.input_index_from_mouse(column, input_area);

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
                    let position = self.input_index_from_mouse(column, input_area);

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

    fn draw(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    ) -> io::Result<()> {
        terminal.draw(|frame| {
            let area = frame.area();
            let layout = Self::terminal_layout(area);

            let logo_color = self.theme.usage_color("logo");
            let separator_color = self.theme.usage_color("separator");
            let foreground_color = self.theme.usage_color("foreground");
            let prompt_color = self.theme.usage_color("prompt");
            let placeholder_color = self.theme.usage_color("placeholder");
            let input_color = self.theme.usage_color("input");
            let border_color = self.theme.usage_color("border");
            let selection_background = self.theme.usage_color("selection_background");
            let selection_foreground = self.theme.usage_color("selection_foreground");

            let logo_lines: Vec<Line> = LOGO
                .lines()
                .map(|line| Line::styled(line, Style::default().fg(logo_color)))
                .collect();

            frame.render_widget(Paragraph::new(logo_lines), layout[0]);

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
                    .style(Style::default().fg(separator_color)),
                divider_area,
            );

            let chat_area = layout[1];

            let max_scroll = self.max_chat_scroll(chat_area.height);

            let scroll = if self.follow_bottom {
                max_scroll
            } else {
                self.chat_scroll.min(max_scroll)
            };

            frame.render_widget(
                Paragraph::new(self.message_lines())
                    .style(Style::default().fg(foreground_color))
                    .scroll((scroll, 0)),
                chat_area,
            );

            let input_area = layout[2];
            let selection = self.input_selection();

            let mut spans = Vec::new();

            spans.push(Span::styled(
                "❯  ",
                Style::default()
                    .fg(prompt_color)
                    .add_modifier(Modifier::BOLD),
            ));

            if self.text.is_empty() {
                spans.push(Span::styled(
                    "Ask anything...",
                    Style::default().fg(placeholder_color),
                ));
            } else {
                for (index, character) in self.text.chars().enumerate() {
                    let mut style = Style::default().fg(input_color);

                    if let Some(selection) = selection {
                        if index >= selection.start && index < selection.end {
                            style = style.fg(selection_foreground).bg(selection_background);
                        }
                    }

                    spans.push(Span::styled(character.to_string(), style));
                }
            }

            frame.render_widget(
                Paragraph::new(Line::from(spans)).block(
                    Block::default()
                        .borders(Borders::TOP | Borders::BOTTOM)
                        .border_style(Style::default().fg(border_color)),
                ),
                input_area,
            );

            let cursor_x = input_area
                .x
                .saturating_add(3)
                .saturating_add(self.cursor as u16);

            let cursor_x = cursor_x.min(
                input_area
                    .x
                    .saturating_add(input_area.width.saturating_sub(1)),
            );

            frame.set_cursor_position((cursor_x, input_area.y.saturating_add(1)));
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

                        // CUT SELECTED TEXT FROM THE ASK ANYTHING BOX ONLY.
                        //
                        // History selections are intentionally ignored.
                        // Nothing is removed unless an input selection exists.
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
                            let visible_height = terminal.size()?.height.saturating_sub(12);

                            self.scroll_up(8, visible_height);
                        }

                        KeyCode::PageDown => {
                            self.scroll_down(8);
                        }

                        KeyCode::Up => {
                            let visible_height = terminal.size()?.height.saturating_sub(12);

                            self.scroll_up(1, visible_height);
                        }

                        KeyCode::Down => {
                            self.scroll_down(1);
                        }

                        _ => {}
                    }
                }

                Event::Mouse(mouse) => {
                    let size = terminal.size()?;
                    let area = Rect::new(0, 0, size.width, size.height);
                    let layout = Self::terminal_layout(area);

                    match mouse.kind {
                        MouseEventKind::ScrollUp => {
                            self.scroll_up(3, layout[1].height);
                        }

                        MouseEventKind::ScrollDown => {
                            self.scroll_down(3);
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
