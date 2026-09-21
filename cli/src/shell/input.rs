use std::io::{self, stdout};

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

const ACCENT: Color = Color::Rgb(247, 118, 142);

const LOGO: &str = r#"

  ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗██╗████████╗███████╗
 ██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║██║╚══██╔══╝██╔════╝
 ██║  ███╗██████╔╝███████║██████╔╝███████║██║   ██║   █████╗
 ██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║██║   ██║   ██╔══╝
 ╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║██║   ██║   ███████╗
  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝   ╚═╝   ╚══════╝

"#;

pub struct InputBox {
    pub text: String,
    messages: Vec<String>,

    chat_scroll: u16,
    follow_bottom: bool,

    dirty: bool,
}

impl InputBox {
    pub fn new() -> Self {
        Self {
            text: String::new(),
            messages: Vec::new(),
            chat_scroll: 0,
            follow_bottom: true,
            dirty: true,
        }
    }

    pub fn start() -> io::Result<Terminal<CrosstermBackend<std::io::Stdout>>> {
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
        self.messages
            .iter()
            .flat_map(|message| {
                vec![
                    Line::from(vec![
                        Span::styled(
                            "You",
                            Style::default()
                                .fg(ACCENT)
                                .add_modifier(Modifier::BOLD),
                        ),
                    ]),
                    Line::from(format!("  {message}")),
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

        if self.chat_scroll == 0 {
            self.follow_bottom = true;
        } else {
            self.follow_bottom = false;
        }

        self.dirty = true;
    }

    fn draw(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    ) -> io::Result<()> {
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

            // ─────────────────────────────────────────────
            // Logo
            // ─────────────────────────────────────────────

            let logo_lines: Vec<Line> = LOGO
                .lines()
                .map(|line| {
                    Line::styled(
                        line,
                        Style::default().fg(ACCENT),
                    )
                })
                .collect();

            frame.render_widget(
                Paragraph::new(logo_lines),
                layout[0],
            );

            // Divider below logo.
            let divider_area = Rect {
                x: layout[0].x,
                y: layout[0].y + layout[0].height - 1,
                width: layout[0].width,
                height: 1,
            };

            frame.render_widget(
                Paragraph::new(
                    "─".repeat(divider_area.width as usize)
                )
                .style(
                    Style::default().fg(Color::DarkGray)
                ),
                divider_area,
            );

            // ─────────────────────────────────────────────
            // Chat
            // ─────────────────────────────────────────────

            let visible_height = layout[1].height;
            let max_scroll = self.max_chat_scroll(visible_height);

            let scroll = if self.follow_bottom {
                max_scroll
            } else {
                self.chat_scroll.min(max_scroll)
            };

            frame.render_widget(
                Paragraph::new(self.message_lines())
                    .scroll((scroll, 0)),
                layout[1],
            );

            // ─────────────────────────────────────────────
            // Input
            // ─────────────────────────────────────────────

            let input_area = layout[2];

            let content = if self.text.is_empty() {
                Line::from(vec![
                    Span::styled(
                        "❯  ",
                        Style::default()
                            .fg(ACCENT)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        "Ask Graphite anything...",
                        Style::default()
                            .fg(Color::DarkGray),
                    ),
                ])
            } else {
                Line::from(vec![
                    Span::styled(
                        "❯  ",
                        Style::default()
                            .fg(ACCENT)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(&self.text),
                ])
            };

            frame.render_widget(
                Paragraph::new(content).block(
                    Block::default()
                        .borders(
                            Borders::TOP | Borders::BOTTOM
                        )
                        .border_style(
                            Style::default()
                                .fg(Color::DarkGray),
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
            // Only redraw when something actually changed.
            if self.dirty {
                self.draw(terminal)?;
            }

            // Block waiting for the next event.
            //
            // We do NOT redraw while waiting.
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

                        _ => {
                            // Ignore mouse movement, clicks, etc.
                            //
                            // Most importantly, these do not trigger
                            // a redraw.
                        }
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