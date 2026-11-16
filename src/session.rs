//! The state every miner of a run shares: the work counter, the stop flag,
//! the verified hits, and the status display.

use crate::keccak::{checksum, hex};
use crate::pattern::Criteria;
use crate::plan::{Guard, Plan, Scheme};
use console::{style, Term};
use std::fmt::Write as _;
use std::fs::File;
use std::io::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// How progress is reported on stderr.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatusMode {
    /// A live panel repainted in place (stderr is a terminal).
    Panel,
    /// One line every few seconds (stderr is a pipe or file).
    Lines,
    /// Nothing but errors.
    Quiet,
}

/// A verified salt and the address it deploys to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hit {
    pub salt: [u8; 32],
    pub address: [u8; 20],
}

impl Hit {
    /// The line printed for a hit: `<salt> <checksummed address>`.
    pub fn line(&self) -> String {
        format!("0x{} {}", hex(&self.salt), checksum(&self.address))
    }
}

struct State {
    hits: Vec<Hit>,
    output: Option<File>,
    error: Option<String>,
    miners: Vec<(&'static str, String)>,
    /// lines of the panel currently on screen
    drawn: usize,
    last_report: Option<Instant>,
    window_start: Instant,
    window_tried: u64,
    rate: f64,
}

pub struct Session {
    pub plan: Plan,
    pub criteria: Criteria,
    /// hits to collect before stopping; 0 means run until interrupted
    want: u64,
    tried: AtomicU64,
    stop: AtomicBool,
    start: Instant,
    mode: StatusMode,
    difficulty_bits: f64,
    state: Mutex<State>,
}

const PANEL_INTERVAL: Duration = Duration::from_secs(1);
const LINES_INTERVAL: Duration = Duration::from_secs(10);

impl Session {
    pub fn new(
        plan: Plan,
        criteria: Criteria,
        want: u64,
        output: Option<File>,
        mode: StatusMode,
    ) -> Self {
        let now = Instant::now();
        Self {
            difficulty_bits: criteria.difficulty_bits(),
            plan,
            criteria,
            want,
            tried: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            start: now,
            mode,
            state: Mutex::new(State {
                hits: Vec::new(),
                output,
                error: None,
                miners: Vec::new(),
                drawn: 0,
                last_report: None,
                window_start: now,
                window_tried: 0,
                rate: 0.0,
            }),
        }
    }

    /// Name a miner (`gpu`, `cpu`) in the status header once it is running.
    pub fn add_miner(&self, label: &'static str, description: String) {
        self.state.lock().unwrap().miners.push((label, description));
    }

    /// Record `n` candidates tried.
    pub fn add_tried(&self, n: u64) {
        self.tried.fetch_add(n, Ordering::Relaxed);
    }

    pub fn tried(&self) -> u64 {
        self.tried.load(Ordering::Relaxed)
    }

    pub fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    /// Stop the run with an error (the first one wins).
    pub fn fail(&self, message: String) {
        self.state.lock().unwrap().error.get_or_insert(message);
        self.stop.store(true, Ordering::Relaxed);
    }

    /// Offer a salt a miner believes is a hit. It is recomputed through the
    /// reference derivation first: a salt that does not actually match means
    /// the backend that produced it is broken, and that ends the run rather
    /// than letting it keep mining with a kernel that cannot be trusted.
    pub fn submit(&self, salt: [u8; 32], source: &str) {
        let address = self.plan.address(&salt);
        if !self.criteria.matches(&address) {
            self.fail(format!(
                "the {source} miner reported salt 0x{} but it derives {}, which does not match \
                 the target; this is a bug in that backend",
                hex(&salt),
                checksum(&address)
            ));
            return;
        }

        let mut state = self.state.lock().unwrap();
        let full = self.want != 0 && state.hits.len() as u64 >= self.want;
        if full || state.hits.iter().any(|hit| hit.salt == salt) {
            return;
        }
        let hit = Hit { salt, address };
        let line = hit.line();

        // the hit goes to stdout above the panel, which is redrawn below it
        self.erase(&mut state);
        let printed = if cfg!(test) {
            Ok(()) // keep the test log free of mined salts
        } else {
            let mut stdout = std::io::stdout().lock();
            writeln!(stdout, "{line}").and_then(|()| stdout.flush())
        };
        let saved = match state.output.as_mut() {
            Some(file) => writeln!(file, "{line}").and_then(|()| file.flush()),
            None => Ok(()),
        };

        state.hits.push(hit);
        let done = self.want != 0 && state.hits.len() as u64 >= self.want;
        if let Err(e) = printed.and(saved) {
            state
                .error
                .get_or_insert(format!("could not write a hit: {e}"));
            self.stop.store(true, Ordering::Relaxed);
        } else if done {
            self.stop.store(true, Ordering::Relaxed);
        } else {
            self.paint(&mut state, Instant::now());
        }
    }

    /// Refresh the status display if it is due. Called from the main thread
    /// a few times a second.
    pub fn tick(&self) {
        let interval = match self.mode {
            StatusMode::Panel => PANEL_INTERVAL,
            StatusMode::Lines => LINES_INTERVAL,
            StatusMode::Quiet => return,
        };
        let now = Instant::now();
        let mut state = self.state.lock().unwrap();
        let since = state.last_report.unwrap_or(self.start);
        // the panel appears immediately; lines wait a full interval
        let first_panel = self.mode == StatusMode::Panel && state.last_report.is_none();
        if !first_panel && now.duration_since(since) < interval {
            return;
        }
        self.paint(&mut state, now);
    }

    /// Take the panel down, print the closing summary, and hand back the
    /// hits or the error that ended the run.
    pub fn finish(&self) -> Result<Vec<Hit>, String> {
        let mut state = self.state.lock().unwrap();
        self.erase(&mut state);
        if let Some(error) = state.error.take() {
            return Err(error);
        }
        if self.mode != StatusMode::Quiet {
            let tried = self.tried();
            let secs = self.start.elapsed().as_secs_f64();
            eprintln!(
                "{} hit(s) after {} salts in {} ({:.1} M/s average)",
                state.hits.len(),
                fmt_count(tried),
                fmt_duration(secs),
                tried as f64 / secs.max(1e-9) / 1e6
            );
        }
        Ok(state.hits.clone())
    }

    fn erase(&self, state: &mut State) {
        if state.drawn > 0 {
            let _ = Term::stderr().clear_last_lines(state.drawn);
            state.drawn = 0;
        }
    }

    fn paint(&self, state: &mut State, now: Instant) {
        if self.mode == StatusMode::Quiet {
            return;
        }
        let tried = self.tried();

        // windowed rate, so it tracks current throughput (thermal throttling
        // included) rather than the lifetime average
        let window = now.duration_since(state.window_start).as_secs_f64();
        if window >= 0.5 {
            state.rate = (tried - state.window_tried) as f64 / window;
            state.window_start = now;
            state.window_tried = tried;
        }
        state.last_report = Some(now);
        let elapsed = now.duration_since(self.start).as_secs_f64();

        if self.mode == StatusMode::Lines {
            eprintln!(
                "{}: {} salts tried, {:.1} M/s, {} found",
                fmt_duration(elapsed),
                fmt_count(tried),
                state.rate / 1e6,
                state.hits.len()
            );
            return;
        }

        let lines = self.panel(state, tried, elapsed);
        let term = Term::stderr();
        let width = term.size().1.max(20) as usize - 1;
        let mut out = String::new();
        for line in &lines {
            // a wrapped line would throw off the count erase() relies on
            let _ = writeln!(out, "{}", console::truncate_str(line, width, "\u{2026}"));
        }
        self.erase(state);
        if term.write_str(&out).is_ok() {
            state.drawn = lines.len();
        }
    }

    fn panel(&self, state: &State, tried: u64, elapsed: f64) -> Vec<String> {
        let label = |s: &str| format!("{:>11}", style(s).dim());
        let dim = |s: String| style(s).dim().to_string();
        let rule = "\u{2500}".repeat(64);
        let plan = &self.plan;

        let mut lines = vec![style(format!("saltmill {rule}")).cyan().to_string()];
        lines.push(format!("{} {}", label("scheme"), scheme_label(plan)));
        lines.push(format!("{} {}", label("factory"), checksum(&plan.factory)));
        if plan.guard != Guard::None || plan.caller != [0u8; 20] {
            lines.push(format!("{} {}", label("caller"), checksum(&plan.caller)));
        }
        let hash_label = match plan.scheme {
            Scheme::Create2 => "init hash",
            Scheme::Create3 => "proxy hash",
        };
        lines.push(format!(
            "{} 0x{}",
            label(hash_label),
            hex(&plan.init_code_hash)
        ));
        for (name, description) in &state.miners {
            lines.push(format!("{} {}", label(name), description));
        }
        lines.push(format!(
            "{} {}   {}",
            label("target"),
            style(&self.criteria).green(),
            dim(format!("1 in 2^{:.1}", self.difficulty_bits))
        ));
        lines.push(dim(format!("{rule}{}", "\u{2500}".repeat(9))));

        let found = match self.want {
            0 => state.hits.len().to_string(),
            want => format!("{} / {want}", state.hits.len()),
        };
        lines.push(format!(
            "{} {:<22}{} {}",
            label("elapsed"),
            fmt_duration(elapsed),
            label("tried"),
            fmt_count(tried),
        ));
        lines.push(format!(
            "{} {:<22}{} {}",
            label("rate"),
            format!("{:.1} M/s", state.rate / 1e6),
            label("found"),
            found,
        ));

        // expected time per hit, and the chance a hit was due by now
        let expected = 2f64.powf(self.difficulty_bits);
        let (eta, due) = if state.rate > 0.0 {
            let chance = 1.0 - (-(tried as f64) / expected).exp();
            (
                fmt_duration(expected / state.rate),
                format!("{:.0}%", chance * 100.0),
            )
        } else {
            ("\u{2014}".to_string(), "\u{2014}".to_string())
        };
        lines.push(format!(
            "{} {:<22}{} {}",
            label("avg/hit"),
            eta,
            label("P(\u{2265}1 hit)"),
            due,
        ));
        lines
    }
}

/// One line naming the derivation a plan mines for.
pub fn scheme_label(plan: &Plan) -> String {
    let scheme = match plan.scheme {
        Scheme::Create2 => "CREATE2",
        Scheme::Create3 => "CREATE3",
    };
    let mined = plan.mined_bytes();
    match plan.guard {
        Guard::None => format!("{scheme} \u{00b7} raw salt, {mined} bytes mined"),
        Guard::CreateX { chain_id: None } => format!(
            "{scheme} via CreateX \u{00b7} permissioned salt, {mined} bytes mined, any chain"
        ),
        Guard::CreateX {
            chain_id: Some(chain_id),
        } => format!(
            "{scheme} via CreateX \u{00b7} permissioned salt, {mined} bytes mined, chain {chain_id} only"
        ),
    }
}

/// Format a count with a scale word (e.g. `1.23 billion`).
pub fn fmt_count(n: u64) -> String {
    const UNITS: [(f64, &str); 4] = [
        (1e12, "trillion"),
        (1e9, "billion"),
        (1e6, "million"),
        (1e3, "thousand"),
    ];
    let f = n as f64;
    for (scale, name) in UNITS {
        if f >= scale {
            return format!("{:.2} {name}", f / scale);
        }
    }
    n.to_string()
}

/// Format a duration in seconds as a compact human string (e.g. `2h 3m`,
/// `6m 12s`, `4.2s`), or a marker for estimates too long to matter.
pub fn fmt_duration(secs: f64) -> String {
    if !secs.is_finite() || secs >= 315_360_000.0 {
        return "> 10 years".to_string();
    }
    if secs < 10.0 {
        return format!("{:.1}s", secs.max(0.0));
    }
    let secs = secs as u64;
    let (d, h, m, s) = (secs / 86400, secs / 3600 % 24, secs / 60 % 60, secs % 60);
    if d > 0 {
        format!("{d}d {h}h")
    } else if h > 0 {
        format!("{h}h {m}m")
    } else if m > 0 {
        format!("{m}m {s}s")
    } else {
        format!("{s}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern::Pattern;
    use crate::plan::tests::all_plans;

    fn session(want: u64) -> Session {
        // every address whose last nibble is zero is a hit: easy to find by
        // scanning counters in a test
        let criteria = Criteria {
            pattern: Pattern::from_suffix(&[0]),
            total_zero_bytes: None,
        };
        let plan = all_plans().remove(5);
        Session::new(plan, criteria, want, None, StatusMode::Quiet)
    }

    fn salts(session: &Session, matching: bool) -> impl Iterator<Item = [u8; 32]> + '_ {
        (0u64..)
            .map(|counter| session.plan.salt(&[0; 3], counter))
            .filter(move |salt| session.criteria.matches(&session.plan.address(salt)) == matching)
    }

    #[test]
    fn submit_collects_hits_until_the_count_is_reached() {
        let session = session(2);
        let hits: Vec<_> = salts(&session, true).take(3).collect();

        session.submit(hits[0], "test");
        session.submit(hits[0], "test"); // duplicates are ignored
        assert!(!session.stopped());
        session.submit(hits[1], "test");
        assert!(session.stopped());
        session.submit(hits[2], "test"); // past the count: dropped

        let found = session.finish().unwrap();
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].salt, hits[0]);
        assert_eq!(found[1].address, session.plan.address(&hits[1]));
    }

    #[test]
    fn submit_fails_the_run_on_a_salt_that_does_not_match() {
        let session = session(1);
        let miss = salts(&session, false).next().unwrap();
        session.submit(miss, "gpu");
        assert!(session.stopped());
        let error = session.finish().unwrap_err();
        assert!(error.contains("gpu miner"), "{error}");
    }

    #[test]
    fn hit_line_is_salt_then_checksummed_address() {
        let hit = Hit {
            salt: [0x0a; 32],
            address: crate::plan::CREATEX,
        };
        assert_eq!(
            hit.line(),
            format!(
                "0x{} 0xba5Ed099633D3B313e4D5F7bdc1305d3c28ba5Ed",
                "0a".repeat(32)
            )
        );
    }

    #[test]
    fn formats_counts_and_durations() {
        assert_eq!(fmt_count(999), "999");
        assert_eq!(fmt_count(1_230_000_000), "1.23 billion");
        assert_eq!(fmt_duration(4.25), "4.2s");
        assert_eq!(fmt_duration(372.0), "6m 12s");
        assert_eq!(fmt_duration(90_000.0), "1d 1h");
        assert_eq!(fmt_duration(f64::INFINITY), "> 10 years");
    }
}
