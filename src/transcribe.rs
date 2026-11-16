//! Voice messages turned into text on this computer, on request.
//!
//! whisper.cpp (through whisper-rs) runs OpenAI's multilingual Whisper
//! large-v3-turbo model, quantized to 5 bits, on the CPU. The model is not
//! bundled: the first request downloads it once from Hugging Face, resuming a
//! cut download, and it is used only once its size and SHA-256 match the pinned
//! ones. No recording or text leaves the computer. The language is detected
//! per message.
//!
//! Everything here blocks and runs on a worker thread, never the window's.
//! One model is loaded at a time, shared by every account, and freed once it
//! has been idle for five minutes.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// The model file, as whisper.cpp publishes it.
pub const MODEL_FILE: &str = "ggml-large-v3-turbo-q5_0.bin";
const MODEL_URL: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin";
const MODEL_SHA256: &str = "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2";
const MODEL_BYTES: u64 = 574_041_195;
/// Names a copy of the same model file to use instead of downloading one,
/// such as one another whisper.cpp tool already fetched. It is checked like a
/// download and refused when it is not the pinned file.
pub const MODEL_ENV: &str = "ZAPFAST_WHISPER_MODEL";
/// How long a loaded model stays in memory after its last transcription.
const IDLE_UNLOAD: Duration = Duration::from_secs(300);
/// Whisper hears 16 kHz mono.
const SAMPLE_RATE: u32 = 16_000;

/// Where a transcription is, for the line under its voice message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Progress {
    /// The one-time model download, in percent once the size is known.
    DownloadingModel(Option<u8>),
    /// The voice message is being fetched or heard.
    Transcribing,
}

#[derive(Default)]
struct Engine {
    context: Option<WhisperContext>,
    /// The model file checked in this process, trusted from then on.
    verified: Option<PathBuf>,
    last_used: Option<Instant>,
    unload_scheduled: bool,
}

fn engine() -> MutexGuard<'static, Engine> {
    static ENGINE: OnceLock<Mutex<Engine>> = OnceLock::new();
    ENGINE
        .get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Transcribes a WhatsApp voice message (OGG/Opus). `models` is the folder
/// the downloaded model lives in; `progress` hears each phase, on this thread.
pub fn transcribe(
    models: &Path,
    recording: &Path,
    mut progress: impl FnMut(Progress),
) -> Result<String, String> {
    static LOGGING: std::sync::Once = std::sync::Once::new();
    LOGGING.call_once(whisper_rs::install_logging_hooks);
    let bytes = std::fs::read(recording)
        .map_err(|error| format!("Could not read the voice message: {error}"))?;
    let mut samples = crate::voice::decode_at(&bytes, SAMPLE_RATE)
        .map_err(|error| format!("Could not decode the voice message: {error}"))?;
    if samples.is_empty() {
        return Err("This voice message has no sound".to_owned());
    }
    // whisper.cpp refuses clips under a second; silence after it is harmless.
    let short = samples.len() < (SAMPLE_RATE * 30) as usize;
    samples.resize(samples.len().max(SAMPLE_RATE as usize), 0.0);

    let mut engine = engine();
    let model = model(&mut engine, models, &mut progress)?;
    progress(Progress::Transcribing);
    if engine.context.is_none() {
        let path = model
            .to_str()
            .ok_or_else(|| "The speech model's path is not valid text".to_owned())?;
        engine.context = Some(
            WhisperContext::new_with_params(path, WhisperContextParameters::default())
                .map_err(|error| format!("Could not load the speech model: {error}"))?,
        );
    }
    let context = engine.context.as_ref().expect("loaded above");
    let mut text = run(
        context,
        &samples,
        short,
        SamplingStrategy::Greedy { best_of: 1 },
    )?;
    // Greedy decoding sometimes returns only punctuation or a bracketed
    // noise tag; one beam search is worth it then, as voxtype does.
    if !text.chars().any(char::is_alphanumeric) {
        text = run(
            context,
            &samples,
            short,
            SamplingStrategy::BeamSearch {
                beam_size: 5,
                patience: -1.0,
            },
        )?;
    }
    engine.last_used = Some(Instant::now());
    schedule_unload(&mut engine);
    drop(engine);
    if !text.chars().any(char::is_alphanumeric) {
        return Err("No speech was found in this voice message".to_owned());
    }
    Ok(text)
}

fn run(
    context: &WhisperContext,
    samples: &[f32],
    short: bool,
    strategy: SamplingStrategy,
) -> Result<String, String> {
    let mut state = context
        .create_state()
        .map_err(|error| format!("Could not start the speech model: {error}"))?;
    let mut params = FullParams::new(strategy);
    params.set_language(Some("auto"));
    params.set_translate(false);
    params.set_n_threads(inference_threads());
    params.set_no_timestamps(true);
    params.set_single_segment(short);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_special(false);
    params.set_print_timestamps(false);
    params.set_suppress_blank(true);
    params.set_suppress_nst(true);
    params.set_no_context(true);
    state
        .full(params, samples)
        .map_err(|error| format!("The speech model could not transcribe this message: {error}"))?;
    let mut text = String::new();
    for segment in state.as_iter() {
        let words = segment
            .to_str_lossy()
            .map_err(|error| format!("The speech model returned unreadable text: {error}"))?;
        text.push_str(&words);
    }
    Ok(text.trim().to_owned())
}

/// Half the logical cores, at least four where there are four, at most 16:
/// whisper.cpp gains little past the performance cores, and the rest keep the
/// window and calls responsive while a long message is heard.
fn inference_threads() -> i32 {
    let logical = std::thread::available_parallelism().map_or(1, usize::from);
    logical.div_ceil(2).clamp(4.min(logical), 16) as i32
}

/// Frees the model once it has gone unused for [`IDLE_UNLOAD`]; it takes
/// about a gigabyte of memory, and the window may stay open for days.
fn schedule_unload(engine: &mut Engine) {
    if engine.unload_scheduled {
        return;
    }
    engine.unload_scheduled = true;
    let spawned = std::thread::Builder::new()
        .name("whisper-idle".into())
        .spawn(|| {
            loop {
                let wait = {
                    let mut engine = self::engine();
                    let idle = engine.last_used.map_or(IDLE_UNLOAD, |used| used.elapsed());
                    if idle >= IDLE_UNLOAD {
                        engine.context = None;
                        engine.unload_scheduled = false;
                        return;
                    }
                    IDLE_UNLOAD - idle
                };
                std::thread::sleep(wait);
            }
        });
    if spawned.is_err() {
        engine.unload_scheduled = false;
    }
}

/// The checked model file: the one [`MODEL_ENV`] names, or the downloaded
/// one, fetched first when it is missing or damaged.
fn model(
    engine: &mut Engine,
    directory: &Path,
    progress: &mut impl FnMut(Progress),
) -> Result<PathBuf, String> {
    if let Some(path) = engine.verified.as_ref().filter(|path| path.is_file()) {
        return Ok(path.clone());
    }
    let chosen = std::env::var_os(MODEL_ENV).filter(|path| !path.is_empty());
    let path = resolve(directory, chosen.map(PathBuf::from), |directory| {
        download(directory, progress)
    })?;
    engine.verified = Some(path.clone());
    // A different file means a different model: load it afresh.
    engine.context = None;
    Ok(path)
}

fn resolve(
    directory: &Path,
    chosen: Option<PathBuf>,
    download: impl FnOnce(&Path) -> Result<PathBuf, String>,
) -> Result<PathBuf, String> {
    if let Some(path) = chosen {
        return if is_model(&path) {
            Ok(path)
        } else {
            Err(format!(
                "{MODEL_ENV} does not name the expected {MODEL_FILE} model"
            ))
        };
    }
    let installed = directory.join(MODEL_FILE);
    if is_model(&installed) {
        return Ok(installed);
    }
    download(directory)
}

/// Whether `path` is the pinned model, by size and then digest.
fn is_model(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|meta| meta.is_file() && meta.len() == MODEL_BYTES)
        && sha256_file(path).is_ok_and(|digest| digest == MODEL_SHA256)
}

/// Downloads the model into `directory`, resuming an earlier partial file.
/// Only a file of the pinned size and digest is installed.
fn download(directory: &Path, progress: &mut impl FnMut(Progress)) -> Result<PathBuf, String> {
    std::fs::create_dir_all(directory)
        .map_err(|error| format!("Could not create the speech model folder: {error}"))?;
    let target = directory.join(MODEL_FILE);
    let partial = directory.join(format!("{MODEL_FILE}.part"));
    let failed = |error: String| format!("Could not download the speech model: {error}");

    let mut hash = Sha256::new();
    let mut have = match std::fs::metadata(&partial) {
        Ok(meta) if meta.len() <= MODEL_BYTES => {
            hash_into(&partial, &mut hash).map_err(|error| failed(error.to_string()))?
        }
        _ => 0,
    };
    let percent = |have: u64| Some((have.saturating_mul(100) / MODEL_BYTES).min(100) as u8);
    progress(Progress::DownloadingModel(percent(have)));

    if have < MODEL_BYTES {
        let mut builder = reqwest::blocking::Client::builder()
            .connect_timeout(Duration::from_secs(30))
            // The blocking client applies this to each read, not the whole
            // body: a stalled download fails, a slow but moving one does not.
            .timeout(Duration::from_secs(60));
        if let Some(proxy) = crate::proxy::reqwest_proxy() {
            builder = builder.proxy(proxy);
        }
        let client = builder.build().map_err(|error| failed(error.to_string()))?;
        let mut request = client.get(MODEL_URL);
        if have > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={have}-"));
        }
        let mut response = request.send().map_err(|error| failed(error.to_string()))?;
        let resumed = have > 0 && response.status() == reqwest::StatusCode::PARTIAL_CONTENT;
        if !response.status().is_success() {
            // A range the server cannot serve: start again next time.
            let _ = std::fs::remove_file(&partial);
            return Err(failed(format!("HTTP {}", response.status().as_u16())));
        }
        let mut file = if resumed {
            std::fs::OpenOptions::new().append(true).open(&partial)
        } else {
            have = 0;
            hash = Sha256::new();
            std::fs::File::create(&partial)
        }
        .map_err(|error| format!("Could not save the speech model: {error}"))?;
        let mut buffer = vec![0u8; 256 * 1024];
        let mut shown = percent(have);
        loop {
            let read = response
                .read(&mut buffer)
                .map_err(|error| failed(error.to_string()))?;
            if read == 0 {
                break;
            }
            have += read as u64;
            if have > MODEL_BYTES {
                drop(file);
                let _ = std::fs::remove_file(&partial);
                return Err(failed("the file is larger than expected".to_owned()));
            }
            hash.update(&buffer[..read]);
            file.write_all(&buffer[..read])
                .map_err(|error| format!("Could not save the speech model: {error}"))?;
            if percent(have) != shown {
                shown = percent(have);
                progress(Progress::DownloadingModel(shown));
            }
        }
        file.sync_all()
            .map_err(|error| format!("Could not save the speech model: {error}"))?;
    }
    if have != MODEL_BYTES || hex(hash.finalize()) != MODEL_SHA256 {
        let _ = std::fs::remove_file(&partial);
        return Err("The downloaded speech model failed its integrity check".to_owned());
    }
    std::fs::rename(&partial, &target)
        .map_err(|error| format!("Could not install the speech model: {error}"))?;
    Ok(target)
}

/// Feeds a file to `hash` and returns its length.
fn hash_into(path: &Path, hash: &mut Sha256) -> std::io::Result<u64> {
    let mut file = std::fs::File::open(path)?;
    let mut buffer = vec![0u8; 256 * 1024];
    let mut length = 0;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            return Ok(length);
        }
        length += read as u64;
        hash.update(&buffer[..read]);
    }
}

fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut hash = Sha256::new();
    hash_into(path, &mut hash)?;
    Ok(hex(hash.finalize()))
}

fn hex(digest: impl AsRef<[u8]>) -> String {
    digest
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_that_is_not_the_model_is_never_used() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join(MODEL_FILE), b"not a model").unwrap();
        assert!(!is_model(&directory.path().join(MODEL_FILE)));
        // The damaged file is fetched again rather than loaded.
        let fetched = resolve(directory.path(), None, |_| Err("fetched".to_owned()));
        assert_eq!(fetched, Err("fetched".to_owned()));
    }

    #[test]
    fn a_chosen_model_is_checked_and_never_replaced_by_a_download() {
        let directory = tempfile::tempdir().unwrap();
        let chosen = directory.path().join("other.bin");
        std::fs::write(&chosen, b"not a model").unwrap();
        let result = resolve(directory.path(), Some(chosen), |_| {
            panic!("a chosen model is not downloaded")
        });
        assert!(result.unwrap_err().contains(MODEL_ENV));
    }

    #[test]
    fn digests_are_lowercase_hex() {
        assert_eq!(
            hex(Sha256::digest(b"fixture")),
            "f16d05ec6b29248d2c61adb1e9263f78e4f7bace1b955014a2d17872cfe4064d"
        );
    }

    #[test]
    fn inference_leaves_cores_for_the_window() {
        let logical = std::thread::available_parallelism().map_or(1, usize::from) as i32;
        let threads = inference_threads();
        assert!((1..=logical.min(16)).contains(&threads));
        if logical >= 8 {
            assert!(threads >= 4 && threads < logical);
        }
    }

    /// A real run, by hand: `ZAPFAST_WHISPER_MODEL=<ggml-large-v3-turbo-q5_0.bin>
    /// ZAPFAST_WHISPER_PROBE=<note.ogg> cargo test --release transcribe::tests::probe
    /// -- --ignored --nocapture`. Prints the text, which stays on the screen.
    #[test]
    #[ignore = "needs the model and a recording"]
    fn probe() {
        let Some(recording) = std::env::var_os("ZAPFAST_WHISPER_PROBE") else {
            return;
        };
        let directory = tempfile::tempdir().unwrap();
        let started = Instant::now();
        let text = transcribe(directory.path(), Path::new(&recording), |_| {}).expect("text");
        let cold = started.elapsed();
        let started = Instant::now();
        let again = transcribe(directory.path(), Path::new(&recording), |_| {}).expect("text");
        assert_eq!(text, again);
        println!("{text}\ncold {cold:.2?}, warm {:.2?}", started.elapsed());
    }
}
