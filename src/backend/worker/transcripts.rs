//! Voice messages transcribed on this computer (`crate::transcribe`).
//!
//! A request is answered from the archive when the message already has a
//! transcript. Otherwise the voice message is downloaded if it has to be, and
//! the transcription runs on a blocking thread; its result comes back as
//! `Command::Transcribed` and is stored only while the message still exists.
//! Every request is answered with `Event::Transcript`, so the window never
//! waits on one forever.

use super::*;

impl Worker {
    pub(super) fn transcribe(&mut self, chat: ChatId, id: String) {
        if !self.transcribing.insert((chat.clone(), id.clone())) {
            return;
        }
        if !self.privacy_ready {
            return self.transcript_failed(chat, id, "Messages are still being restored");
        }
        if let Ok(Some(text)) = self.archive.transcript(&chat, &id) {
            return self.transcript_done(chat, id, Ok(text));
        }
        let message = match self.archive.message(&chat, &id) {
            Ok(Some(message)) => message,
            Ok(None) => {
                return self.transcript_failed(chat, id, "This voice message is not stored here");
            }
            Err(error) => {
                log::warn!("could not read a voice message to transcribe: {error}");
                return self.transcript_failed(chat, id, "Could not read the voice message");
            }
        };
        let Content::Audio {
            voice_note: true,
            media,
            ..
        } = &message.content
        else {
            return self.transcript_failed(chat, id, "Only voice messages can be transcribed");
        };
        match media.path.clone().filter(|path| path.is_file()) {
            Some(path) => self.run_transcription(chat, id, path),
            None => {
                self.emit(Event::TranscriptProgress {
                    chat: chat.clone(),
                    message: id.clone(),
                    progress: crate::transcribe::Progress::Transcribing,
                });
                self.transcript_downloads.insert((chat.clone(), id.clone()));
                self.download(chat, id);
            }
        }
    }

    /// Carries on with a transcription once its voice message has downloaded,
    /// or gives it up when the download failed.
    pub(super) fn transcript_download_settled(
        &mut self,
        chat: ChatId,
        id: String,
        result: Result<PathBuf, String>,
    ) {
        if !self
            .transcript_downloads
            .remove(&(chat.clone(), id.clone()))
        {
            return;
        }
        match result {
            Ok(path) => self.run_transcription(chat, id, path),
            Err(error) => self.transcript_done(chat, id, Err(error)),
        }
    }

    fn run_transcription(&mut self, chat: ChatId, id: String, path: PathBuf) {
        let commands = self.commands.clone();
        let events = self.events.clone();
        let waker = self.waker.clone();
        let models = self.dirs.transcription_model_dir();
        tokio::task::spawn_blocking(move || {
            let result = crate::transcribe::transcribe(&models, &path, |progress| {
                let _ = events.send(Event::TranscriptProgress {
                    chat: chat.clone(),
                    message: id.clone(),
                    progress,
                });
                waker.wake();
            });
            let _ = commands.send(Command::Transcribed {
                chat,
                message: id,
                result,
            });
        });
    }

    /// Stores a finished transcript, unless its message was deleted meanwhile,
    /// and answers the window.
    pub(super) fn transcribed(&mut self, chat: ChatId, id: String, result: Result<String, String>) {
        let result = match result {
            Ok(text) => match self.archive.set_transcript(&chat, &id, &text) {
                Ok(true) => Ok(text),
                Ok(false) => Err("The voice message was deleted".to_owned()),
                Err(error) => {
                    log::warn!("could not store a voice transcript: {error}");
                    // Still shown, just not kept.
                    Ok(text)
                }
            },
            Err(error) => {
                log::warn!("voice transcription failed: {error}");
                Err(error)
            }
        };
        self.transcript_done(chat, id, result);
    }

    fn transcript_failed(&mut self, chat: ChatId, id: String, error: &str) {
        self.transcript_done(chat, id, Err(error.to_owned()));
    }

    fn transcript_done(&mut self, chat: ChatId, id: String, result: Result<String, String>) {
        self.transcribing.remove(&(chat.clone(), id.clone()));
        self.emit(Event::Transcript {
            chat,
            message: id,
            result,
        });
    }

    /// Sends the stored transcripts of messages on their way to the window.
    pub(super) fn emit_transcripts(&self, chat: &str, messages: &[Message]) {
        let ids: Vec<&str> = messages
            .iter()
            .filter(|message| {
                matches!(
                    message.content,
                    Content::Audio {
                        voice_note: true,
                        ..
                    }
                )
            })
            .map(|message| message.id.as_str())
            .collect();
        if ids.is_empty() {
            return;
        }
        match self.archive.transcripts(chat, &ids) {
            Ok(transcripts) if !transcripts.is_empty() => self.emit(Event::Transcripts {
                chat: chat.to_owned(),
                transcripts,
            }),
            Ok(_) => {}
            Err(error) => log::warn!("could not read voice transcripts: {error}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::receipt_tests::{PEER, own_message, worker};
    use super::*;

    fn answers(events: &std::sync::mpsc::Receiver<Event>) -> Vec<Result<String, String>> {
        events
            .try_iter()
            .filter_map(|event| match event {
                Event::Transcript { result, .. } => Some(result),
                _ => None,
            })
            .collect()
    }

    fn voice(id: &str) -> Message {
        let mut row = own_message(id, 1);
        row.from_me = false;
        row.content = Content::Audio {
            media: media(None, None, None, None),
            seconds: Some(3),
            voice_note: true,
            waveform: Vec::new(),
        };
        row
    }

    #[test]
    fn every_request_is_answered() {
        let (mut worker, events, _commands, _wa) = worker();
        worker.archive.ensure_chat(PEER, "Demo").unwrap();
        // Not stored here.
        worker.transcribe(PEER.into(), "missing".into());
        // Not a voice message.
        worker
            .archive
            .insert_message(&own_message("text", 1), None)
            .unwrap();
        worker.transcribe(PEER.into(), "text".into());
        // A voice message that cannot be downloaded (no connection).
        worker
            .archive
            .insert_message(&voice("voice"), None)
            .unwrap();
        worker.transcribe(PEER.into(), "voice".into());
        let answers = answers(&events);
        assert_eq!(answers.len(), 3, "{answers:?}");
        assert!(answers.iter().all(Result::is_err));
        assert!(worker.transcribing.is_empty());
        assert!(worker.transcript_downloads.is_empty());
    }

    #[test]
    fn a_stored_transcript_is_served_from_the_archive() {
        let (mut worker, events, _commands, _wa) = worker();
        worker.archive.ensure_chat(PEER, "Demo").unwrap();
        worker
            .archive
            .insert_message(&voice("voice"), None)
            .unwrap();
        worker
            .archive
            .set_transcript(PEER, "voice", "words")
            .unwrap();
        worker.transcribe(PEER.into(), "voice".into());
        assert_eq!(answers(&events), vec![Ok("words".to_owned())]);
    }

    #[test]
    fn a_result_for_a_deleted_message_is_not_kept() {
        let (mut worker, events, _commands, _wa) = worker();
        worker.archive.ensure_chat(PEER, "Demo").unwrap();
        worker
            .archive
            .insert_message(&voice("voice"), None)
            .unwrap();
        worker.transcribed(PEER.into(), "voice".into(), Ok("kept".into()));
        assert_eq!(answers(&events), vec![Ok("kept".to_owned())]);
        worker.archive.delete_message(PEER, "voice").unwrap();
        worker.transcribed(PEER.into(), "voice".into(), Ok("late".into()));
        assert!(answers(&events)[0].is_err());
        assert_eq!(worker.archive.transcript(PEER, "voice").unwrap(), None);
    }

    #[test]
    fn stored_transcripts_go_out_with_their_page() {
        let (worker, events, _commands, _wa) = worker();
        worker.archive.ensure_chat(PEER, "Demo").unwrap();
        worker
            .archive
            .insert_message(&voice("voice"), None)
            .unwrap();
        worker
            .archive
            .set_transcript(PEER, "voice", "words")
            .unwrap();
        worker.emit_transcripts(PEER, &[voice("voice"), own_message("text", 2)]);
        let sent: Vec<_> = events
            .try_iter()
            .filter_map(|event| match event {
                Event::Transcripts { transcripts, .. } => Some(transcripts),
                _ => None,
            })
            .collect();
        assert_eq!(sent, vec![vec![("voice".to_owned(), "words".to_owned())]]);
    }
}
