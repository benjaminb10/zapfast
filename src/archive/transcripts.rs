//! Text of voice messages transcribed on this computer, kept in the encrypted
//! archive beside the messages.
//!
//! A row lives only as long as its message: deleting the message (here, from
//! another device, or by clearing or removing the chat) deletes the row
//! through a trigger, and a message deleted for everyone loses it with its
//! content. A transcription that finishes after its message went is not
//! stored.

use rusqlite::{OptionalExtension, params};

use super::{Archive, Result};

pub const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS transcripts (
    chat TEXT NOT NULL,
    id TEXT NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (chat, id)
);
CREATE TRIGGER IF NOT EXISTS delete_transcript AFTER DELETE ON messages BEGIN
    DELETE FROM transcripts WHERE chat = OLD.chat AND id = OLD.id;
END;
";

impl Archive {
    /// Stores a message's transcript, replacing an earlier one. Returns false,
    /// storing nothing, when the message is no longer in the archive.
    pub fn set_transcript(&self, chat: &str, id: &str, text: &str) -> Result<bool> {
        let changed = self.connection.execute(
            "INSERT INTO transcripts (chat, id, text)
             SELECT ?1, ?2, ?3
             WHERE EXISTS (SELECT 1 FROM messages WHERE chat = ?1 AND id = ?2
                AND json_extract(content, '$.kind') <> 'revoked')
             ON CONFLICT(chat, id) DO UPDATE SET text = excluded.text",
            params![chat, id, text],
        )?;
        Ok(changed > 0)
    }

    /// One message's transcript. Personal data: never logged.
    pub fn transcript(&self, chat: &str, id: &str) -> Result<Option<String>> {
        self.connection
            .query_row(
                "SELECT text FROM transcripts WHERE chat = ?1 AND id = ?2",
                params![chat, id],
                |row| row.get(0),
            )
            .optional()
    }

    /// The transcripts among `ids` in one chat, as (message id, text).
    pub fn transcripts(&self, chat: &str, ids: &[&str]) -> Result<Vec<(String, String)>> {
        let mut found = Vec::new();
        let mut statement = self
            .connection
            .prepare_cached("SELECT text FROM transcripts WHERE chat = ?1 AND id = ?2")?;
        for id in ids {
            if let Some(text) = statement
                .query_row(params![chat, id], |row| row.get::<_, String>(0))
                .optional()?
            {
                found.push(((*id).to_owned(), text));
            }
        }
        Ok(found)
    }

    /// Drops a message's transcript, when its words are gone for everyone.
    pub(super) fn delete_transcript(&self, chat: &str, id: &str) -> Result<()> {
        self.connection.execute(
            "DELETE FROM transcripts WHERE chat = ?1 AND id = ?2",
            params![chat, id],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::message;
    use super::*;
    use crate::model::Content;

    const CHAT: &str = "1@s.whatsapp.net";

    #[test]
    fn a_transcript_lives_as_long_as_its_message() {
        let archive = Archive::in_memory().unwrap();
        archive.ensure_chat(CHAT, "Ada").unwrap();
        archive
            .insert_message(&message(CHAT, "a", 1, false), None)
            .unwrap();
        assert!(archive.set_transcript(CHAT, "a", "first").unwrap());
        assert!(archive.set_transcript(CHAT, "a", "second").unwrap());
        assert_eq!(
            archive.transcript(CHAT, "a").unwrap().as_deref(),
            Some("second")
        );
        assert_eq!(
            archive.transcripts(CHAT, &["a", "b"]).unwrap(),
            vec![("a".to_owned(), "second".to_owned())]
        );
        archive.delete_message(CHAT, "a").unwrap();
        assert_eq!(archive.transcript(CHAT, "a").unwrap(), None);
        // A result that finishes after the message went is not kept.
        assert!(!archive.set_transcript(CHAT, "a", "late").unwrap());
        assert_eq!(archive.transcript(CHAT, "a").unwrap(), None);
    }

    #[test]
    fn a_message_deleted_for_everyone_loses_its_transcript() {
        let archive = Archive::in_memory().unwrap();
        archive.ensure_chat(CHAT, "Ada").unwrap();
        archive
            .insert_message(&message(CHAT, "a", 1, false), None)
            .unwrap();
        archive.set_transcript(CHAT, "a", "words").unwrap();
        archive
            .set_content(CHAT, "a", &Content::Revoked, false)
            .unwrap();
        assert_eq!(archive.transcript(CHAT, "a").unwrap(), None);
        assert!(!archive.set_transcript(CHAT, "a", "late").unwrap());
    }

    #[test]
    fn clearing_the_archive_clears_transcripts() {
        let archive = Archive::in_memory().unwrap();
        archive.ensure_chat(CHAT, "Ada").unwrap();
        archive
            .insert_message(&message(CHAT, "a", 1, false), None)
            .unwrap();
        archive.set_transcript(CHAT, "a", "words").unwrap();
        archive.clear().unwrap();
        archive
            .insert_message(&message(CHAT, "a", 1, false), None)
            .unwrap();
        assert_eq!(archive.transcript(CHAT, "a").unwrap(), None);
    }
}
