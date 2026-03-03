"""
Email Processor (.msg and .eml)
Extracts subject, sender, recipients, date, body, and attachment names from emails.
"""

import logging
import email
from email import policy
from typing import List, Dict, Any
from pathlib import Path
from .base import BaseProcessor

logger = logging.getLogger(__name__)


class EmailProcessor(BaseProcessor):

    supported_extensions = [".eml", ".msg"]

    def extract(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract text from email files."""
        self.validate_file(file_path)

        ext = Path(file_path).suffix.lower()

        if ext == ".eml":
            return self._extract_eml(file_path)
        elif ext == ".msg":
            return self._extract_msg(file_path)

        return []

    def _extract_eml(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract from standard .eml files (RFC 2822)."""
        file_name = Path(file_path).name

        with open(file_path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=policy.default)

        subject = msg.get("Subject", "No Subject")
        sender = msg.get("From", "Unknown")
        to = msg.get("To", "Unknown")
        date = msg.get("Date", "Unknown")

        # Extract body text
        body = self._get_email_body(msg)

        # Build header block for context
        header_text = (
            f"Subject: {subject}\n"
            f"From: {sender}\n"
            f"To: {to}\n"
            f"Date: {date}"
        )

        # List attachment names
        attachments = []
        for part in msg.walk():
            filename = part.get_filename()
            if filename:
                attachments.append(filename)

        sections = []

        # Header section
        sections.append({
            "content": header_text,
            "metadata": {
                "source": file_name,
                "subject": subject,
                "sender": sender,
                "date": date,
                "attachments": attachments,
            },
            "chunk_type": "email_header",
        })

        # Body section
        if body and body.strip():
            sections.append({
                "content": body.strip(),
                "metadata": {
                    "source": file_name,
                    "subject": subject,
                    "sender": sender,
                    "date": date,
                },
                "chunk_type": "email_body",
            })

        logger.info(f"Extracted {len(sections)} sections from {file_name}")
        return sections

    def _extract_msg(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract from Outlook .msg files using extract-msg library."""
        file_name = Path(file_path).name

        try:
            import extract_msg

            msg = extract_msg.Message(file_path)

            subject = msg.subject or "No Subject"
            sender = msg.sender or "Unknown"
            to = msg.to or "Unknown"
            date = str(msg.date) if msg.date else "Unknown"
            body = msg.body or ""

            attachments = [att.longFilename or att.shortFilename or "unnamed"
                           for att in msg.attachments]

            header_text = (
                f"Subject: {subject}\n"
                f"From: {sender}\n"
                f"To: {to}\n"
                f"Date: {date}"
            )

            sections = []

            sections.append({
                "content": header_text,
                "metadata": {
                    "source": file_name,
                    "subject": subject,
                    "sender": sender,
                    "date": date,
                    "attachments": attachments,
                },
                "chunk_type": "email_header",
            })

            if body.strip():
                sections.append({
                    "content": body.strip(),
                    "metadata": {
                        "source": file_name,
                        "subject": subject,
                        "sender": sender,
                        "date": date,
                    },
                    "chunk_type": "email_body",
                })

            msg.close()
            logger.info(f"Extracted {len(sections)} sections from {file_name}")
            return sections

        except ImportError:
            logger.error(
                "extract-msg library not installed. "
                "Install it with: pip install extract-msg"
            )
            return []
        except Exception as e:
            logger.error(f"Failed to extract .msg file {file_name}: {e}")
            return []

    def _get_email_body(self, msg) -> str:
        """Extract the plain text body from an email message."""
        body = msg.get_body(preferencelist=("plain", "html"))

        if body is None:
            return ""

        content = body.get_content()

        # If HTML, do a basic strip of tags
        if body.get_content_type() == "text/html":
            content = self._strip_html(content)

        return content

    def _strip_html(self, html: str) -> str:
        """Basic HTML tag removal for email bodies."""
        import re
        # Remove script/style blocks
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.DOTALL | re.IGNORECASE)
        # Remove tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
