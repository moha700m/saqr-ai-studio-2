"""Consent attestation for source identities.

Before a face can be used as a swap source, the operator must attest that it is
their own face or that they have the depicted person's explicit permission, and
that the output will not be used to deceive, defraud, harass, or create intimate
imagery. Records are kept in memory for the session only.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

CONSENT_TEXT = (
    "I confirm that this is my own face, or that the person shown has given me explicit permission to use "
    "their likeness for this purpose. I will not use the result to impersonate anyone, deceive or defraud "
    "people, harass anyone, or create sexual or intimate imagery. I understand that SAQR marks its output "
    "as AI-altered."
)
CONSENT_VERSION = "2026-11"


class ConsentRequired(PermissionError):
    pass


@dataclass
class ConsentRecord:
    attested: bool
    basis: str  # "self" | "permission"
    version: str = CONSENT_VERSION
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.time()

    @property
    def valid(self) -> bool:
        return bool(self.attested) and self.basis in ("self", "permission") and self.version == CONSENT_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def attest(basis: str) -> ConsentRecord:
    return ConsentRecord(attested=True, basis=basis)
