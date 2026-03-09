"""
Banking API Client — Simulated banking backend services.

In production, these would be REST/gRPC calls to actual banking microservices.
Here we simulate the APIs with realistic request/response formats to demonstrate
the integration pattern.

API Design:
  All APIs follow a common pattern:
    Request  → Pydantic model with validation
    Response → Standardized envelope: {status, data, error, request_id}
    Errors   → Typed error codes for deterministic handling

Security Notes (production):
  - All API calls must include an auth token (JWT/OAuth2)
  - Card operations require 2FA verification
  - Audit trail for all financial operations
  - PCI-DSS compliance for card data handling
  - No card numbers stored or transmitted in voice pipeline
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field

from src.logging_config import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Common types
# ──────────────────────────────────────────────────────────────────────

class APIStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"


class ErrorCode(str, Enum):
    CARD_NOT_FOUND = "CARD_NOT_FOUND"
    CARD_ALREADY_BLOCKED = "CARD_ALREADY_BLOCKED"
    INVALID_PERIOD = "INVALID_PERIOD"
    ACCOUNT_NOT_FOUND = "ACCOUNT_NOT_FOUND"
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    LOAN_NOT_ELIGIBLE = "LOAN_NOT_ELIGIBLE"


@dataclass
class APIResponse:
    """Standardized API response envelope."""
    status: APIStatus
    data: dict = field(default_factory=dict)
    error: str = ""
    error_code: ErrorCode | None = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    latency_ms: float = 0.0


# ──────────────────────────────────────────────────────────────────────
# Request models
# ──────────────────────────────────────────────────────────────────────

class BlockCardRequest(BaseModel):
    """Request to block a debit/credit card."""
    user_id: str = Field(..., description="Authenticated user ID")
    card_type: str = Field("debit", description="Card type: debit, credit, atm")
    reason: str = Field("lost_or_stolen", description="Reason for blocking")
    # In production: card_last_four: str, verification_token: str


class BankStatementRequest(BaseModel):
    """Request for bank statement generation."""
    user_id: str = Field(..., description="Authenticated user ID")
    days: int = Field(..., ge=1, le=365, description="Number of days for statement")
    format: str = Field("pdf", description="Output format: pdf, csv, json")
    delivery: str = Field("email", description="Delivery method: email, download")


class LoanEnquiryRequest(BaseModel):
    """Request for loan information."""
    user_id: str = Field(..., description="Authenticated user ID")
    loan_type: str = Field("personal", description="Loan type: personal, home, car, education, business")
    amount: float | None = Field(None, description="Desired loan amount")


# ──────────────────────────────────────────────────────────────────────
# Simulated Banking Service
# ──────────────────────────────────────────────────────────────────────

class BankingService:
    """
    Simulated banking backend.
    
    In production, this would be an HTTP/gRPC client calling actual
    banking microservices behind an API gateway.
    
    Simulated latencies:
      - Block card: 200-500ms (includes fraud check)
      - Bank statement: 300-800ms (database query + PDF generation)
      - Loan enquiry: 100-300ms (eligibility lookup)
    """

    # Simulated user database
    _MOCK_USERS = {
        "user_001": {
            "name": "Rahul Sharma",
            "email": "rahul@example.com",
            "cards": {
                "debit": {"status": "active", "last_four": "4532"},
                "credit": {"status": "active", "last_four": "7891"},
            },
            "account_balance": 150000.00,
            "credit_score": 750,
        }
    }

    async def block_card(self, request: BlockCardRequest) -> APIResponse:
        """
        Block a user's card.
        
        Production flow:
          1. Verify user identity (2FA)
          2. Check card exists and is active
          3. Call card management system to block
          4. Generate temporary virtual card
          5. Send confirmation SMS/email
          6. Create audit log entry
          7. Initiate replacement card dispatch
        """
        start = time.monotonic()
        
        # Simulate API latency
        await asyncio.sleep(0.3)

        user = self._MOCK_USERS.get(request.user_id)
        if not user:
            return APIResponse(
                status=APIStatus.ERROR,
                error="User account not found",
                error_code=ErrorCode.ACCOUNT_NOT_FOUND,
                latency_ms=(time.monotonic() - start) * 1000,
            )

        card = user.get("cards", {}).get(request.card_type)
        if not card:
            return APIResponse(
                status=APIStatus.ERROR,
                error=f"No {request.card_type} card found on account",
                error_code=ErrorCode.CARD_NOT_FOUND,
                latency_ms=(time.monotonic() - start) * 1000,
            )

        if card["status"] == "blocked":
            return APIResponse(
                status=APIStatus.ERROR,
                error="Card is already blocked",
                error_code=ErrorCode.CARD_ALREADY_BLOCKED,
                latency_ms=(time.monotonic() - start) * 1000,
            )

        # Block the card
        card["status"] = "blocked"

        logger.info(
            "card_blocked",
            user_id=request.user_id,
            card_type=request.card_type,
            reason=request.reason,
        )

        return APIResponse(
            status=APIStatus.SUCCESS,
            data={
                "card_type": request.card_type,
                "last_four": card["last_four"],
                "blocked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "replacement_eta_days": 7,
                "temporary_virtual_card": True,
            },
            latency_ms=(time.monotonic() - start) * 1000,
        )

    async def get_bank_statement(self, request: BankStatementRequest) -> APIResponse:
        """
        Generate bank statement.
        
        Production flow:
          1. Verify user identity
          2. Query transaction database for date range
          3. Generate PDF/CSV statement
          4. Upload to secure document store
          5. Send download link via email/SMS
          6. Create audit log entry
        """
        start = time.monotonic()
        await asyncio.sleep(0.5)

        user = self._MOCK_USERS.get(request.user_id)
        if not user:
            return APIResponse(
                status=APIStatus.ERROR,
                error="User account not found",
                error_code=ErrorCode.ACCOUNT_NOT_FOUND,
                latency_ms=(time.monotonic() - start) * 1000,
            )

        if request.days > 365:
            return APIResponse(
                status=APIStatus.ERROR,
                error="Statement period cannot exceed 365 days",
                error_code=ErrorCode.INVALID_PERIOD,
                latency_ms=(time.monotonic() - start) * 1000,
            )

        logger.info(
            "statement_generated",
            user_id=request.user_id,
            days=request.days,
            format=request.format,
        )

        return APIResponse(
            status=APIStatus.SUCCESS,
            data={
                "period_days": request.days,
                "format": request.format,
                "delivery_method": request.delivery,
                "total_transactions": 47,
                "opening_balance": 125000.00,
                "closing_balance": user["account_balance"],
                "document_id": str(uuid.uuid4())[:8],
                "download_url": f"https://bank.example.com/statements/{uuid.uuid4()!s:.8}",
                "sent_to": user["email"],
            },
            latency_ms=(time.monotonic() - start) * 1000,
        )

    async def loan_enquiry(self, request: LoanEnquiryRequest) -> APIResponse:
        """
        Process loan enquiry.
        
        Production flow:
          1. Check customer eligibility (credit score, income, existing loans)
          2. Calculate interest rate based on profile
          3. Generate pre-approved offer if eligible
          4. Return loan terms and conditions
        """
        start = time.monotonic()
        await asyncio.sleep(0.2)

        user = self._MOCK_USERS.get(request.user_id)
        if not user:
            return APIResponse(
                status=APIStatus.ERROR,
                error="User account not found",
                error_code=ErrorCode.ACCOUNT_NOT_FOUND,
                latency_ms=(time.monotonic() - start) * 1000,
            )

        # Loan rate calculation based on type and credit score
        base_rates = {
            "personal": 10.5,
            "home": 8.5,
            "car": 9.0,
            "education": 7.5,
            "business": 11.0,
        }

        credit_score = user.get("credit_score", 650)
        base_rate = base_rates.get(request.loan_type, 10.5)
        
        # Adjust rate based on credit score
        if credit_score >= 800:
            rate = base_rate - 1.0
        elif credit_score >= 700:
            rate = base_rate
        else:
            rate = base_rate + 2.0

        max_amounts = {
            "personal": 2500000,
            "home": 50000000,
            "car": 5000000,
            "education": 5000000,
            "business": 10000000,
        }

        logger.info(
            "loan_enquiry_processed",
            user_id=request.user_id,
            loan_type=request.loan_type,
        )

        return APIResponse(
            status=APIStatus.SUCCESS,
            data={
                "loan_type": request.loan_type,
                "annual_interest_rate": rate,
                "max_loan_amount": max_amounts.get(request.loan_type, 2500000),
                "min_loan_amount": 50000,
                "max_tenure_months": 60 if request.loan_type == "personal" else 240,
                "processing_fee_percent": 1.5,
                "eligible": credit_score >= 650,
                "pre_approved_amount": 500000 if credit_score >= 700 else 0,
                "credit_score": credit_score,
            },
            latency_ms=(time.monotonic() - start) * 1000,
        )
