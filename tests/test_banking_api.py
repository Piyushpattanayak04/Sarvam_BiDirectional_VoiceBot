"""Tests for the banking API service."""

import pytest

from src.banking.api_client import (
    APIStatus,
    BankingService,
    BankStatementRequest,
    BlockCardRequest,
    ErrorCode,
    LoanEnquiryRequest,
)


class TestBankingService:
    """Tests for simulated banking APIs."""

    def setup_method(self):
        self.service = BankingService()

    @pytest.mark.asyncio
    async def test_block_card_success(self):
        request = BlockCardRequest(user_id="user_001", card_type="debit")
        response = await self.service.block_card(request)

        assert response.status == APIStatus.SUCCESS
        assert response.data["card_type"] == "debit"
        assert response.data["last_four"] == "4532"
        assert response.latency_ms > 0

    @pytest.mark.asyncio
    async def test_block_card_user_not_found(self):
        request = BlockCardRequest(user_id="nonexistent", card_type="debit")
        response = await self.service.block_card(request)

        assert response.status == APIStatus.ERROR
        assert response.error_code == ErrorCode.ACCOUNT_NOT_FOUND

    @pytest.mark.asyncio
    async def test_block_card_already_blocked(self):
        request = BlockCardRequest(user_id="user_001", card_type="debit")
        # Block once
        await self.service.block_card(request)
        # Try to block again
        response = await self.service.block_card(request)

        assert response.status == APIStatus.ERROR
        assert response.error_code == ErrorCode.CARD_ALREADY_BLOCKED

    @pytest.mark.asyncio
    async def test_bank_statement_success(self):
        request = BankStatementRequest(user_id="user_001", days=30)
        response = await self.service.get_bank_statement(request)

        assert response.status == APIStatus.SUCCESS
        assert response.data["period_days"] == 30
        assert "download_url" in response.data

    @pytest.mark.asyncio
    async def test_bank_statement_user_not_found(self):
        request = BankStatementRequest(user_id="nonexistent", days=30)
        response = await self.service.get_bank_statement(request)

        assert response.status == APIStatus.ERROR
        assert response.error_code == ErrorCode.ACCOUNT_NOT_FOUND

    @pytest.mark.asyncio
    async def test_loan_enquiry_success(self):
        request = LoanEnquiryRequest(user_id="user_001", loan_type="personal")
        response = await self.service.loan_enquiry(request)

        assert response.status == APIStatus.SUCCESS
        assert response.data["loan_type"] == "personal"
        assert response.data["annual_interest_rate"] > 0
        assert response.data["eligible"] is True

    @pytest.mark.asyncio
    async def test_loan_enquiry_rates_by_type(self):
        """Different loan types should have different rates."""
        rates = {}
        for loan_type in ["personal", "home", "education"]:
            request = LoanEnquiryRequest(user_id="user_001", loan_type=loan_type)
            response = await self.service.loan_enquiry(request)
            rates[loan_type] = response.data["annual_interest_rate"]

        # Education loans should typically have lower rates
        assert rates["education"] < rates["personal"]
