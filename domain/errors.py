class AIProviderCreditExhaustedError(RuntimeError):
    """Raised when the configured AI provider has no remaining API credit."""

    code = "ai_provider_credit_exhausted"

    def __init__(self, provider: str = "OpenAI") -> None:
        self.provider = provider
        super().__init__(
            "Le service d’assistance IA est temporairement indisponible : "
            f"le crédit {provider} est épuisé. Aucune recette n’a été créée ou modifiée. "
            f"Rechargez le compte {provider} ou contactez l’administrateur, puis réessayez."
        )
