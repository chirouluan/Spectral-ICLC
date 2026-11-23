"""Copy to local_api_config.py and fill in a provider's API settings."""
PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "",  # Set the exact provider model ID.
        "api_key": "",  # Keep this local; never commit the copied file.
    },
    "openai_compatible": {
        "base_url": "",
        "model": "",
        "api_key": "",
    },
}
