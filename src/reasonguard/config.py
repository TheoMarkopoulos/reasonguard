from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "", "env_file": ".env", "env_file_encoding": "utf-16"}

    upstream_base_url: str = "https://api.openai.com/v1"
    upstream_api_key: str = ""
    redis_url: str = "redis://localhost:6379/0"
    default_tau: float = 0.0
    hvr_gate_enabled: bool = True
    enable_mi_proxy: bool = False
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
