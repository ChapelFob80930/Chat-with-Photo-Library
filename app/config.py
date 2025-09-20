from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    KMP_DUPLICATE_LIB_OK: bool
    GOOGLE_API_KEY: str

    class Config:
        env_file = ".env"
        
#note for me, dont forget to instantiate the Settings class to use it  
settings = Settings()