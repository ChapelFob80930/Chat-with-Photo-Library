from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key:str
    kmp_duplicate_lib_ok: bool #
    
    class Config:
        env_file = ".env"
        encoding = 'utf-8'
        
#note for me, dont forget to instantiate the Settings class to use it  
settings = Settings()