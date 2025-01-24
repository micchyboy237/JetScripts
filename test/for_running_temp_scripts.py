from jet.logger import logger
from jet.transformers.formatters import format_json
from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class UserProfile(BaseModel):
    name: str = Field(..., min_length=1)
    age: Optional[int] = Field(None, ge=0)
    email: EmailStr

    class Config:
        schema_extra = {
            "example": {
                "name": "John Doe",
                "age": 30,
                "email": "johndoe@example.com"
            }
        }


# Generate the schema and add the $schema field manually
schema = UserProfile.model_json_schema()
schema["$schema"] = "http://json-schema.org/draft-07/schema#"


# Generate the schema
logger.success(format_json(schema))
