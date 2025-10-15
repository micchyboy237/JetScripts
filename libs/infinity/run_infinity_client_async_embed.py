from infinity_client.models import OpenAIModelInfo
from infinity_client.api.default import models
from infinity_client.types import Response
from infinity_client import Client

i_client = Client(base_url="https://infinity.modal.michaelfeil.eu")

async def aembed():
    async with i_client as client:
        model_info: OpenAIModelInfo = await models.asyncio(client=client)
        response: Response[OpenAIModelInfo] = await models.asyncio_detailed(client=client)