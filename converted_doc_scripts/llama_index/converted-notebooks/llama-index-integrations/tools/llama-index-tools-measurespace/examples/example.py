async def main():
    from dotenv import load_dotenv
    from jet.adapters.llama_index.ollama_function_calling import OllamaFunctionCalling
    from jet.logger import CustomLogger
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.tools.measurespace import MeasureSpaceToolSpec
    import os
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    load_dotenv()

    api_keys = {
        'hourly_weather': os.getenv('HOURLY_WEATHER_API_KEY'),
        'daily_weather': os.getenv('DAILY_WEATHER_API_KEY'),
        'daily_climate': os.getenv('DAILY_CLIMATE_API_KEY'),
        'air_quality': os.getenv('AIR_QUALITY_API_KEY'),
        'geocoding': os.getenv('GEOCODING_API_KEY'),
    }

    tool_spec = MeasureSpaceToolSpec(api_keys)

    for tool in tool_spec.to_tool_list():
        logger.debug(tool.metadata.name)

    tool_spec.get_daily_weather_forecast('New York')

    tool_spec.get_latitude_longitude_from_location('New York')

    agent = FunctionAgent(
        tools=tool_spec.to_tool_list(),
        llm=OllamaFunctionCalling(model="llama3.2"),
    )

    logger.debug(
        await agent.run("How's the temperature for New York in next 3 days?")
    )
    logger.debug(
        await agent.run("What's the latitude and longitude of New York?")
    )

    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())
