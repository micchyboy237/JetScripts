from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings
initialize_ollama_settings()

@collections_router.post("/create")
async def create_collection(
    request,
    title: str = Form(...),
    description: str = Form(...),
    files: list[UploadedFile] = File(...),
):
    key = None if getattr(request, "auth", None) is None else request.auth
    if key is not None:
        key = await key

    collection_instance = Collection(
        api_key=key,
        title=title,
        description=description,
        status=CollectionStatusEnum.QUEUED,
    )

    await sync_to_async(collection_instance.save)()

    for uploaded_file in files:
        doc_data = uploaded_file.file.read()
        doc_file = ContentFile(doc_data, uploaded_file.name)
        document = Document(collection=collection_instance, file=doc_file)
        await sync_to_async(document.save)()

    create_index.si(collection_instance.id).apply_async()

    return await sync_to_async(CollectionModelSchema)(...)

@collections_router.post(
    "/query",
    response=CollectionQueryOutput,
    summary="Ask a question of a document collection",
)
def query_collection_view(
    request: HttpRequest, query_input: CollectionQueryInput
):
    collection_id = query_input.collection_id
    query_str = query_input.query_str
    response = query_collection(collection_id, query_str)
    return {"response": response}

@collections_router.get(
    "/available",
    response=list[CollectionModelSchema],
    summary="Get a list of all of the collections created with my api_key",
)
async def get_my_collections_view(request: HttpRequest):
    key = None if getattr(request, "auth", None) is None else request.auth
    if key is not None:
        key = await key

    collections = Collection.objects.filter(api_key=key)

    return [{...} async for collection in collections]

@collections_router.post(
    "/{collection_id}/add_file", summary="Add a file to a collection"
)
async def add_file_to_collection(
    request,
    collection_id: int,
    file: UploadedFile = File(...),
    description: str = Form(...),
):
    collection = await sync_to_async(Collection.objects.get)(id=collection_id)

application = ProtocolTypeRouter(
    {
        "http": get_asgi_application(),
        "websocket": TokenAuthMiddleware(
            URLRouter(
                [
                    re_path(
                        r"ws/collections/(?P<collection_id>\w+)/query/$",
                        CollectionQueryConsumer.as_asgi(),
                    ),
                ]
            )
        ),
    }
)

async def connect(self):
    try:
        self.collection_id = extract_connection_id(self.scope["path"])
        self.index = await load_collection_model(self.collection_id)
        await self.accept()

    except ValueError as e:
        await self.accept()
        await self.close(code=4000)
    except Exception as e:
        pass

async def receive(self, text_data):
    text_data_json = json.loads(text_data)

    if self.index is not None:
        query_str = text_data_json["query"]
        modified_query_str = f"Please return a nicely formatted markdown string to this request:\n\n{query_str}"
        query_engine = self.index.as_query_engine()
        response = query_engine.query(modified_query_str)

        markdown_response = f"## Response\n\n{response}\n\n"
        if response.source_nodes:
            markdown_sources = (
                f"## Sources\n\n{response.get_formatted_sources()}"
            )
        else:
            markdown_sources = ""

        formatted_response = f"{markdown_response}{markdown_sources}"

        await self.send(json.dumps({"response": formatted_response}, indent=4))
    else:
        await self.send(
            json.dumps(
                {"error": "No index loaded for this connection."}, indent=4
            )
        )

from llama_index.core import Settings


async def load_collection_model(collection_id: str | int) -> VectorStoreIndex:
    """
    Load the Collection model from cache or the database, and return the index.

    Args:
        collection_id (Union[str, int]): The ID of the Collection model instance.

    Returns:
        VectorStoreIndex: The loaded index.

    This function performs the following steps:
    1. Retrieve the Collection object with the given collection_id.
    2. Check if a JSON file with the name '/cache/model_{collection_id}.json' exists.
    3. If the JSON file doesn't exist, load the JSON from the Collection.model FileField and save it to
       '/cache/model_{collection_id}.json'.
    4. Call VectorStoreIndex.load_from_disk with the cache_file_path.
    """
    collection = await Collection.objects.aget(id=collection_id)
    logger.info(f"load_collection_model() - loaded collection {collection_id}")

    if collection.model.name:
        logger.info("load_collection_model() - Setup local json index file")

        cache_dir = Path(settings.BASE_DIR) / "cache"
        cache_file_path = cache_dir / f"model_{collection_id}.json"
        if not cache_file_path.exists():
            cache_dir.mkdir(parents=True, exist_ok=True)
            with collection.model.open("rb") as model_file:
                with cache_file_path.open(
                    "w+", encoding="utf-8"
                ) as cache_file:
                    cache_file.write(model_file.read().decode("utf-8"))

        logger.info(
            f"load_collection_model() - Setup Settings with tokens {settings.MAX_TOKENS} and "
            f"model {settings.MODEL_NAME}"
        )
        Settings.llm = Ollama(
            temperature=0, model="llama3.2", request_timeout=300.0, context_window=4096, max_tokens=512
        )

        logger.info("load_collection_model() - Load llama index")
        index = VectorStoreIndex.load_from_disk(
            cache_file_path,
        )
        logger.info(
            "load_collection_model() - Llamaindex loaded and ready for query..."
        )

    else:
        logger.error(
            f"load_collection_model() - collection {collection_id} has no model!"
        )
        raise ValueError("No model exists for this collection!")

    return index

const [collections, setCollections] = useState<CollectionModelSchema[]>([]);
const [loading, setLoading] = useState(true);

const
fetchCollections = async () = > {
try {
const accessToken = localStorage.getItem("accessToken");
if (accessToken) {
const response = await getMyCollections(accessToken);
setCollections(response.data);
}
} catch (error) {
console.error(error);
} finally {
setLoading(false);
}
};

< List >
{collections.map((collection) = > (
    < div key={collection.id} >
    < ListItem disablePadding >
    < ListItemButton
    disabled={
    collection.status != = CollectionStatus.COMPLETE | |
    !collection.has_model
    }
    onClick={() = > handleCollectionClick(collection)}
selected = {
    selectedCollection & &
    selectedCollection.id == = collection.id
}
>
< ListItemText
primary = {collection.title} / >
          {collection.status == = CollectionStatus.RUNNING ? (
    < CircularProgress
    size={24}
    style={{position: "absolute", right: 16}}
    / >
): null}
< / ListItemButton >
    < / ListItem >
        < / div >
))}
< / List >

useEffect(() = > {
    let
interval: NodeJS.Timeout;
if (
    collections.some(
        (collection) = >
collection.status == = CollectionStatus.RUNNING | |
collection.status == = CollectionStatus.QUEUED
)
) {
    interval = setInterval(() = > {
    fetchCollections();
}, 15000);
}
return () = > clearInterval(interval);
}, [collections]);

const setupWebsocket = () => {
  setConnecting(true);
  // Here, a new WebSocket object is created using the specified URL, which includes the
  // selected collection's ID and the user's authentication token.

  websocket.current = new WebSocket(
    `ws://localhost:8000/ws/collections/${selectedCollection.id}/query/?token=${authToken}`,
  );

  websocket.current.onopen = (event) => {
    //...
  };

  websocket.current.onmessage = (event) => {
    //...
  };

  websocket.current.onclose = (event) => {
    //...
  };

  websocket.current.onerror = (event) => {
    //...
  };

  return () => {
    websocket.current?.close();
  };
};

websocket.current.onopen = (event) => {
  setError(false);
  setConnecting(false);
  setAwaitingMessage(false);

  console.log("WebSocket connected:", event);
};

# websocket.current.onmessage = (event) => {
#   const data = JSON.parse(event.data);
#   console.log("WebSocket message received:", data);
#   setAwaitingMessage(false);
# 
#   if (data.response) {
#     // Update the messages state with the new message from the server
#     setMessages((prevMessages) => [
#       ...prevMessages,
#       {
#         sender_id: "server",
#         message: data.response,
#         timestamp: new Date().toLocaleTimeString(),
#       },
#     ]);
#   }
# };

websocket.current.onclose = (event) => {
  if (event.code === 4000) {
    toast.warning(
      "Selected collection's model is unavailable. Was it created properly?",
    );
    setError(true);
    setConnecting(false);
    setAwaitingMessage(false);
  }
  console.log("WebSocket closed:", event);
};

websocket.current.onerror = (event) => {
  setError(true);
  setConnecting(false);
  setAwaitingMessage(false);

  console.error("WebSocket error:", event);
};

git clone https://github.com/yourusername/delphic.git

cd delphic

mkdir -p ./.envs/.local/
cp -a ./docs/sample_envs/local/.frontend ./frontend
cp -a ./docs/sample_envs/local/.django ./.envs/.local
cp -a ./docs/sample_envs/local/.postgres ./.envs/.local

sudo docker-compose --profiles fullstack -f local.yml build

sudo docker-compose -f local.yml up

logger.info("\n\n[DONE]", bright=True)