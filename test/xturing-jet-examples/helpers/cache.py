import json
import hashlib
import uuid


class InMemoryCache:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InMemoryCache, cls).__new__(cls)
            cls._instance.evaluation_cache = {}
            cls._instance.models_cache = {}
            cls._instance.datasets_cache = {}
            cls._instance.tokenizers_cache = {}
            cls._instance.ngrams_cache = {}
        return cls._instance

    @property
    def evaluation(self):
        return self._instance.evaluation_cache

    @property
    def models(self):
        return self._instance.models_cache

    @property
    def tokenizers(self):
        return self._instance.tokenizers_cache

    @property
    def datasets(self):
        return self._instance.datasets_cache

    @property
    def ngrams(self):
        return self._instance.ngrams_cache

    def setEvaluation(self, key, value):
        self._instance.evaluation_cache[key] = value

    def setModels(self, key, value):
        self._instance.models_cache[key] = value

    def setDatasets(self, key, value):
        self._instance.datasets_cache[key] = value

    def setTokenizers(self, key, value):
        self._instance.tokenizers_cache[key] = value

    def setNGrams(self, key, value):
        self._instance.ngrams_cache[key] = value
