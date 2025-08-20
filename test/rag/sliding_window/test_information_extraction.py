import pytest
from information_extraction import extract_information

@pytest.fixture
def sample_text():
    return "Apple released the iPhone 16. Google launched Pixel 9." * 10

class TestInformationExtraction:
    def test_extract_information(self, sample_text, mocker):
        # Given: A sample text and entity type
        entity_type = "products"
        expected_entities = ["iPhone 16", "Pixel 9"]
        mocker.patch("mlx_lm.generate", return_value="iPhone 16, Pixel 9")

        # When: Extracting information
        result = extract_information(sample_text, entity_type)

        # Then: Verify extracted entities
        assert len(result) > 0
        assert result[0]["entities"] == expected_entities