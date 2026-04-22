import io

from PIL import Image



def test_multimodal_search_missing_inputs(client):
    """Verify that omitting both query-text and image fails."""
    response = client.post("/multimodal-search")
    assert response.status_code == 400
    assert "either a text query or an image" in response.json()["detail"]


def test_multimodal_search_invalid_image_type(client):
    """Verify image type validation."""
    response = client.post(
        "/multimodal-search",
        files={"file": ("dummy.txt", b"Hello text", "text/plain")},
    )
    assert response.status_code == 400
    assert "File must be an image." in response.json()["detail"]


def test_multimodal_search_text_only(client):
    """Verify textual multimodel search behavior via API."""
    response = client.post("/multimodal-search", data={"query": "Backpack", "top_k": 2})

    assert response.status_code == 200
    data = response.json()
    assert "query_type" in data
    assert data["query_type"] == "text"
    assert "results" in data
    assert "metadata" in data
    assert len(data["results"]) <= 2


def test_multimodal_search_image_only(client):
    """Verify visual multimodel search behavior via API."""
    # Create fake in-memory image
    img = Image.new("RGB", (200, 200), color="blue")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    response = client.post(
        "/multimodal-search",
        files={"file": ("dummy.png", img_byte_arr, "image/png")},
        data={"top_k": 2},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["query_type"] == "image"
    assert len(data["results"]) <= 2


def test_multimodal_search_late_fusion(client):
    """Verify joint input multimodel search behavior via API."""
    img = Image.new("RGB", (200, 200), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    response = client.post(
        "/multimodal-search",
        data={"query": "Backpack", "top_k": 2},
        files={"file": ("dummy.jpg", img_byte_arr, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["query_type"] == "text+image"
    assert len(data["results"]) <= 2
