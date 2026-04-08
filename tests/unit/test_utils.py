from backend.src.utils import (
    deduplicate_and_format_sources,
    format_sources,
    strip_thinking_tokens,
)


def test_strip_thinking_tokens_basic():
    # Case 1: Standard thinking tokens
    raw_response = "<think>I should search for SpaceX.</think>SpaceX was founded in 2002."
    cleaned = strip_thinking_tokens(raw_response)
    assert cleaned == "SpaceX was founded in 2002."

    # Case 2: Multiple thinking blocks
    multi_block = "<think>One</think>Hello<think>Two</think>World"
    cleaned_multi = strip_thinking_tokens(multi_block)
    assert cleaned_multi == "HelloWorld"

    # Case 3: No tokens
    normal_text = "Just normal text."
    assert strip_thinking_tokens(normal_text) == "Just normal text."


def test_format_sources_and_deduplicate():
    search_results = {
        "results": [
            {
                "title": "First",
                "url": "https://example.com/a",
                "content": "A1",
            },
            {
                # Duplicate URL with different title/content should be deduped.
                "title": "First duplicate",
                "url": "https://example.com/a",
                "content": "A2",
            },
            {
                "title": "Second",
                "url": "https://example.com/b",
                "content": "B",
            },
        ]
    }

    bullets = format_sources(search_results)
    # format_sources does NOT deduplicate; it surfaces each result as-is.
    # We expect two entries for the duplicated URL.
    assert bullets.count("https://example.com/a") == 2
    assert bullets.count("https://example.com/b") == 1

    formatted = deduplicate_and_format_sources(search_results, max_tokens_per_source=10, fetch_full_page=False)
    # Deduplicated content should only mention each URL once.
    assert formatted.count("https://example.com/a") == 1
    assert formatted.count("https://example.com/b") == 1
