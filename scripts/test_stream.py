import json

import requests


URL = "http://127.0.0.1:8000/chat/stream"

payload = {
    "query": (
        "Quais habilidades de comunicação são importantes "
        "no cuidado em saúde mental?"
    ),
    "model": "azure/gpt-5-mini",
    "limit": 5,
}


def main() -> None:
    response = requests.post(
        URL,
        json=payload,
        stream=True,
        timeout=120,
    )

    response.raise_for_status()

    print("\nStreaming response:\n")

    for line in response.iter_lines():
        if not line:
            continue

        decoded = line.decode("utf-8")

        if not decoded.startswith("data: "):
            continue

        data = json.loads(decoded[6:])

        event_type = data.get("type")

        if event_type == "sources":
            print("\nSources:")
            for source in data["sources"]:
                print(
                    f"- {source['source']} "
                    f"(chunk={source['chunk_id']}, "
                    f"score={source['score']:.4f})"
                )

            print("\nAnswer:\n")

        elif event_type == "text_delta":
            print(
                data["delta"],
                end="",
                flush=True,
            )

        elif event_type == "done":
            print("\n\nStream completed.")

        elif event_type == "error":
            print(
                f"\nError: {data['message']}"
            )


if __name__ == "__main__":
    main()
