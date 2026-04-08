# Consuming the research stream (SSE)

The **POST /research/stream** endpoint returns a Server-Sent Events (SSE) stream. Each event is a single line of the form:

```
data: <JSON object>\n\n
```

Parse each `data:` line as JSON to get the event payload.

---

## Endpoint

- **URL:** `POST /research/stream`
- **Body:** Same as `POST /research`: `{ "topic": "Your research question", "search_api": null }`
- **Response:** `Content-Type: text/event-stream`; body is a stream of SSE events.

---

## Event types

| `type` | Description | Payload (typical fields) |
|--------|-------------|---------------------------|
| `status` | Generic status message | `message` |
| `todo_list` | Initial list of planned tasks | `tasks` (array), `step` (0) |
| `task_status` | Task state change | `task_id`, `status`, `title`, `intent`, `summary?`, `sources_summary?`, `step`, `stream_token` |
| `sources` | Search results for a task | `task_id`, `latest_sources`, `raw_context`, `step`, `backend` |
| `task_summary_chunk` | Streaming summary text for a task | `task_id`, `content`, `step` |
| `final_report` | Final markdown report | `report` |
| `done` | Stream finished | (no extra fields) |
| `error` | Error during stream | `detail` |

Task `status` values: `researching`, `completed`, `skipped`, `failed`.

---

## Client examples

### 1. Fetch + ReadableStream (recommended for POST)

The endpoint uses **POST**, so use `fetch()` and read the response body as a stream. Parse SSE lines and then JSON for each `data:` line.

```javascript
async function runResearchStream(topic) {
  const response = await fetch('http://localhost:8000/research/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ topic }),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const event = JSON.parse(line.slice(6));
          handleEvent(event);
        } catch (e) {
          console.warn('Parse error', e);
        }
      }
    }
  }

  // Process any remaining line in buffer
  if (buffer.startsWith('data: ')) {
    try {
      handleEvent(JSON.parse(buffer.slice(6)));
    } catch (e) {
      console.warn('Parse error', e);
    }
  }
}

function handleEvent(event) {
  switch (event.type) {
    case 'status':
      console.log('Status:', event.message);
      break;
    case 'todo_list':
      console.log('Tasks:', event.tasks?.length);
      break;
    case 'task_status':
      console.log(`Task ${event.task_id} ${event.status}:`, event.title);
      break;
    case 'task_summary_chunk':
      // Append to UI for task event.task_id
      break;
    case 'final_report':
      console.log('Report length:', event.report?.length);
      break;
    case 'done':
      console.log('Stream finished');
      break;
    case 'error':
      console.error('Error:', event.detail);
      break;
    default:
      console.log('Event', event.type, event);
  }
}
```

### 2. curl

```bash
curl -N -X POST http://localhost:8000/research/stream \
  -H "Content-Type: application/json" \
  -d '{"topic": "Latest trends in renewable energy"}' \
  --no-buffer
```

### 3. Python (requests + streaming)

```python
import json
import requests

url = "http://localhost:8000/research/stream"
payload = {"topic": "Latest trends in renewable energy"}

with requests.post(url, json=payload, stream=True) as r:
    r.raise_for_status()
    buffer = ""
    for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
        if chunk:
            buffer += chunk
        while "\n\n" in buffer:
            line, buffer = buffer.split("\n\n", 1)
            for part in line.split("\n"):
                if part.startswith("data: "):
                    event = json.loads(part[6:])
                    print(event.get("type"), event)
```

---

## Notes

- **Keep-alive:** The server sends events as they occur; the connection stays open until `done` or an error.
- **stream_token:** Each task has a `stream_token` (e.g. `task_1`) so the client can map events to the same task when multiple tasks run in parallel.
- **task_summary_chunk:** Append `content` to the current task’s summary in the UI for live streaming text.
- **CORS:** The API allows all origins; for production you may want to restrict this.
