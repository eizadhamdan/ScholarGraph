# ScholarGraph Frontend

The frontend is a standalone React + TypeScript single-page application built with Vite. It presents the ScholarGraph research agent as a chat workspace and communicates with the Python backend through HTTP; it does not connect directly to Gemini, Neo4j, or ChromaDB.

For the repository architecture and GraphRAG concepts, see the [root README](../README.md). For backend installation, configuration, and endpoint details, see the [backend README](../backend/README.md).

## Requirements

- Node.js 20.19+ or 22.12+ (Node.js 24 is also supported by the current Vite release)
- npm
- The FastAPI backend available at `http://127.0.0.1:8000` for chat functionality

## Install and Run

From this folder:

```bash
npm ci
npm run dev
```

Open <http://localhost:5173>. The Vite development server proxies `/api` requests to `http://127.0.0.1:8000`; start the backend separately as described in the backend guide.

## API Configuration

For local development, the Vite proxy handles the API connection. For a separately hosted backend, create a frontend `.env.local` file and set:

```dotenv
VITE_API_URL=https://your-backend.example.com
```

Set the backend's `FRONTEND_ORIGIN` environment variable to the exact browser origin so FastAPI CORS allows the frontend. Do not put Gemini or Neo4j secrets in frontend environment variables; Vite variables are exposed to browser code.

The frontend uses:

- `GET /api/health` to show API availability and whether Neo4j is reachable.
- `GET /api/conversations` and `GET /api/conversations/{id}` to list and load stored chats.
- `POST /api/conversations` and `DELETE /api/conversations/{id}` to create and remove chats.
- `POST /api/chat` with `{ message, conversation_id }` to send a question; history is loaded by the backend from SQLite.

The health indicator distinguishes an unavailable backend from an unavailable knowledge graph. If Neo4j cannot be reached, the UI displays a warning and chat requests return an explicit error instead of generating an answer from vector results alone. Health checks do not verify Gemini credentials or ChromaDB readiness.

Each saved assistant response includes an expandable **Retrieval evidence** section with each generated/fallback Cypher query and row count, rejected-query details, fallback search terms, returned graph rows, and vector-search paper IDs, titles, dates, and excerpts. Findings cite only IDs returned by those sources. This allows you to inspect what was retrieved; it does not by itself prove that a passage entails a generated claim, so verify important conclusions against the original papers.

## Features and Data

- Conversation list with create/delete controls
- Prompt suggestions, multiline composer, and loading/error states
- Responsive layout for desktop and mobile
- Recent turns loaded from backend SQLite for contextual follow-up questions
- Conversation history persisted in the backend's local SQLite database

All browser clients connected to one backend share its conversation database; the current application does not include user accounts or per-user isolation. The current client waits for a complete answer; it does not use streaming responses.

## Build

```bash
npm run build
npm run preview
```

The production bundle is written to `dist/`. `npm run build` runs the TypeScript project build and Vite production build.

## Source Layout

- `src/App.tsx`: chat workspace, conversation state, and user interactions.
- `src/lib/api.ts`: typed fetch client for health and chat endpoints.
- `src/main.tsx`: React application entry point.
- `src/styles.css`: responsive visual system and component styling.
- `vite.config.ts`: React plugin and local backend proxy.
